from typing import Iterable, Sequence

import numpy as np

import bpy


__all__ = [
    "build_vat_images_from_tracks",
    "create_vat_animation_from_tracks",
    "create_vat_animation_from_images",
    "update_vat_animation_for_object",
]


def ms_to_frame(ms: float, fps: float) -> float:
    return (ms / 1000.0) * fps


def _row_frame(row: dict, fps: float) -> float:
    if "frame" in row:
        try:
            return float(row.get("frame", 0.0))
        except Exception:
            return 0.0
    try:
        return ms_to_frame(float(row.get("t_ms", 0.0)), fps)
    except Exception:
        return 0.0


def _gather_samples(
    tracks: Sequence[dict], fps: float, frame_count: int
) -> list[dict[str, np.ndarray]]:
    target_frames = np.arange(frame_count, dtype=np.float32)
    samples: list[dict[str, np.ndarray]] = []

    for track in tracks:
        data = track.get("data") or []
        if not data:
            zeros = np.zeros(frame_count, dtype=np.float32)
            samples.append({key: zeros for key in ("x", "y", "z", "r", "g", "b")})
            continue

        times = np.array([_row_frame(row, fps) for row in data], dtype=np.float32)

        def _interp(values: np.ndarray) -> np.ndarray:
            return np.interp(
                target_frames, times, values, left=values[0], right=values[-1]
            )

        samples.append(
            {
                key: _interp(
                    np.array([row.get(key, 0.0) for row in data], dtype=np.float32)
                )
                for key in ("x", "y", "z", "r", "g", "b")
            }
        )

    return samples


def _determine_bounds(samples: Iterable[dict[str, np.ndarray]]):
    arrays = list(samples)
    if not arrays:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    xs = np.concatenate([track["x"] for track in arrays])
    ys = np.concatenate([track["y"] for track in arrays])
    zs = np.concatenate([track["z"] for track in arrays])

    return (
        float(xs.min(initial=0.0)),
        float(ys.min(initial=0.0)),
        float(zs.min(initial=0.0)),
    ), (
        float(xs.max(initial=1.0)),
        float(ys.max(initial=1.0)),
        float(zs.max(initial=1.0)),
    )


def _create_image(name: str, width: int, height: int, fb: bool):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    return bpy.data.images.new(name=name, width=width, height=height, float_buffer=fb)


def _normalize_color_values(values: np.ndarray) -> np.ndarray:
    """Normalize color channel values to the 0-1 range."""

    normalized = np.where(values > 1.0, values / 255.0, values)
    return np.clip(normalized, 0.0, 1.0)


def build_vat_images_from_tracks(
    tracks: Sequence[dict], fps: float, *, image_name_prefix: str = "VAT"
):
    if not tracks:
        raise RuntimeError("No CSV tracks supplied for VAT generation")

    # Normalize frames to start at the earliest sample so VAT starts at the render range
    min_frame = min(
        (_row_frame(tr["data"][0], fps) for tr in tracks if tr.get("data")),
        default=0.0,
    )
    adjusted_tracks = []
    for tr in tracks:
        data = tr.get("data") or []
        if not data:
            adjusted_tracks.append({"name": tr.get("name", ""), "data": []})
            continue
        adjusted = []
        for row in data:
            frame = _row_frame(row, fps) - float(min_frame)
            adjusted.append({**row, "frame": frame})
        adjusted_tracks.append({"name": tr.get("name", ""), "data": adjusted})

    max_frame = max(
        (tr["data"][-1]["frame"] for tr in adjusted_tracks if tr["data"]),
        default=0.0,
    )
    duration = int(max_frame)
    frame_count = max(duration + 1, 1)
    samples = _gather_samples(adjusted_tracks, fps, frame_count)
    pos_min, pos_max = _determine_bounds(samples)

    drone_count = len(tracks)
    prefix = image_name_prefix or "VAT"
    pos_img = _create_image(f"{prefix}_Pos", frame_count, drone_count, True)
    col_img = _create_image(f"{prefix}_Color", frame_count, drone_count, False)
    pos_img.colorspace_settings.name = "Non-Color"
    
    rx = (pos_max[0] - pos_min[0]) or 1.0
    ry = (pos_max[1] - pos_min[1]) or 1.0
    rz = (pos_max[2] - pos_min[2]) or 1.0

    pos_pixels = np.empty((drone_count, frame_count, 4), dtype=np.float32)
    col_pixels = np.empty((drone_count, frame_count, 4), dtype=np.float32)

    pos_pixels[:, :, 3] = 1.0
    col_pixels[:, :, 3] = 1.0

    for drone_idx, track in enumerate(samples):
        pos_pixels[drone_idx, :, 0] = (track["x"] - pos_min[0]) / rx
        pos_pixels[drone_idx, :, 1] = (track["y"] - pos_min[1]) / ry
        pos_pixels[drone_idx, :, 2] = (track["z"] - pos_min[2]) / rz

        col_pixels[drone_idx, :, 0] = (track["r"])/ 255.0
        col_pixels[drone_idx, :, 1] = (track["g"])/ 255.0
        col_pixels[drone_idx, :, 2] = (track["b"])/ 255.0

    pos_img.pixels[:] = pos_pixels.ravel()
    col_img.pixels[:] = col_pixels.ravel()

    return pos_img, col_img, pos_min, pos_max, duration, drone_count


def _remove_object_and_mesh(obj):
    """Remove ``obj`` and its mesh data, tolerating already-removed datablocks."""
    if obj is None:
        return
    try:
        mesh_data = getattr(obj, "data", None)
    except ReferenceError:
        return
    if mesh_data is not None:
        try:
            bpy.data.meshes.remove(mesh_data)
        except ReferenceError:
            pass
    try:
        bpy.data.objects.remove(obj)
    except ReferenceError:
        pass


def _create_drone_points_object(drone_count: int, base_name: str, first_positions):
    obj = bpy.data.objects.get(base_name)
    if obj is not None:
        _remove_object_and_mesh(obj)

    mesh = bpy.data.meshes.new(base_name + "_Mesh")
    verts = first_positions or [(0.0, 0.0, 0.0)] * drone_count
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(base_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _create_gn_vat_group(
    pos_img,
    pos_min,
    pos_max,
    frame_count,
    drone_count,
    *,
    start_frame: int | None,
    base_name: str,
):
    group_name = f"GN_DroneVAT_{base_name}"
    existing = bpy.data.node_groups.get(group_name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    ng = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    iface = ng.interface

    geo_in = iface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
        description="Input geometry to be displaced by VAT textures",
    )
    posmin_in = iface.new_socket(
        name="Pos Min",
        in_out="INPUT",
        socket_type="NodeSocketVector",
        description="Minimum XYZ values encoded in the VAT position texture",
    )
    posmax_in = iface.new_socket(
        name="Pos Max",
        in_out="INPUT",
        socket_type="NodeSocketVector",
        description="Maximum XYZ values encoded in the VAT position texture",
    )
    startframe_in = iface.new_socket(
        name="Start Frame",
        in_out="INPUT",
        socket_type="NodeSocketFloat",
        description="Scene frame at which the VAT animation starts",
    )
    framecount_in = iface.new_socket(
        name="Frame Count",
        in_out="INPUT",
        socket_type="NodeSocketFloat",
        description="Number of frames contained in the VAT textures",
    )
    dronecount_in = iface.new_socket(
        name="Drone Count",
        in_out="INPUT",
        socket_type="NodeSocketInt",
        description="Total number of drones encoded along the V axis",
    )
    geo_out = iface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
        description="Geometry with VAT-driven positions and colors applied",
    )

    posmin_in.default_value = pos_min
    posmax_in.default_value = pos_max
    startframe_in.default_value = float(start_frame) if start_frame is not None else 0.0
    framecount_in.default_value = float(frame_count)
    dronecount_in.default_value = int(drone_count)

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    n_input = nodes.new("NodeGroupInput")
    n_input.location = (-900, 0)
    n_output = nodes.new("NodeGroupOutput")
    n_output.location = (500, 0)

    n_time = nodes.new("GeometryNodeInputSceneTime")
    n_time.location = (-700, 200)
    n_index = nodes.new("GeometryNodeInputIndex")
    n_index.location = (-700, -50)

    n_sub = nodes.new("ShaderNodeMath")
    n_sub.operation = "SUBTRACT"
    n_sub.location = (-500, 200)

    n_div = nodes.new("ShaderNodeMath")
    n_div.operation = "DIVIDE"
    n_div.use_clamp = True
    n_div.location = (-300, 200)

    n_fc_minus1 = nodes.new("ShaderNodeMath")
    n_fc_minus1.operation = "SUBTRACT"
    n_fc_minus1.location = (-500, 50)

    n_dc_minus1 = nodes.new("ShaderNodeMath")
    n_dc_minus1.operation = "SUBTRACT"
    n_dc_minus1.location = (-500, -250)

    n_div_index = nodes.new("ShaderNodeMath")
    n_div_index.operation = "DIVIDE"
    n_div_index.use_clamp = True
    n_div_index.location = (-300, -250)

    n_combine_uv = nodes.new("ShaderNodeCombineXYZ")
    n_combine_uv.location = (-100, 0)

    n_tex_pos = nodes.new("GeometryNodeImageTexture")
    n_tex_pos.location = (100, 150)
    n_tex_pos.interpolation = "Closest"
    n_tex_pos.extension = "EXTEND"
    n_tex_pos.inputs["Image"].default_value = pos_img

    n_vsub = nodes.new("ShaderNodeVectorMath")
    n_vsub.operation = "SUBTRACT"
    n_vsub.location = (300, 250)

    n_vmul = nodes.new("ShaderNodeVectorMath")
    n_vmul.operation = "MULTIPLY"
    n_vmul.location = (500, 150)

    n_vadd = nodes.new("ShaderNodeVectorMath")
    n_vadd.operation = "ADD"
    n_vadd.location = (700, 150)

    n_setpos = nodes.new("GeometryNodeSetPosition")
    n_setpos.location = (900, 100)

    links.new(n_input.outputs["Geometry"], n_setpos.inputs["Geometry"])
    links.new(n_setpos.outputs["Geometry"], n_output.inputs["Geometry"])

    links.new(n_time.outputs["Frame"], n_sub.inputs[0])
    links.new(n_input.outputs["Start Frame"], n_sub.inputs[1])

    links.new(n_input.outputs["Frame Count"], n_fc_minus1.inputs[0])
    n_fc_minus1.inputs[1].default_value = 1.0

    links.new(n_sub.outputs[0], n_div.inputs[0])
    links.new(n_fc_minus1.outputs[0], n_div.inputs[1])

    links.new(n_div.outputs[0], n_combine_uv.inputs[0])

    links.new(n_input.outputs["Drone Count"], n_dc_minus1.inputs[0])
    n_dc_minus1.inputs[1].default_value = 1.0

    links.new(n_index.outputs["Index"], n_div_index.inputs[0])
    links.new(n_dc_minus1.outputs[0], n_div_index.inputs[1])

    links.new(n_div_index.outputs[0], n_combine_uv.inputs[1])

    links.new(n_combine_uv.outputs["Vector"], n_tex_pos.inputs["Vector"])

    links.new(n_input.outputs["Pos Max"], n_vsub.inputs[0])
    links.new(n_input.outputs["Pos Min"], n_vsub.inputs[1])

    links.new(n_tex_pos.outputs["Color"], n_vmul.inputs[0])
    links.new(n_vsub.outputs["Vector"], n_vmul.inputs[1])

    links.new(n_input.outputs["Pos Min"], n_vadd.inputs[0])
    links.new(n_vmul.outputs["Vector"], n_vadd.inputs[1])

    links.new(n_vadd.outputs["Vector"], n_setpos.inputs["Position"])

    return ng


def _apply_gn_to_object(obj, node_group):
    for m in list(obj.modifiers):
        if m.type == "NODES":
            obj.modifiers.remove(m)
    mod = obj.modifiers.new(name="Drone VAT", type="NODES")
    mod.node_group = node_group
    return mod


def update_vat_animation_for_object(
    obj,
    tracks: Sequence[dict],
    fps: float,
    *,
    start_frame: int,
    base_name: str,
    storyboard_name: str | None = None,
):
    """Regenerate VAT textures and reapply the GN modifier on ``obj``."""

    if obj is None:
        return None, None, 0, 0

    image_name_prefix = f"{storyboard_name}_VAT" if storyboard_name else "VAT"
    pos_img, col_img, pos_min, pos_max, duration, drone_count = build_vat_images_from_tracks(
        tracks, fps, image_name_prefix=image_name_prefix
    )

    node_group = _create_gn_vat_group(
        pos_img,
        pos_min,
        pos_max,
        duration + 1,
        drone_count,
        start_frame=start_frame,
        base_name=base_name,
    )
    _apply_gn_to_object(obj, node_group)

    return col_img, pos_img, duration, drone_count


def create_vat_animation_from_tracks(
    tracks: Sequence[dict],
    fps: float,
    *,
    start_frame: int,
    base_name: str,
    storyboard_name: str | None = None,
):
    if not tracks:
        return None, None

    image_name_prefix = f"{storyboard_name}_VAT" if storyboard_name else "VAT"
    pos_img, col_img, pos_min, pos_max, duration, drone_count = build_vat_images_from_tracks(
        tracks, fps, image_name_prefix=image_name_prefix
    )

    first_positions = []
    for tr in tracks:
        if tr.get("data"):
            d0 = tr["data"][0]
            first_positions.append((d0["x"], d0["y"], d0["z"]))
    while len(first_positions) < drone_count:
        first_positions.append((0.0, 0.0, 0.0))

    obj = _create_drone_points_object(drone_count, base_name, first_positions)
    node_group = _create_gn_vat_group(
        pos_img,
        pos_min,
        pos_max,
        duration + 1,
        drone_count,
        start_frame=start_frame,
        base_name=base_name,
    )
    _apply_gn_to_object(obj, node_group)

    return obj, col_img, pos_img


def create_vat_animation_from_images(
    pos_img,
    col_img,
    pos_min,
    pos_max,
    *,
    start_frame: int,
    base_name: str,
):
    frame_count = int(getattr(pos_img, "size", (0, 0))[0] or 0)
    drone_count = int(getattr(pos_img, "size", (0, 0))[1] or 0)

    try:
        pixels = np.array(pos_img.pixels[:], dtype=np.float32)
        pixels = pixels.reshape(drone_count, frame_count, 4)
        rx = (pos_max[0] - pos_min[0]) or 1.0
        ry = (pos_max[1] - pos_min[1]) or 1.0
        rz = (pos_max[2] - pos_min[2]) or 1.0
        first_positions = [
            (
                pos_min[0] + float(pixels[row, 0, 0]) * rx,
                pos_min[1] + float(pixels[row, 0, 1]) * ry,
                pos_min[2] + float(pixels[row, 0, 2]) * rz,
            )
            for row in range(drone_count)
        ]
    except Exception:
        first_positions = None

    obj = _create_drone_points_object(drone_count, base_name, first_positions)
    node_group = _create_gn_vat_group(
        pos_img,
        pos_min,
        pos_max,
        frame_count,
        drone_count,
        start_frame=start_frame,
        base_name=base_name,
    )
    _apply_gn_to_object(obj, node_group)

    return obj
