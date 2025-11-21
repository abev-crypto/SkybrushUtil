from typing import Iterable, Sequence

import bpy
from mathutils import Vector


__all__ = [
    "build_vat_images_from_tracks",
    "create_vat_animation_from_tracks",
]


POS_IMAGE_NAME = "VAT_Pos"
COL_IMAGE_NAME = "VAT_Color"


def ms_to_frame(ms: float, fps: float) -> float:
    return (ms / 1000.0) * fps


def _sample_row(track: Sequence[dict], target_ms: float) -> dict:
    if not track:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "g": 0.0, "b": 0.0}
    if target_ms <= track[0]["t_ms"]:
        return track[0]
    for i in range(1, len(track)):
        prev_row = track[i - 1]
        next_row = track[i]
        if target_ms <= next_row["t_ms"]:
            span = max(next_row["t_ms"] - prev_row["t_ms"], 1.0)
            factor = (target_ms - prev_row["t_ms"]) / span
            return {
                key: prev_row[key] * (1.0 - factor) + next_row[key] * factor
                for key in ("x", "y", "z", "r", "g", "b")
            }
    return track[-1]


def _gather_samples(tracks: Sequence[dict], fps: float, frame_count: int) -> list[list[dict]]:
    samples: list[list[dict]] = []
    for track in tracks:
        frames: list[dict] = []
        for frame in range(frame_count):
            target_ms = (frame / fps) * 1000.0
            frames.append(_sample_row(track["data"], target_ms))
        samples.append(frames)
    return samples


def _determine_bounds(samples: Iterable[Iterable[dict]]):
    xs, ys, zs = [], [], []
    for track in samples:
        for row in track:
            xs.append(row.get("x", 0.0))
            ys.append(row.get("y", 0.0))
            zs.append(row.get("z", 0.0))
    if not xs:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _create_image(name: str, width: int, height: int):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    return bpy.data.images.new(name=name, width=width, height=height, float_buffer=True)


def build_vat_images_from_tracks(tracks: Sequence[dict], fps: float):
    if not tracks:
        raise RuntimeError("No CSV tracks supplied for VAT generation")

    max_t_ms = max(tr["data"][-1]["t_ms"] for tr in tracks if tr["data"])
    duration = int(ms_to_frame(max_t_ms, fps))
    frame_count = max(duration + 1, 1)
    samples = _gather_samples(tracks, fps, frame_count)
    pos_min, pos_max = _determine_bounds(samples)

    drone_count = len(tracks)
    pos_img = _create_image(POS_IMAGE_NAME, frame_count, drone_count)
    col_img = _create_image(COL_IMAGE_NAME, frame_count, drone_count)

    rx = (pos_max[0] - pos_min[0]) or 1.0
    ry = (pos_max[1] - pos_min[1]) or 1.0
    rz = (pos_max[2] - pos_min[2]) or 1.0

    pos_pixels = [0.0] * (frame_count * drone_count * 4)
    col_pixels = [0.0] * (frame_count * drone_count * 4)

    for drone_idx, track in enumerate(samples):
        for frame_idx, row in enumerate(track):
            nx = (row["x"] - pos_min[0]) / rx
            ny = (row["y"] - pos_min[1]) / ry
            nz = (row["z"] - pos_min[2]) / rz

            cr = row["r"] / 255.0 if row["r"] > 1.0 else row["r"]
            cg = row["g"] / 255.0 if row["g"] > 1.0 else row["g"]
            cb = row["b"] / 255.0 if row["b"] > 1.0 else row["b"]

            idx = (drone_idx * frame_count + frame_idx) * 4
            pos_pixels[idx + 0] = nx
            pos_pixels[idx + 1] = ny
            pos_pixels[idx + 2] = nz
            pos_pixels[idx + 3] = 1.0

            col_pixels[idx + 0] = cr
            col_pixels[idx + 1] = cg
            col_pixels[idx + 2] = cb
            col_pixels[idx + 3] = 1.0

    pos_img.pixels[:] = pos_pixels
    col_img.pixels[:] = col_pixels

    return pos_img, col_img, pos_min, pos_max, duration, drone_count


def _create_drone_points_object(drone_count: int, base_name: str, first_positions):
    obj = bpy.data.objects.get(base_name)
    if obj is not None:
        if obj.data:
            bpy.data.meshes.remove(obj.data)
        bpy.data.objects.remove(obj)

    mesh = bpy.data.meshes.new(base_name + "_Mesh")
    verts = first_positions or [(0.0, 0.0, 0.0)] * drone_count
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(base_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _create_gn_vat_group(
    pos_img,
    col_img,
    pos_min,
    pos_max,
    frame_count,
    drone_count,
    *,
    start_frame: int,
    base_name: str,
):
    group_name = f"GN_DroneVAT_{base_name}"
    existing = bpy.data.node_groups.get(group_name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    ng = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    iface = ng.interface

    geo_in = iface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    posmin_in = iface.new_socket(
        name="Pos Min", in_out="INPUT", socket_type="NodeSocketVector"
    )
    posmax_in = iface.new_socket(
        name="Pos Max", in_out="INPUT", socket_type="NodeSocketVector"
    )
    startframe_in = iface.new_socket(
        name="Start Frame", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    framecount_in = iface.new_socket(
        name="Frame Count", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    dronecount_in = iface.new_socket(
        name="Drone Count", in_out="INPUT", socket_type="NodeSocketInt"
    )
    geo_out = iface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    posmin_in.default_value = pos_min
    posmax_in.default_value = pos_max
    startframe_in.default_value = float(start_frame)
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
    n_tex_pos.interpolation = "Linear"
    n_tex_pos.extension = "EXTEND"
    n_tex_pos.inputs["Image"].default_value = pos_img

    n_tex_col = nodes.new("GeometryNodeImageTexture")
    n_tex_col.location = (100, -50)
    n_tex_col.interpolation = "Linear"
    n_tex_col.extension = "EXTEND"
    n_tex_col.inputs["Image"].default_value = col_img

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

    n_store_col = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store_col.location = (1100, 0)
    n_store_col.data_type = "FLOAT_COLOR"
    n_store_col.domain = "POINT"
    n_store_col.name = "color"

    links.new(n_input.outputs["Geometry"], n_setpos.inputs["Geometry"])
    links.new(n_setpos.outputs["Geometry"], n_store_col.inputs["Geometry"])
    links.new(n_store_col.outputs["Geometry"], n_output.inputs["Geometry"])

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
    links.new(n_combine_uv.outputs["Vector"], n_tex_col.inputs["Vector"])

    links.new(n_input.outputs["Pos Max"], n_vsub.inputs[0])
    links.new(n_input.outputs["Pos Min"], n_vsub.inputs[1])

    links.new(n_tex_pos.outputs["Color"], n_vmul.inputs[0])
    links.new(n_vsub.outputs["Vector"], n_vmul.inputs[1])

    links.new(n_input.outputs["Pos Min"], n_vadd.inputs[0])
    links.new(n_vmul.outputs["Vector"], n_vadd.inputs[1])

    links.new(n_vadd.outputs["Vector"], n_setpos.inputs["Position"])

    links.new(n_tex_col.outputs["Color"], n_store_col.inputs["Value"])

    return ng


def _apply_gn_to_object(obj, node_group):
    for m in list(obj.modifiers):
        if m.type == "NODES":
            obj.modifiers.remove(m)
    mod = obj.modifiers.new(name="Drone VAT", type="NODES")
    mod.node_group = node_group
    return mod


def create_vat_animation_from_tracks(
    tracks: Sequence[dict],
    fps: float,
    *,
    start_frame: int,
    base_name: str,
):
    if not tracks:
        return None

    pos_img, col_img, pos_min, pos_max, duration, drone_count = build_vat_images_from_tracks(
        tracks, fps
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
        col_img,
        pos_min,
        pos_max,
        duration + 1,
        drone_count,
        start_frame=start_frame,
        base_name=base_name,
    )
    _apply_gn_to_object(obj, node_group)

    return obj
