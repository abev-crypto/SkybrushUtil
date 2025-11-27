import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup, UIList
import csv, os, re, math
from mathutils import Vector

from sbutil import csv_vat_gn
from sbutil.light_effects import OUTPUT_VERTEX_COLOR

from sbutil.color_key_utils import apply_color_keys_from_key_data

# ---------- Utilities ----------

def detect_delimiter(sample_path):
    # Try to sniff; fall back to tab if header matches the sample
    try:
        with open(sample_path, "r", newline="") as f:
            head = f.read(2048)
        dialect = csv.Sniffer().sniff(head, delimiters=",\t; ")
        return dialect.delimiter
    except Exception:
        # Heuristic: if header contains tabs or 'Time [msec]\tx [m]' pattern, use '\t'
        if "\t" in head or re.search(r"Time\s*\[msec\]\s*\tx\s*\[m\]", head):
            return "\t"
        return ","  # default

def load_csv(path, delimiter="auto"):
    if delimiter == "auto":
        delimiter = detect_delimiter(path)
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize expected headers
        colmap = {k.strip(): k for k in reader.fieldnames}
        req = ["Time [msec]", "x [m]", "y [m]", "z [m]", "Red", "Green", "Blue"]
        for r in reader:
            row = {
                "t_ms": float(r[colmap["Time [msec]"]]),
                "x": float(r[colmap["x [m]"]]),
                "y": float(r[colmap["y [m]"]]),
                "z": float(r[colmap["z [m]"]]),
                "r": float(r[colmap["Red"]]),
                "g": float(r[colmap["Green"]]),
                "b": float(r[colmap["Blue"]]),
            }
            rows.append(row)
    # Ensure sorted by time
    rows.sort(key=lambda d: d["t_ms"])
    return rows

def ms_to_frame(ms, fps):
    return (ms / 1000.0) * fps


def parse_gap_value(value, fps):
    """Parse ``value`` into a frame count using ``fps``.

    The suffix ``s`` means seconds, ``f`` means frames (or when there is no
    suffix). Non-matching values resolve to zero to keep the existing schedule.
    """

    if not value:
        return 0
    m = re.match(r"(?i)^(?P<amount>\d+(?:\.\d+)?)(?P<unit>[sf]?)$", value.strip())
    if not m:
        return 0
    amount = float(m.group("amount"))
    unit = m.group("unit").lower()
    if unit == "s":
        return int(round(amount * fps))
    return int(round(amount))


def split_name_and_gap(folder_name, fps):
    """Return ``(base_name, gap_frames)`` parsed from ``folder_name``."""

    m = re.match(r"^(.*)_(\d+(?:\.\d+)?[sf]?)$", folder_name)
    if not m:
        return folder_name, 0
    base_name = m.group(1)
    gap_frames = parse_gap_value(m.group(2), fps)
    return base_name, gap_frames

def build_tracks_from_folder(folder, delimiter="auto"):
    files = []
    for name in os.listdir(folder):
        low = name.lower()
        if low.endswith(".csv") or low.endswith(".tsv"):
            files.append(os.path.join(folder, name))
    files.sort()
    tracks = []
    for p in files:
        data = load_csv(p, delimiter=delimiter)
        if not data:
            continue
        tracks.append({"name": os.path.splitext(os.path.basename(p))[0], "data": data})
    return tracks


def calculate_duration_from_tracks(tracks, fps):
    """Return the duration in frames for ``tracks`` at the given ``fps``."""

    if not tracks:
        return 0
    max_t_ms = max(tr["data"][-1]["t_ms"] for tr in tracks if tr["data"])
    return int(ms_to_frame(max_t_ms, fps))


def tracks_to_keydata(tracks, fps):
    """Convert loaded ``tracks`` to Skybrush key data format."""
    key_data = []
    for tr in tracks:
        if not tr["data"]:
            continue
        first = tr["data"][0]
        keys = {0: [], 1: [], 2: []}
        for row in tr["data"]:
            frame = ms_to_frame(row["t_ms"], fps)
            keys[0].append((frame, row["r"]))
            keys[1].append((frame, row["g"]))
            keys[2].append((frame, row["b"]))
        key_data.append({
            "name": tr["name"],
            "location": [first["x"], first["y"], first["z"]],
            "keys": keys,
        })
    return key_data

def ensure_mesh_with_armature(name="CSV_Tracks", count=1, first_positions=None):
    # Create an armature with one bone per vertex
    arm = bpy.data.armatures.new(name + "_arm")
    arm_obj = bpy.data.objects.new(name + "_Arm", arm)
    bpy.context.scene.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for i in range(count):
        bone = arm.edit_bones.new(f"Bone_{i}")
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 0.1)
    bpy.ops.object.mode_set(mode="POSE")
    for i in range(count):
        pb = arm_obj.pose.bones[f"Bone_{i}"]
        if first_positions and len(first_positions) == count:
            pb.location = first_positions[i]
        else:
            pb.location = (0.0, 0.0, 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Create mesh with vertices at origin
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    verts = [(0.0, 0.0, 0.0) for _ in range(count)]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    # Parent mesh to armature with modifier
    obj.parent = arm_obj
    mod = obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

    # Vertex groups per bone
    for i in range(count):
        vg = obj.vertex_groups.new(name=f"Bone_{i}")
        vg.add([i], 1.0, "REPLACE")

    # Match original orientation without manual rotation
    arm_obj.rotation_euler[0] = math.radians(-90)

    return obj


def _create_grid_positions(count, spacing=1.0):
    """Return a list of XY-plane grid positions for ``count`` items."""

    if count <= 0:
        return []

    side = math.ceil(math.sqrt(count))
    half = (side - 1) / 2.0
    positions = []
    for idx in range(count):
        row = idx // side
        col = idx % side
        x = (col - half) * spacing
        y = (row - half) * spacing
        positions.append((x, y, 0.0))
    return positions


def _world_bounds(obj):
    if obj is None:
        return None
    coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
    max_v = Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
    return min_v, max_v


def create_grid_mid_pose(
    context,
    frame_start,
    base_name="MidPose",
    spacing=1.0,
    reference_obj=None,
):
    """Create and add a grid-formation storyboard entry at ``frame_start``.

    The grid will have as many vertices as there are drone objects in the
    "Drones" collection. The formation is appended to the storyboard with a
    duration of 1 frame.
    """

    drones = bpy.data.collections.get("Drones")
    drone_count = len(drones.objects) if drones else 0
    if drone_count == 0:
        return None

    ref_spacing = spacing
    ref_center = Vector((0.0, 0.0, 0.0))
    z_offset = spacing
    bounds = _world_bounds(reference_obj)
    if bounds:
        min_v, max_v = bounds
        ref_center = (min_v + max_v) * 0.5
        dims = max_v - min_v
        max_dim = max(dims.x, dims.y, dims.z)
        side = max(1, math.ceil(math.sqrt(drone_count)))
        if side > 1 and max_dim > 0:
            ref_spacing = max_dim / (side - 1)
        elif max_dim > 0:
            ref_spacing = max_dim
        z_offset = dims.z * 0.5 if dims.z > 0 else max_dim * 0.25

    positions = _create_grid_positions(drone_count, spacing=ref_spacing)
    mesh = bpy.data.meshes.new(f"{base_name}_mesh")
    mesh.from_pydata(positions, [], [])
    mesh.update()

    obj = bpy.data.objects.new(base_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.rotation_euler[0] = math.radians(90)
    obj.location = ref_center + Vector((0.0, 0.0, z_offset))

    vg = obj.vertex_groups.new(name="Drones")
    vg.add(range(len(mesh.vertices)), 1.0, "REPLACE")
    if hasattr(obj, "skybrush"):
        obj.skybrush.formation_vertex_group = "Drones"

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj

    try:
        bpy.ops.skybrush.create_formation(name=obj.name, contents="SELECTED_OBJECTS")
        bpy.ops.skybrush.append_formation_to_storyboard()
        storyboard = context.scene.skybrush.storyboard
        entry = storyboard.entries[-1]
        entry.name = base_name
        entry.frame_start = frame_start
        entry.duration = 1
        return entry
    except Exception:
        return None

# ---------- Utilities for Replacement ----------

def clear_drone_keys(start_frame, duration):
    """Remove color keyframes on drones within the given frame range."""
    drones_col = bpy.data.collections.get("Drones")
    if not drones_col:
        return
    end_frame = start_frame + duration
    for obj in drones_col.objects:
        mat = obj.active_material
        if not mat or not mat.node_tree:
            continue
        anim = mat.node_tree.animation_data
        if not anim or not anim.action:
            continue
        for fcurve in anim.action.fcurves:
            for key in reversed(fcurve.keyframe_points):
                if start_frame <= key.co.x <= end_frame:
                    fcurve.keyframe_points.remove(key)


def _update_frame_range_from_storyboard(context):
    """Extend the scene frame range to cover all storyboard entries."""

    scene = context.scene
    storyboard = getattr(getattr(scene, "skybrush", None), "storyboard", None)
    entries = getattr(storyboard, "entries", None)
    if not entries:
        return

    max_end = 0
    for entry in entries:
        try:
            start = int(getattr(entry, "frame_start", 0))
            duration = int(getattr(entry, "duration", 0))
        except (TypeError, ValueError):
            continue
        max_end = max(max_end, start + max(duration, 0))

    if max_end > 0:
        scene.frame_end = max(scene.frame_end, max_end)


def _shift_subsequent_storyboard_entries(storyboard, start_index: int, delta: int):
    """Shift storyboard entries after ``start_index`` by ``delta`` frames."""

    if delta == 0 or storyboard is None:
        return

    entries = getattr(storyboard, "entries", None)
    if not entries:
        return

    for idx in range(start_index + 1, len(entries)):
        entry = entries[idx]
        try:
            entry.frame_start = int(getattr(entry, "frame_start", 0)) + delta
        except Exception:
            continue


def _create_light_effect_for_storyboard(
    context,
    mesh_obj,
    storyboard_entry,
    *,
    effect_type: str = "VERTEX_COLOR",
    color_image=None,
    assign_mesh: bool = True,
):
    """Create a light effect aligned with ``storyboard_entry``."""

    skybrush = getattr(context.scene, "skybrush", None)
    light_effects = getattr(skybrush, "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if entries is None:
        return None

    le_entry = None
    append_entry = getattr(light_effects, "append_new_entry", None)
    if callable(append_entry):
        try:
            le_entry = append_entry(
                storyboard_entry.name,
                storyboard_entry.frame_start,
                storyboard_entry.duration,
                select=False,
                context=context,
            )
        except Exception:
            le_entry = None

    if le_entry is None:
        try:
            le_entry = entries.add()
            le_entry.name = storyboard_entry.name
            le_entry.frame_start = storyboard_entry.frame_start
            le_entry.duration = storyboard_entry.duration
        except Exception:
            return None

    if hasattr(le_entry, "type"):
        try:
            le_entry.type = effect_type
        except Exception:
            pass
    if effect_type == "VERTEX_COLOR" and hasattr(le_entry, "output"):
        try:
            le_entry.output = OUTPUT_VERTEX_COLOR
        except Exception:
            pass
    if assign_mesh and hasattr(le_entry, "mesh"):
        try:
            le_entry.mesh = mesh_obj
        except Exception:
            pass
    if hasattr(le_entry, "convert_srgb"):
        try:
            le_entry.convert_srgb = False
        except Exception:
            pass

    if color_image is not None and hasattr(le_entry, "texture"):
        tex = getattr(le_entry, "texture", None)
        if tex is None:
            tex = bpy.data.textures.new(
                name=f"{storyboard_entry.name}_ColorTex", type="IMAGE"
            )
            le_entry.texture = tex

        if tex is not None:
            tex.image = color_image

    return le_entry


def _find_light_effect_entry(scene, name: str):
    light_effects = getattr(getattr(scene, "skybrush", None), "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if not entries:
        return None

    for entry in entries:
        if getattr(entry, "name", None) == name:
            return entry
    return None


def _replace_light_effect_texture(light_effect, color_image):
    if light_effect is None:
        return

    tex = getattr(light_effect, "texture", None)
    if tex is None:
        try:
            tex = bpy.data.textures.new(name=f"{light_effect.name}_ColorTex", type="IMAGE")
            light_effect.texture = tex
        except Exception:
            tex = None

    if tex is not None:
        try:
            old_img = getattr(tex, "image", None)
            if old_img is not None:
                bpy.data.images.remove(old_img)
        except Exception:
            pass
        try:
            tex.image = color_image
        except Exception:
            pass

# ---------- Import Helper ----------

def import_csv_folder(context, folder, start_frame, *, use_vat: bool = False):
    """Import CSV tracks from ``folder`` and create a mesh with animation.

    Returns a tuple ``(object, duration, key_entries)`` where ``object`` is the
    created mesh or ``None`` if no tracks were found, ``duration`` is the
    animation length in frames and ``key_entries`` contains color keyframe data
    prepared for later application."""
    fps = context.scene.render.fps

    folder_name = os.path.basename(os.path.normpath(folder))
    folder_name, _gap = split_name_and_gap(folder_name, fps)
    tracks = build_tracks_from_folder(folder)
    if not tracks:
        return None, 0, None


    # Determine total animation duration strictly from the CSV content
    duration = calculate_duration_from_tracks(tracks, fps)


    # Build initial positions array
    first_positions = []
    for tr in tracks:
        d0 = tr["data"][0]
        first_positions.append((d0["x"], d0["y"], d0["z"]))

    if use_vat:
        obj, color_image = csv_vat_gn.create_vat_animation_from_tracks(
            tracks,
            fps,
            start_frame=start_frame,
            base_name=f"{folder_name}_CSV",
            storyboard_name=folder_name,
        )
        key_entries = None
        color_image_for_le = color_image
    else:
        # Create mesh and armature with N bones/vertices
        # Name it after the storyboard entry + "_CSV" for clarity
        obj = ensure_mesh_with_armature(
            name=f"{folder_name}_CSV",
            count=len(tracks),
            first_positions=first_positions,
        )

        # Animate bones directly with keyframes
        arm_obj = obj.parent
        for i, tr in enumerate(tracks):
            pb = arm_obj.pose.bones.get(f"Bone_{i}")
            if not pb:
                continue
            for row in tr["data"]:
                frame = start_frame + ms_to_frame(row["t_ms"], fps)
                pb.location = (row["x"], row["y"], row["z"])
                pb.keyframe_insert(data_path="location", frame=frame)

        if arm_obj.animation_data and arm_obj.animation_data.action:
            for fcurve in arm_obj.animation_data.action.fcurves:
                for key in fcurve.keyframe_points:
                    key.interpolation = 'LINEAR'

        # Prepare color keyframe data for later application
        key_entries = tracks_to_keydata(tracks, fps)

    if obj is None:
        return None, duration, None

    # Ensure a vertex group exists for formation mapping
    vg = obj.vertex_groups.get("Drones")
    if vg is None:
        vg = obj.vertex_groups.new(name="Drones")
        vg.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')

    if hasattr(obj, "skybrush"):
        obj.skybrush.formation_vertex_group = "Drones"

    # Create a formation and add a storyboard entry
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.skybrush.create_formation(name=obj.name, contents='SELECTED_OBJECTS')
    entry = None
    try:
        bpy.ops.skybrush.append_formation_to_storyboard()
        storyboard = context.scene.skybrush.storyboard
        entry = storyboard.entries[-1]
        entry.name = folder_name
        entry.frame_start = start_frame
        entry.duration = duration
    except Exception:
        entry = None

    if use_vat and entry is not None:
        _create_light_effect_for_storyboard(
            context,
            obj,
            entry,
            effect_type="CAT",
            color_image=color_image_for_le,
            assign_mesh=False,
        )

    _update_frame_range_from_storyboard(context)

    return obj, duration, key_entries


# ---------- Properties / UI ----------


class CSVVA_PreviewItem(PropertyGroup):
    name: StringProperty(name="Name")
    folder: StringProperty(name="Folder")
    start_frame: IntProperty(name="Start Frame", default=0)
    duration: IntProperty(name="Duration", default=0)
    checked: BoolProperty(name="Include", default=False)
    exists: BoolProperty(name="Exists", default=False)
    frame_mismatch: BoolProperty(name="Frame Mismatch", default=False)


class CSVVA_Props(PropertyGroup):
    folder: StringProperty(
        name="CSV Folder",
        description="Folder containing CSV/TSV files with columns: Time[msec], x[m], y[m], z[m], Red, Green, Blue",
        subtype="DIR_PATH",
    )
    use_vat: BoolProperty(
        name="Use VAT",
        description="Generate vertex animation textures instead of bone-based animation",
        default=False,
    )
    preview_items: CollectionProperty(type=CSVVA_PreviewItem)
    preview_index: IntProperty(default=0)


class CSVVA_UL_Preview(UIList):
    bl_idname = "CSVVA_UL_Preview"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "checked", text="")
        main = row.row(align=True)
        main.label(text=item.name)
        main.label(text=f"Start: {item.start_frame}")
        main.label(text=f"Dur: {item.duration}")
        status_icon = "CHECKMARK" if not item.exists else "FILE_REFRESH"
        status_text = "New" if not item.exists else "Existing"
        if item.frame_mismatch:
            status_icon = "ERROR"
            status_text = "Frame diff"
        main.label(text=status_text, icon=status_icon)


class CSVVA_OT_Import(Operator):
    bl_idname = "csvva.import_setup"
    bl_label = "Import & Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)
        storyboard = context.scene.skybrush.storyboard
        base_start = 0
        if storyboard.entries:
            last = storyboard.entries[-1]
            base_start = last.frame_start + last.duration
        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}

        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        if subdirs:
            created = []
            next_start = base_start
            key_data_collection = []
            for idx, d in enumerate(subdirs):
                sub_path = os.path.join(folder, d)
                base_name, gap_frames = split_name_and_gap(d, context.scene.render.fps)
                sf = next_start
                obj, dur, key_entries = import_csv_folder(
                    context,
                    sub_path,
                    sf,
                    use_vat=prefs.use_vat,
                )
                if obj:
                    created.append(obj)
                    if key_entries:
                        key_data_collection.append((key_entries, sf, obj, sub_path))

                    if gap_frames >= 2 and idx < len(subdirs) - 1:
                        mid_offset = max(1, int(round(gap_frames / 2)))
                        mid_frame = sf + dur + mid_offset
                        create_grid_mid_pose(
                            context,
                            mid_frame,
                            base_name=f"{base_name}_MidPose",
                            reference_obj=obj,
                        )

                    next_start = max(next_start, sf + dur + gap_frames)
            if not created:
                self.report({"ERROR"}, "No CSV/TSV files found in subfolders")
                return {"CANCELLED"}
            try:
                bpy.ops.skybrush.recalculate_transitions(scope="ALL")
            except Exception:
                pass
            for key_entries, sf, obj, sub_path in key_data_collection:
                current_frame = context.scene.frame_current
                context.scene.frame_set(sf)
                apply_color_keys_from_key_data(
                    key_entries,
                    sf,
                )
                context.scene.frame_set(current_frame)
            self.report({"INFO"}, f"Setup complete for {len(created)} folders")
            return {"FINISHED"}

        folder_name = os.path.basename(os.path.normpath(folder))
        base_name, _gap = split_name_and_gap(
            folder_name, context.scene.render.fps
        )
        start_frame = base_start
        storyboard = context.scene.skybrush.storyboard
        for idx, sb in enumerate(storyboard.entries):
            if sb.name == base_name:
                old_obj = bpy.data.objects.get(sb.name)
                if old_obj:
                    mesh = old_obj.data
                    bpy.data.objects.remove(old_obj, do_unlink=True)
                    if mesh and mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
                clear_drone_keys(sb.frame_start, sb.duration)
                storyboard.entries.remove(idx)
                break

        obj, _, key_entries = import_csv_folder(
            context, folder, start_frame, use_vat=prefs.use_vat
        )
        if not obj:
            self.report({"ERROR"}, "No CSV/TSV files found in folder")
            return {"CANCELLED"}
        try:
            bpy.ops.skybrush.recalculate_transitions(scope="ALL")
        except Exception:
            pass
        if key_entries:
            current_frame = context.scene.frame_current
            context.scene.frame_set(start_frame)
            apply_color_keys_from_key_data(
                key_entries,
                start_frame,
            )
            context.scene.frame_set(current_frame)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({"INFO"}, f"Setup complete: {obj.name}")
        return {"FINISHED"}


class CSVVA_OT_Preview(Operator):
    bl_idname = "csvva.preview"
    bl_label = "Preview"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)
        storyboard = context.scene.skybrush.storyboard
        prefs.preview_items.clear()

        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}

        fps = context.scene.render.fps
        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        target_folders = []
        if subdirs:
            target_folders = [os.path.join(folder, d) for d in subdirs]
        else:
            target_folders = [folder]

        base_start = 0
        if storyboard.entries:
            last = storyboard.entries[-1]
            base_start = last.frame_start + last.duration

        next_start = base_start
        created = 0
        fps = context.scene.render.fps
        for path in target_folders:
            folder_name = os.path.basename(os.path.normpath(path))
            base_name, gap_frames = split_name_and_gap(folder_name, fps)
            start_frame = next_start

            tracks = build_tracks_from_folder(path)
            duration = calculate_duration_from_tracks(tracks, fps)
            if duration == 0:
                continue

            existing_entry = None
            for sb in storyboard.entries:
                if sb.name == base_name:
                    existing_entry = sb
                    break

            frame_mismatch = False
            checked = False
            display_duration = duration
            if existing_entry:
                start_frame = existing_entry.frame_start
                frame_mismatch = existing_entry.duration != duration
                display_duration = existing_entry.duration
                checked = frame_mismatch
            else:
                checked = True

            item = prefs.preview_items.add()
            item.name = base_name
            item.folder = path
            item.start_frame = start_frame
            item.duration = display_duration
            item.checked = checked
            item.exists = existing_entry is not None
            item.frame_mismatch = frame_mismatch
            created += 1

            next_start = max(next_start, start_frame + display_duration + gap_frames)

        if not created:
            self.report({"ERROR"}, "No valid CSV folders found")
            return {"CANCELLED"}

        self.report({"INFO"}, "Preview generated")
        return {"FINISHED"}


class CSVVA_OT_Update(Operator):
    bl_idname = "csvva.update"
    bl_label = "Update"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        storyboard = context.scene.skybrush.storyboard
        checked_items = [item for item in prefs.preview_items if item.checked]

        if not checked_items:
            self.report({"ERROR"}, "No items selected for update")
            return {"CANCELLED"}

        processed = 0
        key_data_collection = []
        fps = context.scene.render.fps

        for item in checked_items:
            existing_entry = None
            existing_index = None
            for idx, sb in enumerate(storyboard.entries):
                if sb.name == item.name:
                    existing_entry = sb
                    existing_index = idx
                    break

            target_start = item.start_frame
            keep_start = None
            keep_duration = None
            if existing_entry:
                keep_start = existing_entry.frame_start
                keep_duration = existing_entry.duration
                target_start = keep_start if not item.frame_mismatch else item.start_frame

            tracks = build_tracks_from_folder(item.folder)
            duration = calculate_duration_from_tracks(tracks, fps)
            if duration == 0:
                self.report({"WARNING"}, f"No CSV/TSV files found in {item.folder}")
                continue

            if existing_entry and prefs.use_vat:
                obj = None
                for candidate_name in (f"{existing_entry.name}_CSV", existing_entry.name):
                    obj = bpy.data.objects.get(candidate_name)
                    if obj:
                        break

                color_image = None
                if obj:
                    old_end = existing_entry.frame_start + existing_entry.duration
                    try:
                        color_image, imported_duration, _drone_count = csv_vat_gn.update_vat_animation_for_object(
                            obj,
                            tracks,
                            fps,
                            start_frame=target_start,
                            base_name=obj.name,
                            storyboard_name=existing_entry.name,
                        )
                        duration = imported_duration
                    except Exception:
                        color_image = None
                        imported_duration = duration

                    try:
                        existing_entry.frame_start = target_start
                        existing_entry.duration = duration
                    except Exception:
                        pass

                    if existing_index is not None:
                        new_end = target_start + duration
                        delta = new_end - old_end
                        _shift_subsequent_storyboard_entries(storyboard, existing_index, delta)

                    le_entry = _find_light_effect_entry(context.scene, existing_entry.name)
                    if le_entry is not None:
                        _replace_light_effect_texture(le_entry, color_image)
                        try:
                            le_entry.frame_start = existing_entry.frame_start
                            le_entry.duration = duration
                        except Exception:
                            pass

                    processed += 1
                    continue

            if existing_entry:
                clear_drone_keys(existing_entry.frame_start, existing_entry.duration)
                for candidate_name in (existing_entry.name, f"{existing_entry.name}_CSV"):
                    old_obj = bpy.data.objects.get(candidate_name)
                    if old_obj:
                        mesh = old_obj.data
                        bpy.data.objects.remove(old_obj, do_unlink=True)
                        if mesh and mesh.users == 0:
                            bpy.data.meshes.remove(mesh)
                storyboard.entries.remove(existing_index)

            obj, imported_duration, key_entries = import_csv_folder(
                context, item.folder, target_start, use_vat=prefs.use_vat
            )
            if not obj:
                self.report({"WARNING"}, f"Failed to import from {item.folder}")
                continue

            entry = storyboard.entries[-1]
            if existing_entry is not None:
                storyboard.entries.move(len(storyboard.entries) - 1, existing_index)
                if not item.frame_mismatch:
                    entry.frame_start = keep_start
                    entry.duration = keep_duration
                else:
                    entry.frame_start = item.start_frame
                    entry.duration = duration

                old_end = (keep_start or 0) + (keep_duration or 0)
                new_end = entry.frame_start + entry.duration
                _shift_subsequent_storyboard_entries(storyboard, existing_index, new_end - old_end)

            if key_entries:
                key_data_collection.append((key_entries, entry.frame_start))
            processed += 1

        if not processed:
            self.report({"ERROR"}, "Nothing was updated")
            return {"CANCELLED"}

        current_frame = context.scene.frame_current
        for key_entries, start_frame in key_data_collection:
            context.scene.frame_set(start_frame)
            apply_color_keys_from_key_data(
                key_entries,
                start_frame,
            )
        context.scene.frame_set(current_frame)

        _update_frame_range_from_storyboard(context)

        self.report(
            {"INFO"},
            f"Updated {processed} formation(s). Recalculate transitions manually if needed.",
        )
        return {"FINISHED"}

class CSVVA_PT_UI(Panel):
    bl_label = "CSV Vertex Anim"
    bl_idname = "CSVVA_PT_UI"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        lay = self.layout
        prefs = context.scene.csvva_props
        col = lay.column(align=True)
        col.prop(prefs, "folder")
        col.prop(prefs, "use_vat")
        row = col.row(align=True)
        row.operator(CSVVA_OT_Import.bl_idname, icon="IMPORT")
        row.operator(CSVVA_OT_Preview.bl_idname, icon="VIEWZOOM")
        col.template_list(
            CSVVA_UL_Preview.bl_idname,
            "",
            prefs,
            "preview_items",
            prefs,
            "preview_index",
            rows=4,
        )
        col.operator(CSVVA_OT_Update.bl_idname, icon="FILE_REFRESH")


# ---------- Registration ----------

classes = (
    CSVVA_PreviewItem,
    CSVVA_Props,
    CSVVA_UL_Preview,
    CSVVA_OT_Import,
    CSVVA_OT_Preview,
    CSVVA_OT_Update,
    CSVVA_PT_UI,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.csvva_props = bpy.props.PointerProperty(type=CSVVA_Props)


def unregister():
    del bpy.types.Scene.csvva_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
