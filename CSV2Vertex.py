import bpy
from bpy.props import StringProperty
from bpy.types import Operator, Panel, PropertyGroup
import csv, json, os, re, math
from mathutils import Vector
from bisect import bisect_left

HANDLER_TAG = "_csv_vertex_anim_handler"

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

def unregister_handler():
    sc = bpy.context.scene
    # Remove existing handler if exists
    if getattr(sc, HANDLER_TAG, None):
        tag = sc.get(HANDLER_TAG)
        # Clear all matching handlers
        for h in list(bpy.app.handlers.frame_change_post):
            if getattr(h, "__name__", "") == tag:
                bpy.app.handlers.frame_change_post.remove(h)
        del sc[HANDLER_TAG]

def register_handler(fn):
    unregister_handler()
    bpy.app.handlers.frame_change_post.append(fn)
    bpy.context.scene[HANDLER_TAG] = fn.__name__

# ---------- Core: Frame Update ----------

def make_frame_handler(obj):
    """
    Returns a frame_change_post handler function that:
      - Reads JSON tracks from obj['csv_tracks_json']
      - Moves each vertex to interpolated position at current frame
    """
    # Capture name to resolve object each call (avoid stale refs on file reload)
    obj_name = obj.name

    def handler(scene):
        ob = scene.objects.get(obj_name)
        if not ob or "csv_tracks_json" not in ob:
            return
        if not ob.data or ob.data.name not in bpy.data.meshes:
            return

        arm = ob.find_armature()
        if not arm:
            return

        try:
            payload = json.loads(ob["csv_tracks_json"])
        except Exception:
            return

        fps = float(payload.get("fps", 24.0))
        start_frame = int(payload.get("start_frame", 0))
        tracks = payload.get("tracks", [])
        if not tracks:
            return

        # Build per-track time list cache (optional)
        frame = scene.frame_current
        t_ms = ((frame - start_frame) / fps) * 1000.0

        bones = arm.pose.bones
        n = min(len(tracks), len(bones))

        for i in range(n):
            seq = tracks[i]["data"]  # list of dicts
            # binary search on time
            times = payload["time_index"][i]
            idx = bisect_left(times, t_ms)
            if idx <= 0:
                p = seq[0]
                ob_co = (p["x"], p["y"], p["z"])
            elif idx >= len(seq):
                p = seq[-1]
                ob_co = (p["x"], p["y"], p["z"])
            else:
                a = seq[idx - 1]
                b = seq[idx]
                # Linear interpolation
                t0, t1 = a["t_ms"], b["t_ms"]
                k = 0.0 if t1 == t0 else (t_ms - t0) / (t1 - t0)
                ob_co = (
                    a["x"] + (b["x"] - a["x"]) * k,
                    a["y"] + (b["y"] - a["y"]) * k,
                    a["z"] + (b["z"] - a["z"]) * k,
                )
            pb = bones.get(f"Bone_{i}")
            if pb:
                pb.location = Vector(ob_co)

        arm.update_tag(refresh={"DATA"})

    # Give a stable name for clean unregistration
    handler.__name__ = "csv_vertex_anim_handler_runtime"
    return handler

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

# ---------- Import Helper ----------

def import_csv_folder(context, folder, start_frame):
    """Import CSV tracks from ``folder`` and create a mesh with animation.

    Returns a tuple ``(object, duration)`` where ``object`` is the created
    mesh or ``None`` if no tracks were found, and ``duration`` is the animation
    length in frames."""
    folder_name = os.path.basename(os.path.normpath(folder))
    m = re.search(r"(.+)_\d+$", folder_name)
    if m:
        folder_name = m.group(1)
    tracks = build_tracks_from_folder(folder)
    if not tracks:
        return None, 0

    fps = context.scene.render.fps

    # Determine total duration in frames
    max_t_ms = max(tr["data"][-1]["t_ms"] for tr in tracks if tr["data"])
    duration = int(ms_to_frame(max_t_ms, fps))

    # Build initial positions array
    first_positions = []
    for tr in tracks:
        d0 = tr["data"][0]
        first_positions.append((d0["x"], d0["y"], d0["z"]))

    # Create mesh and armature with N bones/vertices
    obj = ensure_mesh_with_armature(name=folder_name, count=len(tracks), first_positions=first_positions)

    # Create a vertex group containing all vertices for formations
    vg = obj.vertex_groups.new(name="Drones")
    vg.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')

    if hasattr(obj, "skybrush"):
        obj.skybrush.formation_vertex_group = "Drones"

    # Create a formation and add a storyboard entry
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.skybrush.create_formation(name=obj.name, contents='SELECTED_OBJECTS')
    try:
        bpy.ops.skybrush.append_formation_to_storyboard()
        storyboard = context.scene.skybrush.storyboard
        entry = storyboard.entries[-1]
        entry.name = folder_name
        entry.frame_start = start_frame
        entry.duration = duration
    except Exception:
        pass

    # Build payload for handler: store compact JSON on the object
    payload = {
        "fps": fps,
        "start_frame": start_frame,
        "duration": duration,
        "tracks": tracks,
        "time_index": [[row["t_ms"] for row in tr["data"]] for tr in tracks],
    }
    obj["csv_tracks_json"] = json.dumps(payload)

    # Color keyframes: store on object custom properties
    normalize = False
    fps = float(fps)

    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data.action:
        obj.animation_data.action = bpy.data.actions.new(name="CSV_ColorKeys")

    for i, tr in enumerate(tracks):
        for ch in ("R", "G", "B"):
            key = f'vc[{i}]_{ch}'
            if key not in obj:
                obj[key] = 0.0

        for row in tr["data"]:
            frame = start_frame + ms_to_frame(row["t_ms"], fps)
            raw_r, raw_g, raw_b = row["r"], row["g"], row["b"]
            prop_r, prop_g, prop_b = (
                (raw_r / 255.0, raw_g / 255.0, raw_b / 255.0)
                if normalize
                else (raw_r, raw_g, raw_b)
            )
            obj[f'vc[{i}]_R'] = float(prop_r)
            obj[f'vc[{i}]_G'] = float(prop_g)
            obj[f'vc[{i}]_B'] = float(prop_b)
            obj.keyframe_insert(f'["vc[{i}]_R"]', frame=frame)
            obj.keyframe_insert(f'["vc[{i}]_G"]', frame=frame)
            obj.keyframe_insert(f'["vc[{i}]_B"]', frame=frame)

    # Register/update frame handler
    handler = make_frame_handler(obj)
    register_handler(handler)

    return obj, duration

# ---------- Color Key Transfer ----------

def transfer_color_keys(obj):
    """Transfer vertex color keyframes to nearest drones and clean up.

    ``obj`` should be the mesh object created by ``import_csv_folder`` that
    stores color keyframes in custom properties of the form ``vc[index]``.
    The function finds the nearest object from the ``Drones`` collection for
    each vertex and applies the stored color keyframes to that object's
    material, then removes the temporary keyframes and properties from
    ``obj``.
    """

    if not obj or "csv_tracks_json" not in obj:
        return

    from color_key_utils import apply_color_keys_to_nearest

    payload = json.loads(obj["csv_tracks_json"])
    start_frame = int(payload.get("start_frame", bpy.context.scene.frame_current))
    bpy.context.scene.frame_set(start_frame)

    drones_col = bpy.data.collections.get("Drones")
    if not drones_col:
        return
    available_objects = list(drones_col.objects)
    if not available_objects:
        return

    if not obj.animation_data or not obj.animation_data.action:
        return
    action = obj.animation_data.action

    key_map = {}
    for fc in action.fcurves:
        m = re.match(r'\["vc\[(\d+)\]_([RGB])"\]', fc.data_path)
        if not m:
            continue
        vid = int(m.group(1))
        ch = m.group(2)
        pts = [(kp.co.x, kp.co.y) for kp in fc.keyframe_points]
        if pts:
            key_map.setdefault(vid, {})[ch] = pts

    mesh = obj.data

    for vid, channels in key_map.items():
        if vid >= len(mesh.vertices):
            continue
        v_world = obj.matrix_world @ mesh.vertices[vid].co
        channel_map = {"R": 0, "G": 1, "B": 2}
        data = {channel_map[ch]: frames for ch, frames in channels.items()}
        apply_color_keys_to_nearest(
            v_world,
            data,
            available_objects,
            frame_offset=0,
            normalize_255=True,
        )

    # Remove vc custom property keyframes and properties
    action = obj.animation_data.action
    for fc in list(action.fcurves):
        if fc.data_path.startswith('["vc['):
            action.fcurves.remove(fc)
    if not action.fcurves:
        obj.animation_data.action = None
        bpy.data.actions.remove(action)

    for key in list(obj.keys()):
        if key.startswith("vc["):
            del obj[key]

# ---------- Properties / UI ----------

class CSVVA_Props(PropertyGroup):
    folder: StringProperty(
        name="CSV Folder",
        description="Folder containing CSV/TSV files with columns: Time[msec], x[m], y[m], z[m], Red, Green, Blue",
        subtype="DIR_PATH",
    )

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
            for d in subdirs:
                sub_path = os.path.join(folder, d)
                m = re.search(r".*_(\d+)$", d)
                if m:
                    sf = int(m.group(1))
                else:
                    sf = next_start
                obj, dur = import_csv_folder(context, sub_path, sf)
                if obj:
                    created.append(obj)
                    next_start = max(next_start, sf + dur)
            if not created:
                self.report({"ERROR"}, "No CSV/TSV files found in subfolders")
                return {"CANCELLED"}
            try:
                bpy.ops.skybrush.recalculate_transitions(scope="ALL")
            except Exception:
                pass
            for obj in created:
                try:
                    transfer_color_keys(obj)
                except Exception:
                    pass
            self.report({"INFO"}, f"Setup complete for {len(created)} folders")
            return {"FINISHED"}

        folder_name = os.path.basename(os.path.normpath(folder))
        start_frame = base_start
        base_name = folder_name
        m = re.search(r"(.+)_([0-9]+)$", folder_name)
        if m:
            base_name, start_frame = m.group(1), int(m.group(2))
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

        obj, _ = import_csv_folder(context, folder, start_frame)
        if not obj:
            self.report({"ERROR"}, "No CSV/TSV files found in folder")
            return {"CANCELLED"}
        try:
            bpy.ops.skybrush.recalculate_transitions(scope="ALL")
        except Exception:
            pass
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        try:
            transfer_color_keys(obj)
        except Exception:
            pass
        self.report({"INFO"}, f"Setup complete: {obj.name}")
        return {"FINISHED"}

class CSVVA_OT_RemoveHandler(Operator):
    bl_idname = "csvva.remove_handler"
    bl_label = "Remove Position Update Handler"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        unregister_handler()
        self.report({"INFO"}, "Removed frame change handler")
        return {"FINISHED"}

class CSVVA_PT_UI(Panel):
    bl_label = "CSV Vertex Anim"
    bl_idname = "CSVVA_PT_UI"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CSV Vertex Anim"

    def draw(self, context):
        lay = self.layout
        prefs = context.scene.csvva_props
        col = lay.column(align=True)
        col.prop(prefs, "folder")
        col.operator(CSVVA_OT_Import.bl_idname, icon="IMPORT")
        col.separator()
        col.operator(CSVVA_OT_RemoveHandler.bl_idname, icon="X")


# ---------- Registration ----------

classes = (
    CSVVA_Props,
    CSVVA_OT_Import,
    CSVVA_OT_RemoveHandler,
    CSVVA_PT_UI,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.csvva_props = bpy.props.PointerProperty(type=CSVVA_Props)


def unregister():
    unregister_handler()
    del bpy.types.Scene.csvva_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
