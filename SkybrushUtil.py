bl_info = {
    "name": "SkyBrushUtil",
    "author": "ABEYUYA",
    "version": (3, 0, 0),
    "blender": (4, 3, 2),
    "location": "3D View > Sidebar > SBUtil",
    "description": "SkybrushTransfarUtil",
    "category": "Animation",
}

import bpy
import importlib
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty
from bpy.types import Panel, Operator, PropertyGroup, AddonPreferences
from mathutils import Vector
import json, os, shutil, tempfile, urllib.request
from sbstudio.plugin.operators import RecalculateTransitionsOperator
from sbstudio.plugin.operators.base import StoryboardOperator
from sbstudio.plugin.constants import Collections
from sbutil import formation_patch
from sbutil import light_effects as light_effects_patch
from sbutil import light_effects_result_patch
from sbutil import recalculate_transitions_patch
from sbutil import CSV2Vertex
from sbutil import drone_mesh_gn
from sbutil import reflow_vertex
from sbutil import drone_check_gn
from sbutil import view_setup
from sbutil import storyboard_patch

try:  # pragma: no cover - depends on sbstudio
    from sbstudio.plugin.operators.safety_check import RunFullProximityCheckOperator
except Exception:  # pragma: no cover - optional dependency
    try:
        from sbstudio.plugin.operators import RunFullProximityCheckOperator
    except Exception:  # pragma: no cover - optional dependency
        RunFullProximityCheckOperator = None

_RUN_FULL_PROXIMITY_OP = (
    RunFullProximityCheckOperator.bl_idname
    if RunFullProximityCheckOperator is not None
    else "skybrush.run_full_proximity_check"
)


KeydataStr = "_KeyData.json"
LightdataStr = "_LightData.json"


class DRONE_OT_UpdateAddon(Operator):
    """Fetch and install the latest release of the add-on from GitHub."""

    bl_idname = "drone.update_addon"
    bl_label = "Update Add-on"
    bl_description = "Download and install the latest version of SkyBrushUtil"

    RELEASE_API = (
        "https://api.github.com/repos/abev-crypto/SkybrushUtil/releases/latest"
    )

    def execute(self, context):
        current = tuple(bl_info.get("version", (0, 0)))
        try:
            with urllib.request.urlopen(self.RELEASE_API) as response:
                data = json.load(response)
        except Exception as exc:  # pragma: no cover - network failure
            self.report({'ERROR'}, f"Update check failed: {exc}")
            return {'CANCELLED'}

        tag = data.get("tag_name") or ""
        try:
            latest = tuple(int(x) for x in tag.lstrip("vV").split("."))
        except ValueError:
            self.report({'ERROR'}, f"Invalid version format: {tag}")
            return {'CANCELLED'}

        if latest <= current:
            self.report({'INFO'}, "Add-on is up to date")
            return {'CANCELLED'}

        zip_url = data.get("zipball_url")
        if not zip_url:
            self.report({'ERROR'}, "Release package not found")
            return {'CANCELLED'}

        try:
            with urllib.request.urlopen(zip_url) as resp:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp.write(resp.read())
                    zip_path = tmp.name
            bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
            bpy.ops.preferences.addon_enable(module=__name__)
            self.report({'INFO'}, f"Updated to {'.'.join(map(str, latest))}")
            return {'FINISHED'}
        except Exception as exc:  # pragma: no cover - installation failure
            self.report({'ERROR'}, f"Update failed: {exc}")
            return {'CANCELLED'}
        finally:
            try:
                os.remove(zip_path)
            except Exception:
                pass


def ensure_json_files(blend_dir, prefix):
    """Ensure key and light JSON files exist in ``blend_dir`` for ``prefix``.

    If either file is missing, the user is prompted to select a source
    directory and the files are copied from there.  Returns ``True`` when
    both files are present after the optional copy attempt.
    """
    key_path = os.path.join(blend_dir, prefix + KeydataStr)
    light_path = os.path.join(blend_dir, prefix + LightdataStr)
    if os.path.exists(key_path) and os.path.exists(light_path):
        return True

    class DRONE_OT_SelectJsonDir(bpy.types.Operator):
        bl_idname = "drone.select_json_dir"
        bl_label = "Select JSON Source Directory"

        directory: StringProperty(subtype="DIR_PATH")

        def execute(self, context):
            src_key = os.path.join(self.directory, prefix + KeydataStr)
            src_light = os.path.join(self.directory, prefix + LightdataStr)
            if os.path.exists(src_key):
                shutil.copy(src_key, key_path)
            if os.path.exists(src_light):
                shutil.copy(src_light, light_path)
            return {'FINISHED'}

        def invoke(self, context, event):
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

    bpy.utils.register_class(DRONE_OT_SelectJsonDir)
    bpy.ops.drone.select_json_dir('INVOKE_DEFAULT')
    bpy.utils.unregister_class(DRONE_OT_SelectJsonDir)

    return os.path.exists(key_path) and os.path.exists(light_path)


def ensure_light_json_file(blend_dir, prefix):
    """Ensure a light JSON file exists in ``blend_dir`` for ``prefix``."""

    light_path = os.path.join(blend_dir, prefix + LightdataStr)
    if os.path.exists(light_path):
        return True

    class DRONE_OT_SelectLightJsonDir(bpy.types.Operator):
        bl_idname = "drone.select_light_json_dir"
        bl_label = "Select Light JSON Source Directory"

        directory: StringProperty(subtype="DIR_PATH")

        def execute(self, context):
            src_light = os.path.join(self.directory, prefix + LightdataStr)
            if os.path.exists(src_light):
                shutil.copy(src_light, light_path)
            return {'FINISHED'}

        def invoke(self, context, event):
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

    bpy.utils.register_class(DRONE_OT_SelectLightJsonDir)
    bpy.ops.drone.select_light_json_dir('INVOKE_DEFAULT')
    bpy.utils.unregister_class(DRONE_OT_SelectLightJsonDir)

    return os.path.exists(light_path)


def _find_drone_collection():
    collection = bpy.data.collections.get("DroneCollection")
    if collection is not None:
        return collection

    spec = importlib.util.find_spec("sbstudio.plugin.constants")
    if spec is not None:
        from sbstudio.plugin.constants import Collections

        try:
            collection = Collections.find_drones()
        except Exception:
            collection = None
        if collection is not None:
            return collection

    return bpy.data.collections.get("Drones")


def _iter_drone_mesh_objects(collection):
    objects = getattr(collection, "all_objects", None) or collection.objects
    for obj in objects:
        if obj.type == 'MESH':
            yield obj

# -------------------------------
# LightEffect Export Helpers
# -------------------------------
def convert_value(value):
    """Blender固有型やPropertyGroupをJSON化可能な値に変換"""
    if hasattr(value, "__annotations__"):
        return propertygroup_to_dict(value)

    if isinstance(value, bpy.types.bpy_prop_collection):
        return [convert_value(item) for item in value]

    if isinstance(value, bpy.types.ID):
        if isinstance(value, bpy.types.Image):
            return value.filepath_raw or value.name
        return value.name

    return value


def convert_color_ramp(texture):
    """ColorRamp情報を辞書化"""
    if not texture or not getattr(texture, "use_color_ramp", False):
        return None

    ramp = getattr(texture, "color_ramp", None)
    if ramp is None:
        return None

    data = {
        "texture_name": texture.name,
        "type": getattr(texture, "type", None),
        "use_color_ramp": bool(texture.use_color_ramp),
        "color_mode": getattr(ramp, "color_mode", None),
        "interpolation": getattr(ramp, "interpolation", None),
        "hue_interpolation": getattr(ramp, "hue_interpolation", None),
        "elements": [],
    }

    for elem in ramp.elements:
        data["elements"].append(
            {
                "position": float(elem.position),
                "color": [float(c) for c in elem.color],
            }
        )

    return data


def propertygroup_to_dict(pg):
    """PropertyGroupを辞書化（name含める）"""
    data = {"name": pg.name}

    # 新しいプロパティ(loop_count, loop_method)を含めた全プロパティ名を収集
    props = set(getattr(pg, "__annotations__", {}).keys())
    for extra in ("loop_count", "loop_method"):
        if hasattr(pg, extra):
            props.add(extra)

    span = None
    if hasattr(pg, "sequence_mode") and hasattr(light_effects_patch, "calculate_effective_sequence_span"):
        try:
            span = light_effects_patch.calculate_effective_sequence_span(pg)
        except Exception:
            span = None

    frame_end_override = total_duration_override = None
    if span is not None:
        frame_end_override, total_duration_override = span

    for prop in props:
        if prop == "frame_end" and frame_end_override is not None:
            data[prop] = frame_end_override
            continue
        if prop == "duration" and total_duration_override is not None:
            data[prop] = total_duration_override
            continue
        val = getattr(pg, prop)
        data[prop] = convert_value(val)

    if getattr(pg, "type", None) == "COLOR_RAMP":
        texture = getattr(pg, "texture", None)
        ramp_data = convert_color_ramp(texture)
        if ramp_data:
            data["color_ramp"] = ramp_data

    return data


def export_light_effects_to_json(filepath, context):
    scene = context.scene
    light_effects = scene.skybrush.light_effects.entries
    effects_data = [propertygroup_to_dict(effect) for effect in light_effects]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(effects_data, f, ensure_ascii=False, indent=4)


def active_timebind_prefix(context, default_name):
    """Return the active TimeBind prefix or a sanitized default."""
    tb = context.scene.time_bind
    entries = tb.entries
    index = tb.active_index
    if 0 <= index < len(entries):
        return entries[index].Prefix.rstrip("_")
    return default_name.rstrip("_")

# -------------------------------
# プロパティ
# -------------------------------
class DroneKeyTransferProperties(PropertyGroup):
    file_name : StringProperty(
        name="File Name",
        default="color_key_data",
        description="Save/Load JSON file name"
    )

# TimeBind用のPropertyGroup
class TimeBindEntry(bpy.types.PropertyGroup):
    Prefix : StringProperty(name="Prefix")
    StartFrame : bpy.props.IntProperty(name="Start Frame")

class ShiftPrefixEntry(bpy.types.PropertyGroup):
    Prefix : StringProperty(name="Prefix")

# コレクション全体の管理用
class TimeBindCollection(bpy.types.PropertyGroup):
    entries : bpy.props.CollectionProperty(type=TimeBindEntry)
    active_index : bpy.props.IntProperty()


class ShiftPrefixList(bpy.types.PropertyGroup):
    entries : bpy.props.CollectionProperty(type=ShiftPrefixEntry)
    active_index : bpy.props.IntProperty()
    shift_amount : bpy.props.IntProperty(name="Shift Frames", default=0)


class SBUTIL_StoryboardBatchItem(bpy.types.PropertyGroup):
    name: StringProperty(name="Name")
    start_frame: bpy.props.IntProperty(name="Start", default=0)
    end_frame: bpy.props.IntProperty(name="End", default=0)
    duration: bpy.props.IntProperty(name="Duration", default=0)
    is_transition: BoolProperty(name="Is Transition", default=False)
    selected: BoolProperty(name="Export", default=True)


class SBUTIL_StoryboardBatchSettings(bpy.types.PropertyGroup):
    entries: bpy.props.CollectionProperty(type=SBUTIL_StoryboardBatchItem)
    active_index: bpy.props.IntProperty()
    include_transitions: BoolProperty(
        name="Include Transitions",
        description="Add gaps between storyboard entries as transition ranges",
        default=True,
    )

class TIMEBIND_UL_entries(bpy.types.UIList):
    """TimeBind entriesを表示するUIList"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item: TimeBindEntry
        split = layout.split(factor=0.5)
        split.label(text=item.Prefix if item.Prefix else "-")


class TIMEBIND_UL_shift_prefixes(bpy.types.UIList):
    """Shift prefix entriesを表示するUIList"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        split = layout.split(factor=1.0)
        split.label(text=item.Prefix if item.Prefix else "-")


class SBUTIL_UL_StoryboardBatch(bpy.types.UIList):
    bl_idname = "SBUTIL_UL_StoryboardBatch"

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "selected", text="")
            icon = 'ARROW_LEFTRIGHT' if item.is_transition else 'SEQ_STRIP_META'
            row.label(text=item.name, icon=icon)
            row.label(text=str(item.start_frame))
            row.label(text=str(item.end_frame))
            row.label(text=str(item.duration))
        elif self.layout_type == "GRID":
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


class SBUTIL_OT_LoadStoryboardBatch(Operator):
    bl_idname = "sbutil.load_storyboard_batch"
    bl_label = "Load Storyboard"
    bl_description = "Load storyboard start/end frames into the export list"

    def execute(self, context):
        settings = context.scene.sbutil_storyboard_export
        entries = getattr(getattr(context.scene, "skybrush", None), "storyboard", None)

        settings.entries.clear()

        storyboard_entries = getattr(entries, "entries", None)
        if not storyboard_entries:
            self.report({'ERROR'}, "No storyboard entries found")
            settings.active_index = -1
            return {'CANCELLED'}

        sorted_entries = sorted(storyboard_entries, key=lambda e: e.frame_start)
        previous_entry = None
        previous_end = None

        i = 0
        while i < len(sorted_entries):
            entry = sorted_entries[i]
            start = int(getattr(entry, "frame_start", 0))
            duration = int(getattr(entry, "duration", 0))
            end = start + max(duration, 0)

            is_mid_pose = "MidPose" in getattr(entry, "name", "")

            if (
                settings.include_transitions
                and is_mid_pose
                and previous_entry is not None
                and previous_end is not None
                and i + 1 < len(sorted_entries)
            ):
                next_entry = sorted_entries[i + 1]
                next_start = int(getattr(next_entry, "frame_start", 0))
                transition_start = previous_end
                transition_end = next_start - 1

                if transition_end >= transition_start:
                    transition = settings.entries.add()
                    transition.name = f"{previous_entry.name}_To_{next_entry.name}"
                    transition.start_frame = transition_start
                    transition.end_frame = transition_end
                    transition.duration = transition_end - transition_start + 1
                    transition.is_transition = True

                previous_entry = None
                previous_end = None
                i += 1
                continue

            if (
                settings.include_transitions
                and previous_entry is not None
                and previous_end is not None
                and start > previous_end
            ):
                transition_start = previous_end
                transition_end = start - 1

                if transition_end >= transition_start:
                    transition = settings.entries.add()
                    transition.name = f"{previous_entry.name}_To_{entry.name}"
                    transition.start_frame = transition_start
                    transition.end_frame = transition_end
                    transition.duration = transition_end - transition_start + 1
                    transition.is_transition = True

            item = settings.entries.add()
            item.name = entry.name
            item.start_frame = start
            item.end_frame = end
            item.duration = max(duration, 0)
            item.is_transition = False

            previous_entry = entry
            previous_end = end
            i += 1

        settings.active_index = 0 if settings.entries else -1
        self.report({'INFO'}, f"Loaded {len(settings.entries)} range(s)")
        return {'FINISHED'}


class SBUTIL_OT_ExportStoryboardBatch(Operator):
    bl_idname = "sbutil.export_storyboard_batch"
    bl_label = "Export CSV Batch"
    bl_description = "Export checked storyboard ranges as Skybrush CSV archives"

    def execute(self, context):
        scene = context.scene
        settings = scene.sbutil_storyboard_export
        blend_path = bpy.data.filepath
        if not blend_path:
            self.report({'ERROR'}, "Save the .blend file before exporting")
            return {'CANCELLED'}

        base_dir = os.path.dirname(blend_path)
        if not settings.entries:
            self.report({'ERROR'}, "No storyboard ranges loaded")
            return {'CANCELLED'}

        exported = 0
        for item in settings.entries:
            if not item.selected:
                continue

            start = int(item.start_frame)
            end = int(item.end_frame)
            if end < start:
                continue

            scene.frame_start = start
            scene.frame_end = end
            scene.frame_set(start)

            filepath = os.path.join(base_dir, f"{item.name}.zip")

            try:
                result = bpy.ops.export_scene.skybrush_csv(
                    filepath=filepath,
                    check_existing=False,
                    export_selected=False,
                    frame_range='RENDER',
                    redraw='AUTO',
                    output_fps=24.0,
                )
            except Exception as exc:
                self.report({'ERROR'}, f"Export failed for {item.name}: {exc}")
                return {'CANCELLED'}

            if 'FINISHED' not in result:
                self.report({'ERROR'}, f"Export failed for {item.name}")
                return {'CANCELLED'}

            exported += 1

        if exported == 0:
            self.report({'WARNING'}, "No entries selected for export")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Exported {exported} storyboard range(s)")
        return {'FINISHED'}


class SBUTIL_OT_SetRenderRangeFromStoryboard(Operator):
    bl_idname = "sbutil.set_render_range_from_storyboard"
    bl_label = "Render Range from Storyboard"
    bl_description = "Match the render range to the active storyboard entry"

    def execute(self, context):
        entry, start, end, entries = _active_storyboard_entry_with_range(context)
        if not entries:
            self.report({'WARNING'}, "No storyboard entries available")
            return {'CANCELLED'}
        if entry is None:
            self.report({'WARNING'}, "No storyboard entry contains the current frame")
            return {'CANCELLED'}

        context.scene.frame_start = start
        context.scene.frame_end = end
        context.scene.frame_set(start)

        self.report({'INFO'}, f"Render range set to {getattr(entry, 'name', '')}")
        return {'FINISHED'}


def _formation_meshes_for_entry(entry):
    for attr in ("formation_collection", "collection", "formation"):
        collection = getattr(entry, attr, None)
        if isinstance(collection, bpy.types.Collection):
            objects = getattr(collection, "all_objects", None) or collection.objects
            return [obj for obj in objects if obj.type == 'MESH']
    return []


class SBUTIL_OT_IsolateActiveFormation(Operator):
    bl_idname = "sbutil.isolate_active_formation"
    bl_label = "Show Active Formation Only"
    bl_description = (
        "Hide meshes from inactive storyboard entries' formation collections"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        entry, _, _, entries = _active_storyboard_entry_with_range(context)
        if not entries:
            self.report({'WARNING'}, "No storyboard entries available")
            return {'CANCELLED'}
        target_index = next(
            (idx for idx, item in enumerate(entries) if item == entry),
            -1,
        )

        if target_index < 0:
            self.report({'WARNING'}, "No storyboard entry contains the current frame")
            return {'CANCELLED'}

        shown, hidden = 0, 0
        for idx, entry in enumerate(entries):
            for obj in _formation_meshes_for_entry(entry):
                try:
                    if idx == target_index:
                        obj.hide_set(False)
                        obj.hide_select = False
                        shown += 1
                    else:
                        obj.hide_set(True)
                        obj.hide_select = True
                        hidden += 1
                except Exception:
                    continue

        self.report({'INFO'}, f"Shown {shown} mesh(es); hidden {hidden} mesh(es)")
        return {'FINISHED'}

# -------------------------------
# 抽出（保存）オペレーター
# -------------------------------
class DRONE_OT_SaveKeys(Operator):
    bl_idname = "drone.save_keys"
    bl_label = "Save Keys"
    bl_description = "Export keyframe and light effect data to JSON files"

    def execute(self, context):
        """Prepare paths, export key data and light effects, and report."""
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        props = context.scene.drone_key_props
        prefix = active_timebind_prefix(context, props.file_name)
        # Ensure textures and materials have prefixed names for export
        add_prefix_le_tex(context)
        # Export keyframe data
        export_key(context, blend_dir, prefix)
        # Export light effect definitions
        export_light_effects_to_json(
            os.path.join(blend_dir, prefix + LightdataStr), context
        )
        self.report({'INFO'}, f"Keys saved: {blend_dir}")
        return {'FINISHED'}
    
class DRONE_OT_SaveSignleKeys(Operator):
    bl_idname = "drone.save_single_keys"
    bl_label = "Save Single Keys"
    bl_description = "Export keyframe data for the current selection to a JSON file"

    def execute(self, context):
        """Save keyframes for the current selection using the configured file name.

        The keyframes are written to the directory of the active blend file and the
        resulting path is reported.
        """
        props = context.scene.drone_key_props
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        prefix = active_timebind_prefix(context, props.file_name)
        export_key(context, blend_dir, prefix)
        self.report({'INFO'}, f"Keys saved: {blend_dir}")
        return {'FINISHED'}


class DRONE_OT_SaveLightEffects(Operator):
    bl_idname = "drone.save_light_effects"
    bl_label = "Save Light Effects"
    bl_description = "Export light effect data to a JSON file"

    def execute(self, context):
        props = context.scene.drone_key_props
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        prefix = active_timebind_prefix(context, props.file_name)
        export_light_effects_to_json(
            os.path.join(blend_dir, prefix + LightdataStr), context
        )
        self.report({'INFO'}, f"Light effects saved: {blend_dir}")
        return {'FINISHED'}


class DRONE_OT_LoadLightEffects(Operator):
    bl_idname = "drone.load_light_effects"
    bl_label = "Load Light Effects"
    bl_description = "Import light effect data from a JSON file"

    def execute(self, context):
        props = context.scene.drone_key_props
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        prefix = active_timebind_prefix(context, props.file_name)
        frame_offset = bpy.context.scene.frame_current

        tb_entries = context.scene.time_bind.entries
        index = context.scene.time_bind.active_index
        if 0 <= index < len(tb_entries) and tb_entries:
            frame_offset = tb_entries[index].StartFrame
            prefix = tb_entries[index].Prefix

        if not ensure_light_json_file(blend_dir, prefix):
            self.report({'WARNING'}, f"Light JSON for prefix '{prefix}' not found")
            return {'CANCELLED'}

        import_light_effects_from_json(
            os.path.join(blend_dir, prefix + LightdataStr), frame_offset, context
        )
        update_texture_key(prefix + "_", frame_offset)
        add_timebind_prop(context, prefix + "_", frame_offset)
        self.report({'INFO'}, f"Light effects loaded: {blend_dir}")
        return {'FINISHED'}

# -------------------------------
# 移植（ロード）オペレーター
# -------------------------------
class DRONE_OT_LoadKeys(Operator):
    bl_idname = "drone.load_keys"
    bl_label = "Load Keys"
    bl_description = "Import keyframe and light effect data using the active TimeBind entry"

    def execute(self, context):
        """
        Load previously saved animation and effect data.

        Steps:
        1. Determine the prefix and start frame from the active TimeBind entry.
        2. Apply keyframe data with the calculated frame offset.
        3. Import light effect definitions.
        4. Update texture animations and record the binding.
        5. Report completion.
        """
        fn = context.scene.drone_key_props.file_name
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        current_frame = bpy.context.scene.frame_current
        storyboard = context.scene.skybrush.storyboard
        duration = 0
        tb_entries = context.scene.time_bind.entries
        index = context.scene.time_bind.active_index
        if 0 <= index < len(tb_entries) and tb_entries:
            current_frame = tb_entries[index].StartFrame
            fn = tb_entries[index].Prefix
        for entry in storyboard.entries:
            if entry.name.startswith(fn):
                duration = entry.duration
        if not ensure_json_files(blend_dir, fn):
            self.report({'WARNING'}, f"JSON files for prefix '{fn}' not found")
            return {'CANCELLED'}
        apply_key(os.path.join(blend_dir, fn + KeydataStr), current_frame, duration)  # Apply stored keyframe data
        import_light_effects_from_json(os.path.join(blend_dir, fn + LightdataStr), current_frame, context)  # Import light effects
        update_texture_key(fn + "_", current_frame)
        add_timebind_prop(context, fn + "_", current_frame)  # Save TimeBind entry for quick access later
        self.report({'INFO'}, f"Keys loaded: {blend_dir}")
        return {'FINISHED'}
    
class DRONE_OT_LoadAllKeys(Operator):
    bl_idname = "drone.load_all_keys"
    bl_label = "Load All Keys"
    bl_description = "Import keyframe and light effect data for all storyboard entries with exported files"

    def execute(self, context):
        """
        Iterate through the storyboard and load key and light effect
        data for each entry that has exported files.

        The method ensures that the required JSON files exist for each
        prefix, optionally asking the user for a directory to copy them
        from. For prefixes where the files cannot be found, a warning is
        issued and processing continues with the next entry. Successfully
        processed prefixes are collected and included in the report.
        """
        scene = context.scene
        storyboard = scene.skybrush.storyboard
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        flog = []
        for sb_entry in storyboard.entries:
            pref = sb_entry.name.split("_")[0]
            if not ensure_json_files(blend_dir, pref):
                self.report({'WARNING'}, f"JSON files for prefix '{pref}' not found")
                continue
            apply_key(os.path.join(blend_dir, pref + KeydataStr), sb_entry.frame_start, sb_entry.duration)  # Apply keyframes for this prefix
            import_light_effects_from_json(os.path.join(blend_dir, pref + LightdataStr), sb_entry.frame_start, context)  # Import related light effects
            update_texture_key(pref + "_", sb_entry.frame_start)
            add_timebind_prop(context, pref + "_", sb_entry.frame_start)  # Remember TimeBind information
            flog.append(pref)
        if flog:
            self.report({'INFO'}, f"Keys loaded: {blend_dir} from {flog}")
        else:
            self.report({'INFO'}, "Not loaded")
        return {'FINISHED'}


class DRONE_OT_append_assets(Operator):
    """Append collections and textures from a .blend file"""

    bl_idname = "drone.append_assets"
    bl_label = "Append Assets"
    bl_description = "Append collections and textures from another Blender file"

    filepath: StringProperty(subtype="FILE_PATH")
    exclude_drone_collections: BoolProperty(
        name="Exclude Drone Collections",
        default=True,
        description="Skip collections whose name contains 'drone'",
    )

    def execute(self, context):
        path = bpy.path.abspath(self.filepath)
        if not path or not os.path.exists(path):
            self.report({'ERROR'}, "Invalid filepath")
            return {'CANCELLED'}

        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            colls = [
                name
                for name in data_from.collections
                if not (
                    self.exclude_drone_collections and "drone" in name.lower()
                )
            ]
            texs = list(data_from.textures)
            data_to.collections = colls
            data_to.textures = texs

        for coll in data_to.collections:
            if coll.name not in context.scene.collection.children:
                context.scene.collection.children.link(coll)

        add_prefix_le_tex(context)
        self.report(
            {"INFO"},
            f"Appended {len(data_to.collections)} collections and {len(data_to.textures)} textures",
        )
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class DRONE_OT_RecalcTransitionsWithKeys(Operator):
    """Recalculate transitions and reapply stored material keys"""

    bl_idname = "drone.recalc_transitions_with_keys"
    bl_label = "Recalc Transitions With Keys"
    bl_description = (
        "Save material keys for each storyboard entry, recalculate transitions, and"
        " reapply the keys to the recalculated positions"
    )

    scope: bpy.props.EnumProperty(
        items=[
            ("FROM_SELECTED", "From selected formation", ""),
            ("TO_SELECTED", "To selected formation", ""),
            ("ALL", "Entire storyboard", ""),
        ],
        name="Scope",
        description="Scope of the transition recalculation",
        default="ALL",
    )

    def execute(self, context):
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        storyboard = context.scene.skybrush.storyboard

        prefixes = []
        for sb_entry in storyboard.entries:
            pref = sb_entry.name.split("_")[0]
            prefixes.append(pref)
            export_key(context, blend_dir, pref)

        bpy.ops.skybrush.recalculate_transitions(scope=self.scope)

        for sb_entry in storyboard.entries:
            pref = sb_entry.name.split("_")[0]
            if pref in prefixes:
                apply_key(
                    os.path.join(blend_dir, pref + KeydataStr),
                    sb_entry.frame_start,
                    sb_entry.duration,
                )

        self.report({'INFO'}, f"Transitions recalculated ({self.scope}) and keys reloaded")
        return {'FINISHED'}


class DRONE_OT_UseNewDroneSpec(Operator):
    """Convert drones to the new specification and enable patched updates."""

    bl_idname = "drone.use_new_drone_spec"
    bl_label = "Convert to New Drone Spec"
    bl_description = (
        "Set drone meshes to single vertices, clear materials, rebuild geometry nodes, "
        "and force patched light effect updates."
    )

    def execute(self, context):
        collection = _find_drone_collection()
        if collection is None:
            self.report({'ERROR'}, "DroneCollection not found")
            return {'CANCELLED'}
        converted = 0
        for obj in _iter_drone_mesh_objects(collection):
            self._convert_object(obj)
            converted += 1
        drone_mesh_gn.setup_for_collection(collection)

        scene = context.scene
        scene.sbutil_use_patched_light_effects = True
        light_effects_result_patch._base_color_cache.clear()
        light_effects_result_patch._last_frame = None

        self.report({'INFO'}, f"Converted {converted} drones to the new specification")
        return {'FINISHED'}

    @staticmethod
    def _convert_object(obj):
        old_mesh = obj.data
        new_mesh = bpy.data.meshes.new(name=f"{obj.name}_SingleVert")
        new_mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
        new_mesh.update()

        obj.data = new_mesh
        obj.data.materials.clear()

        if old_mesh and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

class DRONE_OT_ApplyProximityLimit(Operator):
    """Add Limit Distance constraints for drones closer than the safety threshold."""

    bl_idname = "drone.apply_proximity_limit"
    bl_label = "Apply Proximity Limit"
    bl_description = (
        "Apply Limit Distance constraints between drones that are closer than the"
        " safety_check proximity threshold in the current frame"
    )

    def execute(self, context):
        scene = context.scene
        threshold = scene.skybrush.safety_check.proximity_warning_threshold
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        storyboard = scene.skybrush.storyboard
        curr_frame = scene.frame_current
        entries = sorted(storyboard.entries, key=lambda e: e.frame_start)

        previous_entry = None
        next_entry = None
        for entry in entries:
            if entry.frame_start <= curr_frame:
                previous_entry = entry
            elif entry.frame_start > curr_frame:
                next_entry = entry
                break

        def frame_end(entry):
            if hasattr(entry, "frame_end"):
                return int(entry.frame_end)
            duration = getattr(entry, "duration", None)
            if duration is None:
                return int(entry.frame_start)
            return int(entry.frame_start) + int(duration) - 1

        start_frame = max(curr_frame, scene.frame_start)
        if previous_entry:
            start_frame = frame_end(previous_entry)

        end_frame = int(scene.frame_end)
        if next_entry:
            end_frame = int(next_entry.frame_start)

        depsgraph = context.evaluated_depsgraph_get()
        all_drones = list(drones_collection.objects)
        drone_names = {obj.name for obj in all_drones}
        selected_drones = [
            obj for obj in context.selected_objects if obj.name in drone_names
        ]
        targets = selected_drones if selected_drones else all_drones

        evaluated_positions = {
            obj: obj.evaluated_get(depsgraph).matrix_world.translation for obj in all_drones
        }

        pair_count = 0
        processed_pairs: set[tuple[str, str]] = set()

        for obj1 in targets:
            loc1 = evaluated_positions[obj1]
            for obj2 in all_drones:
                if obj1 == obj2:
                    continue
                pair_key = tuple(sorted((obj1.name, obj2.name)))
                if pair_key in processed_pairs:
                    continue

                loc2 = evaluated_positions[obj2]
                if (loc1 - loc2).length < threshold:
                    processed_pairs.add(pair_key)
                    pair_count += 1
                    cname = f"Limit_{obj2.name}"
                    const = obj1.constraints.get(cname)
                    if not const or const.type != 'LIMIT_DISTANCE':
                        const = obj1.constraints.new('LIMIT_DISTANCE')
                        const.name = cname
                    const.limit_mode = scene.proximity_limit_mode
                    const.target = obj2
                    const.distance = threshold
                    start, end = start_frame, end_frame
                    frame_values: list[tuple[int, float]] = []
                    span = end - start
                    if not scene.proximity_skip_influence_keys:
                        if span >= 2:
                            pre_start = start - 1
                            if pre_start >= scene.frame_start:
                                frame_values.append((pre_start, 0.0))
                            frame_values.append((start, 1.0))
                            frame_values.append((end - 1, 1.0))
                            frame_values.append((end, 0.0))
                        else:
                            frame_values.append((start, 1.0))
                            frame_values.append((end, 0.0))

                        cons_fcurve = None
                        anim = getattr(obj1, "animation_data", None)
                        if anim and anim.action:
                            try:
                                cons_fcurve = anim.action.fcurves.find(
                                    f'constraints["{const.name}"].influence'
                                )
                            except Exception:
                                cons_fcurve = None

                        for frame, value in frame_values:
                            const.influence = value
                            const.keyframe_insert('influence', frame=frame)

                        if cons_fcurve is not None:
                            allowed_frames = [float(frame) for frame, _val in frame_values]
                            indices_to_remove = [
                                idx
                                for idx, key in enumerate(cons_fcurve.keyframe_points)
                                if not any(abs(key.co.x - allowed) <= 1e-3 for allowed in allowed_frames)
                            ]
                            for idx in reversed(indices_to_remove):
                                key = cons_fcurve.keyframe_points[idx]
                                cons_fcurve.keyframe_points.remove(key)
                            cons_fcurve.update()
                    else:
                        const.influence = 1.0

        self.report({'INFO'}, f"Applied proximity constraints to {pair_count} pairs")
        return {'FINISHED'}


class DRONE_OT_RemoveProximityLimit(Operator):
    """Remove all Limit Distance constraints from drones."""

    bl_idname = "drone.remove_proximity_limit"
    bl_label = "Remove Proximity Limit"
    bl_description = (
        "Remove all Limit Distance constraints from selected drones or,"
        " if none are selected, from all drones in the Drones collection"
    )

    def execute(self, context):
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        all_drones = list(drones_collection.objects)
        drone_names = {obj.name for obj in all_drones}
        selected = [
            obj for obj in context.selected_objects if obj.name in drone_names
        ]
        targets = selected if selected else all_drones

        removed = 0
        for obj in targets:
            constraints = [c for c in obj.constraints if c.type == 'LIMIT_DISTANCE']
            for c in constraints:
                obj.constraints.remove(c)
                removed += 1

        self.report(
            {'INFO'},
            f"Removed {removed} Limit Distance constraints from {len(targets)} drones",
        )
        return {'FINISHED'}


def _shape_copyloc_influence_curve(
    fcurve, handle_frames: float, key_filter=None
) -> bool:
    """Shape Copy Location influence handles with absolute offsets per key.

    Every key’s handles are pushed horizontally by ``handle_frames`` (clamped so
    they never cross neighbouring keys). Works with any number of keys. Returns
    True when the curve was updated.
    """

    keys = list(getattr(fcurve, "keyframe_points", []))
    if len(keys) < 2:
        return False

    keys.sort(key=lambda k: k.co.x)
    eps = 1e-4
    updated = False

    for idx, key in enumerate(keys):
        if key_filter and not key_filter(key):
            continue

        prev_x = keys[idx - 1].co.x if idx > 0 else None
        next_x = keys[idx + 1].co.x if idx < len(keys) - 1 else None

        key.interpolation = 'BEZIER'
        key.handle_left_type = 'FREE'
        key.handle_right_type = 'FREE'

        target_left = key.co.x - handle_frames
        target_right = key.co.x + handle_frames

        if prev_x is not None:
            min_left = prev_x + eps
            key.handle_left.x = max(target_left, min_left)
            key.handle_left.y = key.co.y
        else:
            key.handle_left.x = target_left
            key.handle_left.y = key.co.y

        if next_x is not None:
            max_right = next_x - eps
            key.handle_right.x = min(target_right, max_right)
            key.handle_right.y = key.co.y
        else:
            key.handle_right.x = target_right
            key.handle_right.y = key.co.y

        updated = True

    if updated:
        fcurve.keyframe_points.update()
    return updated


class DRONE_OT_LinearizeCopyLocationInfluence(Operator):
    """Shape Copy Location influence keys into an S-curve"""

    bl_idname = "drone.linearize_copyloc_influence"
    bl_label = "Linearize CopyLoc Influence"
    bl_description = (
        "Shape Copy Location constraint influence curves for selected drones,"
        " or all drones when none are selected"
    )

    def execute(self, context):
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        scene = context.scene
        handle_frames = getattr(scene, "copyloc_handle_frames", 5.0)

        collection_objects = list(drones_collection.objects)
        drone_names = {obj.name for obj in collection_objects}
        selected = [
            obj for obj in context.selected_objects if obj.name in drone_names
        ]
        targets = selected if selected else collection_objects

        updated = 0
        for obj in targets:
            anim = obj.animation_data
            if not anim or not anim.action:
                continue
            action = anim.action
            for const in obj.constraints:
                if const.type != 'COPY_LOCATION':
                    continue
                fcurve = action.fcurves.find(
                    f'constraints["{const.name}"].influence'
                )
                if not fcurve:
                    continue
                if _shape_copyloc_influence_curve(fcurve, handle_frames):
                    updated += 1

        if updated == 0:
            self.report({'WARNING'}, "No Copy Location influence curves updated")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Shaped {updated} Copy Location influence curves")
        return {'FINISHED'}


def _active_storyboard_entry_with_range(context):
    storyboard = getattr(getattr(context.scene, "skybrush", None), "storyboard", None)
    entries = getattr(storyboard, "entries", None)
    index = getattr(storyboard, "active_index", -1)
    current_frame = getattr(context.scene, "frame_current", 0)

    def _entry_range(entry):
        start = int(getattr(entry, "frame_start", 0))
        end_attr = getattr(entry, "frame_end", None)
        duration = getattr(entry, "duration", None)
        end = int(end_attr) if end_attr is not None else start + max(int(duration or 0), 0)
        return start, end

    def _entry_contains_frame(entry):
        start, end = _entry_range(entry)
        return start <= current_frame <= end

    entry = None
    if entries and 0 <= index < len(entries) and _entry_contains_frame(entries[index]):
        entry = entries[index]
    elif entries:
        for item in entries:
            if _entry_contains_frame(item):
                entry = item
                break

    if entry is None:
        return None, None, None, entries

    start, end = _entry_range(entry)
    return entry, start, end, entries


class DRONE_OT_LinearizeCopyLocationInfluenceActiveFormation(Operator):
    """Shape Copy Location influence keys for the active formation range"""

    bl_idname = "drone.linearize_copyloc_influence_active_formation"
    bl_label = "Linearize CopyLoc (Formation)"
    bl_description = (
        "Shape Copy Location influence curves near the active formation range for "
        "selected drones, or all drones when none are selected"
    )

    def execute(self, context):
        entry, start, end, entries = _active_storyboard_entry_with_range(context)
        if not entries:
            self.report({'WARNING'}, "No storyboard entries available")
            return {'CANCELLED'}
        if entry is None:
            self.report({'WARNING'}, "No storyboard entry contains the current frame")
            return {'CANCELLED'}

        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        handle_frames = getattr(context.scene, "copyloc_handle_frames", 5.0)

        collection_objects = list(drones_collection.objects)
        drone_names = {obj.name for obj in collection_objects}
        selected = [
            obj for obj in context.selected_objects if obj.name in drone_names
        ]
        targets = selected if selected else collection_objects

        frame_min = start - 1
        frame_max = end + 1

        def _key_in_range(key):
            try:
                frame = float(getattr(key.co, "x", None))
            except Exception:
                return False
            return frame_min <= frame <= frame_max

        updated = 0
        for obj in targets:
            anim = obj.animation_data
            if not anim or not anim.action:
                continue
            action = anim.action
            for const in obj.constraints:
                if const.type != 'COPY_LOCATION':
                    continue
                fcurve = action.fcurves.find(
                    f'constraints["{const.name}"].influence'
                )
                if not fcurve:
                    continue
                if _shape_copyloc_influence_curve(
                    fcurve, handle_frames, key_filter=_key_in_range
                ):
                    updated += 1

        if updated == 0:
            self.report({'WARNING'}, "No Copy Location influence keys in range")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Shaped {updated} Copy Location influence curves in frames {frame_min}-{frame_max}",
        )
        return {'FINISHED'}


def _set_vat_start_frame(modifier, start_frame: int) -> bool:
    node_group = getattr(modifier, "node_group", None)
    if node_group is None:
        return False

    updated = False
    interface = getattr(node_group, "interface", None)
    if interface is not None:
        for item in getattr(interface, "items_tree", []):
            if (
                getattr(item, "name", "") == "Start Frame"
                and getattr(item, "in_out", "") == "INPUT"
            ):
                try:
                    item.default_value = float(start_frame)
                    updated = True
                except Exception:
                    pass
                break

    return updated


class DRONE_OT_ShiftVatStartFrames(Operator):
    """Shift VAT geometry node Start Frame values for storyboard entries"""

    bl_idname = "drone.shift_vat_start_frames"
    bl_label = "Shift VAT Start Frames"
    bl_description = (
        "Shift Geometry Nodes VAT start frames for objects linked to storyboard entries"
    )

    def execute(self, context):
        storyboard = getattr(getattr(context.scene, "skybrush", None), "storyboard", None)
        entries = getattr(storyboard, "entries", None)
        if not entries:
            self.report({'ERROR'}, "Storyboard entries not found")
            return {'CANCELLED'}

        offset = getattr(context.scene, "vat_start_shift_frames", 0)

        updated = 0
        skipped = 0

        for entry in entries:
            target_start = int(getattr(entry, "frame_start", 0)) + offset
            obj = None
            for candidate in (entry.name, f"{entry.name}_CSV"):
                obj = bpy.data.objects.get(candidate)
                if obj is not None:
                    break

            if obj is None:
                skipped += 1
                continue

            modifier = next(
                (
                    m
                    for m in obj.modifiers
                    if m.type == "NODES"
                    and getattr(getattr(m, "node_group", None), "name", "").startswith(
                        "GN_DroneVAT_"
                    )
                ),
                None,
            )

            if modifier is None:
                skipped += 1
                continue

            if _set_vat_start_frame(modifier, target_start):
                updated += 1
            else:
                skipped += 1

        self.report(
            {'INFO'},
            f"Updated VAT start frames for {updated} entr(ies); skipped {skipped}",
        )
        return {'FINISHED'}


def _find_active_transition_range(scene):
    storyboard = getattr(getattr(scene, "skybrush", None), "storyboard", None)
    if storyboard is None:
        return None

    entries = sorted(storyboard.entries, key=lambda e: e.frame_start)
    if len(entries) < 2:
        return None

    frame = scene.frame_current
    for idx, entry in enumerate(entries):
        start = entry.frame_start
        end = start + entry.duration
        if start <= frame < end:
            if idx + 1 < len(entries):
                next_entry = entries[idx + 1]
                return end, next_entry.frame_start
            return None

    for idx in range(len(entries) - 1):
        transition_start = entries[idx].frame_start + entries[idx].duration
        transition_end = entries[idx + 1].frame_start
        if transition_start <= frame <= transition_end:
            return transition_start, transition_end
        if frame < transition_start:
            return transition_start, transition_end

    return None


def _pick_transition_keys(fcurve, start_frame, end_frame):
    keys = sorted(fcurve.keyframe_points, key=lambda k: k.co.x)
    if len(keys) < 2:
        return None

    inside = [k for k in keys if start_frame <= k.co.x <= end_frame]
    if len(inside) >= 2:
        return inside[0], inside[-1]
    if len(inside) == 1:
        single = inside[0]
        before = next((k for k in reversed(keys) if k.co.x < single.co.x), None)
        after = next((k for k in keys if k.co.x > single.co.x), None)
        if before or after:
            return before or single, after or single

    before_start = next((k for k in reversed(keys) if k.co.x <= start_frame), None)
    after_end = next((k for k in keys if k.co.x >= end_frame), None)
    if before_start and after_end and before_start != after_end:
        return before_start, after_end

    if keys[0] != keys[-1]:
        return keys[0], keys[-1]

    return None

_STAGGER_BACKUP_PROP = "sbutil_copyloc_stagger_backup"
_stagger_backup_cache: dict[int, list[dict[str, float]]] = {}


def _build_stagger_offsets(count, max_offset, layers):
    layers = max(1, layers)
    max_offset = max(0, max_offset)
    if count <= 0:
        return []

    step = max_offset / max(1, layers - 1) if layers > 1 else 0
    levels = [0.0]
    for level in range(1, layers):
        magnitude = step * level
        levels.append(magnitude)
        if len(levels) >= layers:
            break
        levels.append(-magnitude)
        if len(levels) >= layers:
            break

    offsets = []
    for idx in range(count):
        base = levels[idx % len(levels)] if levels else 0.0
        if idx // len(levels) % 2 == 1 and base != 0:
            base = -base
        offsets.append(base)
    return offsets


def _backup_keyframe_positions(fcurve, keys):
    backup = []
    for key in keys:
        try:
            index = next(
                idx
                for idx, candidate in enumerate(fcurve.keyframe_points)
                if candidate == key
            )
        except StopIteration:
            continue
        backup.append({"index": int(index), "frame": float(key.co.x)})

    if not backup:
        return False

    pointer = fcurve.as_pointer()
    try:
        fcurve[_STAGGER_BACKUP_PROP] = backup
        _stagger_backup_cache.pop(pointer, None)
    except (TypeError, RuntimeError):
        _stagger_backup_cache[pointer] = backup
    return True

class DRONE_OT_StaggerCopyLocationInfluence(Operator):
    """Offset Copy Location influence keys to stagger transitions"""

    bl_idname = "drone.stagger_copyloc_influence"
    bl_label = "Stagger CopyLoc Transition"
    bl_description = (
        "Shift Copy Location influence start/end keys around the current "
        "transition so nearby drones move alternately"
    )

    def execute(self, context):
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        transition_range = _find_active_transition_range(context.scene)
        if not transition_range:
            self.report({'ERROR'}, "No valid storyboard transition found")
            return {'CANCELLED'}

        transition_start, transition_end = transition_range
        if transition_end <= transition_start:
            self.report({'ERROR'}, "Transition range is too small to adjust")
            return {'CANCELLED'}

        collection_objects = list(drones_collection.objects)
        drone_names = {obj.name for obj in collection_objects}
        selected = [
            obj for obj in context.selected_objects if obj.name in drone_names
        ]
        targets = selected if selected else collection_objects
        if not targets:
            self.report({'ERROR'}, "No drones available for adjustment")
            return {'CANCELLED'}

        scene = context.scene
        current_frame = scene.frame_current
        max_offset = max(0, int(getattr(scene, "copyloc_stagger_frames", 0)))

        layers = max(1, int(getattr(scene, "copyloc_stagger_layers", 2)))
        offsets = _build_stagger_offsets(len(targets), max_offset, layers)

        targets_sorted = sorted(
            targets, key=lambda obj: tuple(obj.matrix_world.translation)
        )

        adjusted = 0
        for offset, obj in zip(offsets, targets_sorted):
            anim = obj.animation_data
            if not anim or not anim.action:
                continue

            for const in obj.constraints:
                if const.type != 'COPY_LOCATION':
                    continue

                fcurve = anim.action.fcurves.find(
                    f'constraints["{const.name}"].influence'
                )
                if not fcurve:
                    continue

                try:
                    influence_value = float(fcurve.evaluate(current_frame))
                except Exception:
                    influence_value = float(getattr(const, "influence", 0.0))
                if abs(influence_value) <= 1e-4 or abs(influence_value - 1.0) <= 1e-4:
                    continue

                key_pair = _pick_transition_keys(
                    fcurve, transition_start, transition_end
                )
                if key_pair is None:
                    continue

                start_key, end_key = key_pair

                _backup_keyframe_positions(fcurve, key_pair)
                available = (end_key.co.x - start_key.co.x) / 2
                if available <= 0:
                    continue

                effective_shift = min(abs(offset), max_offset, available)
                if effective_shift == 0:
                    continue
                effective_shift *= 1 if offset >= 0 else -1

                new_start = start_key.co.x + effective_shift
                new_end = end_key.co.x - effective_shift

                new_start = max(transition_start, min(new_start, transition_end))
                new_end = max(transition_start, min(new_end, transition_end))

                if new_start >= new_end:
                    midpoint = (start_key.co.x + end_key.co.x) / 2
                    new_start = midpoint - 0.001
                    new_end = midpoint + 0.001
                    if new_start >= new_end:
                        continue

                start_key.co.x = new_start
                end_key.co.x = new_end
                for key in (start_key, end_key):
                    key.interpolation = 'LINEAR'
                    key.handle_left_type = 'VECTOR'
                    key.handle_right_type = 'VECTOR'

                fcurve.update()
                adjusted += 1

        if not adjusted:
            self.report({'WARNING'}, "No Copy Location keys found to adjust")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Adjusted {adjusted} Copy Location key pairs")
        return {'FINISHED'}

class DRONE_OT_RestoreCopyLocationInfluence(Operator):
    """Restore Copy Location influence keys to their last staggered backup"""

    bl_idname = "drone.restore_copyloc_influence"
    bl_label = "Restore CopyLoc Keys"
    bl_description = (
        "Reset Copy Location influence keys to the positions saved before the "
        "last stagger operation"
    )

    def execute(self, context):
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        collection_objects = list(drones_collection.objects)
        drone_names = {obj.name for obj in collection_objects}
        selected = [obj for obj in context.selected_objects if obj.name in drone_names]
        targets = selected if selected else collection_objects

        restored = 0
        for obj in targets:
            anim = obj.animation_data
            if not anim or not anim.action:
                continue

            for const in obj.constraints:
                if const.type != 'COPY_LOCATION':
                    continue

                fcurve = anim.action.fcurves.find(
                    f'constraints["{const.name}"].influence'
                )
                if not fcurve:
                    continue

                pointer = fcurve.as_pointer()
                try:
                    backup = fcurve.get(_STAGGER_BACKUP_PROP)
                except (TypeError, RuntimeError, AttributeError):
                    backup = None
                if not backup:
                    backup = _stagger_backup_cache.get(pointer)
                if not backup:
                    continue

                try:
                    points = fcurve.keyframe_points
                    for item in backup:
                        idx = int(item.get("index", -1))
                        frame = float(item.get("frame", 0))
                        if 0 <= idx < len(points):
                            points[idx].co.x = frame
                    fcurve.update()
                    try:
                        del fcurve[_STAGGER_BACKUP_PROP]
                    except Exception:
                        pass
                    _stagger_backup_cache.pop(pointer, None)
                    restored += 1
                except Exception:
                    continue

        if not restored:
            self.report({'WARNING'}, "No backup data found to restore")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Restored {restored} Copy Location curves")
        return {'FINISHED'}

class LIGHTEFFECT_OTadd_prefix_le_tex(bpy.types.Operator):
    bl_idname = "drone.add_prefix"
    bl_label = "Add Prefix"
    bl_description = "Prefix texture and light-effect names with the configured file name"

    def execute(self, context):
        add_prefix_le_tex(context)
        return {'FINISHED'}
    
class DRONE_OT_shift_collecion(bpy.types.Operator):
    bl_idname = "drone.shift_coll_frame"
    bl_label = "ShiftCollectionFrame"
    bl_description = "Create an '_Animated' collection from the selection and shift its keys to the current frame"

    def execute(self, context):
        """Create an "_Animated" collection, link selected objects, shift keys, and finish."""
        # 新しいコレクションを作成
        shift_collection_name = context.scene.drone_key_props.file_name + "_Animated"
        shift_collection = bpy.data.collections.new(shift_collection_name)
        bpy.context.scene.collection.children.link(shift_collection)
        
        # Link objects selected in the Outliner (even if hidden in the viewport)
        selected_objects = [
            obj for obj in context.view_layer.objects if obj.select_get()
        ]
        for obj in selected_objects:
            # Link instead of move
            if obj.name not in shift_collection.objects:
                shift_collection.objects.link(obj)
        shift_collection_key(shift_collection)
        return {'FINISHED'}

class TIMEBIND_OT_entry_add(bpy.types.Operator):
    bl_idname = "timebind.entry_add"
    bl_label = "Add TimeBind Entry"
    bl_description = "Add a new TimeBind entry with a default prefix and start frame"

    def execute(self, context):
        add_timebind_prop(context, "ANYPREFIX", 0)
        return {'FINISHED'}
    
class TIMEBIND_OT_goto_startframe(bpy.types.Operator):
    bl_idname = "timebind.goto_startframe"
    bl_label = "Goto StartFrame"
    bl_description = "Jump to the storyboard start frame of the active TimeBind entry"

    def execute(self, context):
        tb_entries = context.scene.time_bind.entries
        index = context.scene.time_bind.active_index
        storyboard = context.scene.skybrush.storyboard
        if 0 <= index < len(tb_entries) and tb_entries:
            for sb_entry in storyboard.entries:
                if sb_entry.name.startswith(tb_entries[index].Prefix):
                    context.scene.frame_set(sb_entry.frame_start)
                    break
        return {'FINISHED'}

class TIMEBIND_OT_entry_remove(bpy.types.Operator):
    bl_idname = "timebind.entry_remove"
    bl_label = "Remove TimeBind Entry"
    bl_description = "Remove the active TimeBind entry from the list"

    def execute(self, context):
        tb = context.scene.time_bind
        if tb.entries and tb.active_index >= 0:
            tb.entries.remove(tb.active_index)
            tb.active_index = min(tb.active_index, len(tb.entries) - 1)
        return {'FINISHED'}
    
class TIMEBIND_OT_deselect(bpy.types.Operator):
    bl_idname = "timebind.deselect"
    bl_label = "Deselect"
    bl_description = "Clear the TimeBind entry selection"

    def execute(self, context):
        context.scene.time_bind.active_index = -1
        return {'FINISHED'}

class TIMEBIND_OT_entry_move(bpy.types.Operator):
    """Move entry up or down"""
    bl_idname = "timebind.entry_move"
    bl_label = "Move Entry"
    bl_description = "Move the active TimeBind entry up or down in the list"
    direction : bpy.props.EnumProperty(items=(('UP', "Up", ""), ('DOWN', "Down", "")))

    def execute(self, context):
        tb = context.scene.time_bind
        idx = tb.active_index
        if self.direction == 'UP' and idx > 0:
            tb.entries.move(idx, idx - 1)
            tb.active_index -= 1
        elif self.direction == 'DOWN' and idx < len(tb.entries) - 1:
            tb.entries.move(idx, idx + 1)
            tb.active_index += 1
        return {'FINISHED'}
    
class TIMEBIND_OT_refresh(bpy.types.Operator):
    """Recalculate frame offsets between storyboard and TimeBind entries
    and update matching light-effect frame ranges."""
    bl_idname = "timebind.refresh"
    bl_label = "Refresh TimeBind"
    bl_description = "Sync TimeBind entries with the storyboard and adjust related keys"

    def execute(self, context):
        scene = context.scene
        storyboard = scene.skybrush.storyboard
        light_effects = scene.skybrush.light_effects

        # Dronesコレクション参照
        drones_collection = bpy.data.collections.get("Drones")

        for bind_entry in scene.time_bind.entries:
            # Storyboardエントリ検索
            for sb_entry in storyboard.entries:
                if sb_entry.name.startswith(bind_entry.Prefix):
                    # Calculate frame offset between storyboard and TimeBind entry
                    diff = sb_entry.frame_start - bind_entry.StartFrame

                    # Apply the offset to light effect frame ranges with matching prefixes
                    for le_entry in light_effects.entries:
                        if le_entry.name.startswith(bind_entry.Prefix):
                            le_entry.frame_start += diff  # shift start frame
                            le_entry.frame_end += diff    # shift end frame

                    update_texture_key(bind_entry.Prefix, diff)
                    if bind_entry.Prefix + "_Animated" in bpy.data.collections:
                        shift_collection_key(
                            bpy.data.collections.get(bind_entry.Prefix + "_Animated"),
                            diff,
                            bind_entry.StartFrame,
                            sb_entry.duration,
                        )
                    # Drones内のオブジェクトキーを移動
                    if drones_collection:
                        move_material_keys(
                            drones_collection.objects,
                            bind_entry.StartFrame,
                            sb_entry.duration,  # Duration使う
                            diff,
                        )
                    # TimeBindをStoryboardに同期
                    bind_entry.StartFrame = sb_entry.frame_start
        return {'FINISHED'}


class TIMEBIND_OT_add_shift_prefix(bpy.types.Operator):
    bl_idname = "timebind.add_shift_prefix"
    bl_label = "Add Shift Prefix"
    bl_description = "Add active TimeBind prefix to the shift list"

    def execute(self, context):
        tb = context.scene.time_bind
        sp_list = context.scene.shift_prefix_list
        if tb.entries and 0 <= tb.active_index < len(tb.entries):
            pref = tb.entries[tb.active_index].Prefix
            if pref and all(p.Prefix != pref for p in sp_list.entries):
                new = sp_list.entries.add()
                new.Prefix = pref
                sp_list.active_index = len(sp_list.entries) - 1
        return {'FINISHED'}


class TIMEBIND_OT_remove_shift_prefix(bpy.types.Operator):
    bl_idname = "timebind.remove_shift_prefix"
    bl_label = "Remove Shift Prefix"
    bl_description = "Remove selected prefix from the shift list"

    def execute(self, context):
        sp_list = context.scene.shift_prefix_list
        idx = sp_list.active_index
        if 0 <= idx < len(sp_list.entries):
            sp_list.entries.remove(idx)
            sp_list.active_index = min(max(0, idx - 1), len(sp_list.entries) - 1)
        return {'FINISHED'}


class TIMEBIND_OT_shift_prefixes(bpy.types.Operator):
    bl_idname = "timebind.shift_prefixes"
    bl_label = "Shift Prefixes"
    bl_description = "Shift storyboard and related keys for prefixes in the shift list"

    def execute(self, context):
        scene = context.scene
        sp_list = scene.shift_prefix_list
        diff = sp_list.shift_amount
        storyboard = scene.skybrush.storyboard
        light_effects = scene.skybrush.light_effects
        drones_collection = bpy.data.collections.get("Drones")

        prefixes = [entry.Prefix.replace("_", "") for entry in sp_list.entries]
        if not prefixes:
            return {'CANCELLED'}

        reverse = diff > 0

        sb_infos = []
        ranges = []
        for sb in storyboard.entries:
            start = sb.frame_start
            duration = sb.duration
            selected = any(sb.name.startswith(p) for p in prefixes)
            sb_infos.append({"start": start, "duration": duration, "selected": selected})
            if selected:
                ranges.append((start, duration))

        sb_infos.sort(key=lambda info: info["start"])
        for prev, curr in zip(sb_infos, sb_infos[1:]):
            if prev["selected"] and curr["selected"]:
                gap_start = prev["start"] + prev["duration"]
                gap_end = curr["start"]
                if gap_end > gap_start:
                    ranges.append((gap_start, gap_end - gap_start))

        for prefix in prefixes:
            matching_sb_entries = [sb for sb in storyboard.entries if sb.name.startswith(prefix)]
            for sb_entry in sorted(
                matching_sb_entries, key=lambda e: e.frame_start, reverse=reverse
            ):
                start_frame = sb_entry.frame_start
                duration = sb_entry.duration
                sb_entry.frame_start += diff

                update_texture_key(prefix, diff, start_frame, duration)
                if prefix + "_Animated" in bpy.data.collections:
                    shift_collection_key(
                        bpy.data.collections.get(prefix + "_Animated"),
                        diff,
                        start_frame,
                        duration,
                    )

            for bind_entry in scene.time_bind.entries:
                if bind_entry.Prefix.replace("_", "") == prefix:
                    bind_entry.StartFrame += diff

            for le_entry in light_effects.entries:
                if le_entry.name.startswith(prefix):
                    le_entry.frame_start += diff
                    le_entry.frame_end += diff

        if drones_collection:
            for start_frame, duration in sorted(ranges, key=lambda r: r[0], reverse=reverse):
                move_material_keys(drones_collection.objects, start_frame, duration, diff)
                move_constraint_keys(drones_collection.objects, start_frame, duration, diff)

        return {'FINISHED'}


class TIMEBIND_OT_create_animated_collection(bpy.types.Operator):
    bl_idname = "timebind.create_animated_collection"
    bl_label = "Create Animated Collection"
    bl_description = (
        "Create a collection for the active TimeBind prefix and link selected objects"
    )

    def execute(self, context):
        scene = context.scene
        tb = scene.time_bind
        index = tb.active_index
        if index < 0 or index >= len(tb.entries):
            return {'CANCELLED'}

        prefix = tb.entries[index].Prefix.replace("_", "")
        if not prefix:
            return {'CANCELLED'}

        coll_name = prefix + "_Animated"
        coll = bpy.data.collections.get(coll_name)
        if coll is None:
            coll = bpy.data.collections.new(coll_name)
            scene.collection.children.link(coll)

        selected_objects = [
            obj for obj in context.view_layer.objects if obj.select_get()
        ]
        for obj in selected_objects:
            if obj.name not in coll.objects:
                coll.objects.link(obj)

        return {'FINISHED'}


def shift_collection_key(shift_collection, diff=None, start_frame=None, duration=None):
    # === 設定 ===
    shift_amount = bpy.context.scene.frame_current if diff is None else diff

    end_frame = start_frame + duration if start_frame is not None and duration is not None else None

    def shift_keyframes(anim_data, amount, start, end):
        if anim_data and anim_data.action:
            for fcurve in anim_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    frame = keyframe.co.x
                    if start is None or (start <= frame <= end):
                        keyframe.co.x += amount
                        keyframe.handle_left.x += amount
                        keyframe.handle_right.x += amount

    # --- コレクション内すべてのオブジェクト処理 ---
    for obj in shift_collection.all_objects:
        # オブジェクト本体
        shift_keyframes(obj.animation_data, shift_amount, start_frame, end_frame)

        # シェイプキー
        if obj.data and hasattr(obj.data, "shape_keys") and obj.data.shape_keys:
            shift_keyframes(obj.data.shape_keys.animation_data, shift_amount, start_frame, end_frame)

def move_material_keys(objects, start_frame, duration, diff):
    end_frame = start_frame + duration

    for obj in objects:
        if not obj.material_slots:
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.node_tree:
                continue

            for node in mat.node_tree.nodes:
                if not node.inputs or node.inputs[0].type != 'RGBA':
                    continue

                anim = mat.node_tree.animation_data
                if not anim or not anim.action:
                    continue

                for fcurve in anim.action.fcurves:
                    if "default_value" not in fcurve.data_path:
                        continue

                    for keyframe in fcurve.keyframe_points:
                        frame = keyframe.co.x
                        if start_frame <= frame <= end_frame:
                            keyframe.co.x += diff
                            keyframe.handle_left.x += diff
                            keyframe.handle_right.x += diff

def move_constraint_keys(objects, start_frame, duration, diff):
    end_frame = start_frame + duration

    for obj in objects:
        anim = obj.animation_data
        if not anim or not anim.action:
            continue

        for fcurve in anim.action.fcurves:
            if "constraints[" not in fcurve.data_path:
                continue

            for keyframe in fcurve.keyframe_points:
                frame = keyframe.co.x
                if start_frame <= frame <= end_frame:
                    keyframe.co.x += diff
                    keyframe.handle_left.x += diff
                    keyframe.handle_right.x += diff

def add_timebind_prop(context, prefix, frame):
    tb = context.scene.time_bind
    for entry in tb.entries:
        if entry.Prefix == prefix:
            entry.StartFrame = frame
            return
    new_entry = tb.entries.add()
    new_entry.Prefix = prefix
    new_entry.StartFrame = frame
    tb.active_index = len(tb.entries) - 1

def update_texture_key(prefix, diff, start_frame=None, duration=None):
    # 全テクスチャをチェック
    end_frame = start_frame + duration if start_frame is not None and duration is not None else None
    for tex in bpy.data.textures:
        # 名前が prefix で始まる場合のみ処理
        if tex.name.startswith(prefix):
            if tex.animation_data and tex.animation_data.action:
                action = tex.animation_data.action
                for fcurve in action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        frame = keyframe.co.x
                        if start_frame is None or (start_frame <= frame <= end_frame):
                            keyframe.co.x += diff
                            keyframe.handle_left.x += diff
                            keyframe.handle_right.x += diff

                # 更新を通知
                for fcurve in action.fcurves:
                    fcurve.update()

def add_prefix_le_tex(context):
    def prefix_light_effect_names(prefix):
        """
        LightEffectCollection 内の全 name にプレフィックスを追加する
        """
        entries = bpy.context.scene.skybrush.light_effects.entries

        for effect in entries:
            # 既に prefix が付いている場合は重複しないようにする（任意）
            if not effect.name.startswith(prefix):
                effect.name = prefix + effect.name
    props = context.scene.drone_key_props
    prefix_light_effect_names(props.file_name + "_")
    for tex in bpy.data.textures:
        # すでにプレフィックスが付いていない場合のみ追加
        if not tex.name.startswith(props.file_name + "_"):
            tex.name = props.file_name + "_" + tex.name

def set_propertygroup_from_dict(pg, data):
    """辞書データをPropertyGroupにセット（texture/meshのみ特殊処理）"""
    annotations = getattr(pg, "__annotations__", {}) or {}

    for key, value in data.items():
        annotation = annotations.get(key)
        if annotation is None and not hasattr(pg, key):
            continue  # JSONに余分なキーがあっても無視

        # 空文字や None はスキップ
        if value == "" or value is None:
            continue

        # texture / mesh は名前から検索して代入
        if key == "texture":
            tex = bpy.data.textures.get(value)
            if tex is None:
                tex = bpy.data.textures.new(name=value, type="NONE")
            setattr(pg, key, tex)
            continue

        if key == "mesh":
            obj = bpy.data.objects.get(value)
            if obj:
                setattr(pg, key, obj)
            continue

        if key == "target_collection":
            coll = bpy.data.collections.get(value)
            if coll:
                setattr(pg, key, coll)
            continue

        current_value = getattr(pg, key)

        # ネストしたPropertyGroupの場合
        if hasattr(current_value, "__annotations__") and isinstance(value, dict):
            set_propertygroup_from_dict(current_value, value)
            continue

        # EnumProperty の場合（候補外はスキップ）
        if annotation is not None and "EnumProperty" in str(type(annotation)):
            enum_items = annotation.keywords["items"]
            if value not in [item[0] for item in enum_items]:
                continue
            try:
                setattr(pg, key, value)
            except (AttributeError, TypeError, ValueError):
                continue  # read-only or incompatible, skip safely
            continue

        # 通常プロパティ
        try:
            setattr(pg, key, value)
        except (AttributeError, TypeError, ValueError):
            continue  # read-only or incompatible, skip safely

def _ensure_color_ramp_texture(effect, ramp_data):
    """Ensure a texture exists for restoring ColorRamp information."""

    texture = getattr(effect, "texture", None)
    if texture is not None:
        return texture

    if not isinstance(ramp_data, dict):
        return None

    tex_name = ramp_data.get("texture_name")
    tex_type = ramp_data.get("type") or "NONE"
    if not tex_name:
        return None

    try:
        texture = bpy.data.textures.get(tex_name)
        if texture is None:
            texture = bpy.data.textures.new(name=tex_name, type=tex_type)
        effect.texture = texture
    except Exception:
        return None

    return texture


def apply_color_ramp(texture, ramp_data):
    """ColorRamp情報をTextureに適用"""
    if not texture or not hasattr(texture, "color_ramp"):
        return
    if texture and texture.animation_data and texture.animation_data.action:
        return

    color_ramp = texture.color_ramp
    if color_ramp is None:
        return

    if isinstance(ramp_data, dict):
        elements = ramp_data.get("elements") or []
        texture.use_color_ramp = bool(ramp_data.get("use_color_ramp", True))
        if ramp_data.get("type"):
            texture.type = ramp_data["type"]
        if ramp_data.get("color_mode"):
            color_ramp.color_mode = ramp_data["color_mode"]
        if ramp_data.get("interpolation"):
            color_ramp.interpolation = ramp_data["interpolation"]
        if ramp_data.get("hue_interpolation") and hasattr(
            color_ramp, "hue_interpolation"
        ):
            color_ramp.hue_interpolation = ramp_data["hue_interpolation"]
    else:
        elements = ramp_data or []

    # 1つだけ残して全削除
    while len(color_ramp.elements) > 1:
        color_ramp.elements.remove(color_ramp.elements[-1])

    if not elements:
        return

    # 残した要素を最初のJSONデータで上書き
    color_ramp.elements[0].position = elements[0]["position"]
    color_ramp.elements[0].color = elements[0]["color"]

    # 2番目以降の要素を追加
    for point in elements[1:]:
        elem = color_ramp.elements.new(point["position"])
        elem.color = point["color"]

def import_light_effects_from_json(filepath, frame_offset, context):
    """
    JSONからLightEffectsを復元する
    frame_offset: frame_start / frame_end に加算するオフセット
    """
    scene = bpy.context.scene
    entries = scene.skybrush.light_effects.entries
    props = context.scene.drone_key_props
    for i in reversed(range(len(entries))):
        if entries[i].name.startswith(props.file_name):
            entries.remove(i)

    # JSON読み込み
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for effect_data in data:
        effect = entries.add()

        # JSONのnameキーをUIリストの名前に反映
        if "name" in effect_data:
            effect.name = effect_data["name"]

        # color_rampデータを一時退避
        color_ramp_data = effect_data.pop("color_ramp", [])

        if "frame_start" in effect_data:
            effect_data["frame_start"] += frame_offset
        if "frame_end" in effect_data:
            effect_data["frame_end"] += frame_offset

        # プロパティをセット
        set_propertygroup_from_dict(effect, effect_data)

        # ColorRamp復元
        if effect.type == "COLOR_RAMP" and color_ramp_data:
            texture = _ensure_color_ramp_texture(effect, color_ramp_data)
            apply_color_ramp(texture, color_ramp_data)

def apply_key(filepath, frame_offset, duration=0):
    from sbutil.color_key_utils import apply_color_keys_to_nearest

    drones_collection = bpy.data.collections.get("Drones")
    available_objects = list(drones_collection.objects)
    if duration != 0:
        for obj in available_objects:
            mat = obj.active_material
            anim = mat.node_tree.animation_data
            if not anim or not anim.action:
                continue
            for fcurve in anim.action.fcurves:
                for keyframe in reversed(fcurve.keyframe_points):
                    if frame_offset <= keyframe.co.x <= frame_offset + duration:
                        fcurve.keyframe_points.remove(keyframe)
    with open(filepath, "r") as f:
        color_key_data = json.load(f)

    # JSONキーをintに変換
    for d in color_key_data:
        d["keys"] = {int(k): v for k, v in d["keys"].items()}

    for data in color_key_data:
        available_objects = apply_color_keys_to_nearest(
            Vector(data["location"]),
            data["keys"],
            available_objects,
            frame_offset=frame_offset,
        )

def export_key(context, br_path, pref):
    pref = active_timebind_prefix(context, pref)

    def extract():
        for obj in drones_collection.objects:
            mat = obj.active_material
            if not mat or not mat.node_tree or not mat.node_tree.animation_data:
                continue

            action = mat.node_tree.animation_data.action
            if not action:
                continue

            # inputs[0].default_value を持つFカーブを抽出
            node_color_curves = [fc for fc in action.fcurves if 'inputs[0].default_value' in fc.data_path]
            if not node_color_curves:
                continue

            keys = {0: [], 1: [], 2: [], 3: []}
            for fc in node_color_curves:
                for kp in fc.keyframe_points:
                        frame = kp.co.x
                        value = kp.co.y
                        # 指定範囲のみ保存
                        if frame_start <= frame <= frame_start + duration:
                            keys[fc.array_index].append((frame - frame_start, value))
                        elif duration == 0:
                            keys[fc.array_index].append((frame, value))

            data.append({
                "name": obj.name,
                "location": list(obj.matrix_world.translation),
                "keys": keys
            })

        with open(os.path.join(br_path, pref + KeydataStr), "w") as f:
            json.dump(data, f)

    drones_collection = bpy.data.collections.get("Drones")
    data = []
    frame_start = 0
    duration = 0
    storyboard = context.scene.skybrush.storyboard
    for sb_entry in storyboard.entries:
        if pref == sb_entry.name.split("_")[0]:
            frame_start = sb_entry.frame_start
            duration = sb_entry.duration
            extract()
            return
    extract()



# ======= Recalculate Operator Patch =======

def _entries_for_scope(storyboard, scene, scope):
    entries = list(storyboard.entries)
    active = getattr(storyboard, "active_index", 0)
    if scope == "CURRENT_FRAME":
        frame = scene.frame_current
        for sb in entries:
            if sb.frame_start <= frame < sb.frame_start + sb.duration:
                return [sb]
        return []
    if scope == "TO_SELECTED" and 0 <= active < len(entries):
        return entries[: active + 1]
    if scope == "FROM_SELECTED" and 0 <= active < len(entries):
        return entries[active : active + 2]
    if scope == "FROM_SELECTED_TO_END" and 0 <= active < len(entries):
        return entries[active:]
    return entries

class Patched_RTOP(StoryboardOperator):
    def execute(self, context):
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        storyboard = context.scene.skybrush.storyboard
        scene = context.scene

        handle_keys = getattr(scene, "handle_keys_on_recalculate", True)

        tb = context.scene.time_bind
        original_index = tb.active_index
        tb.active_index = -1

        targets = _entries_for_scope(storyboard, context.scene, self.scope)
        prefixes = [sb.name.split("_")[0] for sb in targets]
        if handle_keys:
            for pref in prefixes:
                print(pref)
                export_key(context, blend_dir, pref)

        tb.active_index = original_index
        result = self._original_execute(context)

        if handle_keys:
            for pref in prefixes:
                sb_entry = next(
                    (sb for sb in storyboard.entries if sb.name.startswith(pref)),
                    None,
                )
                if not sb_entry:
                    continue
                apply_key(
                    os.path.join(blend_dir, pref + KeydataStr),
                    sb_entry.frame_start,
                    sb_entry.duration,
                )

        return result


def patch_recalculate_operator():
    RecalculateTransitionsOperator._original_execute = RecalculateTransitionsOperator.execute
    RecalculateTransitionsOperator.execute = Patched_RTOP.execute
    panel = _get_storyboard_editor_panel()
    if panel and not getattr(panel, "_SBUTIL_KEYS_DRAW_APPENDED", False):
        panel.append(_draw_recalculate_key_handling)
        panel._SBUTIL_KEYS_DRAW_APPENDED = True


def patch_storyboard_panel_extras():
    panel = _get_storyboard_editor_panel()
    if panel and not getattr(panel, "_SBUTIL_UTILS_DRAW_APPENDED", False):
        panel.append(_draw_storyboard_extras)
        panel._SBUTIL_UTILS_DRAW_APPENDED = True

def unpatch_recalculate_operator():
    if RecalculateTransitionsOperator._original_execute:
        RecalculateTransitionsOperator.execute = RecalculateTransitionsOperator._original_execute
        RecalculateTransitionsOperator._original_execute = None
    panel = _get_storyboard_editor_panel()
    if panel and getattr(panel, "_SBUTIL_KEYS_DRAW_APPENDED", False):
        try:
            panel.remove(_draw_recalculate_key_handling)
        except Exception:
            pass
        panel._SBUTIL_KEYS_DRAW_APPENDED = False


def unpatch_storyboard_panel_extras():
    panel = _get_storyboard_editor_panel()
    if panel and getattr(panel, "_SBUTIL_UTILS_DRAW_APPENDED", False):
        try:
            panel.remove(_draw_storyboard_extras)
        except Exception:
            pass
        panel._SBUTIL_UTILS_DRAW_APPENDED = False


def _get_storyboard_editor_panel():
    module_spec = importlib.util.find_spec("sbstudio.plugin.panels.storyboard_editor")
    if module_spec is None:
        return None
    module = importlib.import_module("sbstudio.plugin.panels.storyboard_editor")
    return getattr(module, "StoryboardEditor", None)


def _draw_recalculate_key_handling(self, context):
    scene = context.scene
    layout = self.layout
    layout.separator()
    col = layout.column()
    col.use_property_split = True
    col.prop(scene, "handle_keys_on_recalculate", text="Export & Apply Keys")


def _draw_storyboard_extras(self, context):
    layout = self.layout
    layout.separator()
    col = layout.column(align=True)
    col.label(text="SBUtil")
    col.operator(
        SBUTIL_OT_SetRenderRangeFromStoryboard.bl_idname,
        text="Render Range from Storyboard",
        icon='PREVIEW_RANGE',
    )
    col.operator(
        SBUTIL_OT_IsolateActiveFormation.bl_idname,
        text="Show Active Formation",
        icon='HIDE_OFF',
    )
    col.prop(context.scene, "copyloc_handle_frames", slider=True)
    col.operator(
        "drone.linearize_copyloc_influence_active_formation",
        text="Linearize CopyLoc (Formation)",
    )
    col.operator("drone.linearize_copyloc_influence", text="Linearize CopyLoc")
    col.prop(context.scene, "vat_start_shift_frames", text="VAT Start Offset")
    col.operator("drone.shift_vat_start_frames", text="Shift VAT Start")


def _draw_auto_proximity_toggle(self, context):  # pragma: no cover - Blender UI
    self.layout.prop(
        context.scene,
        "auto_proximity_check",
        text="Auto Check",
    )


def patch_safety_check_panel():  # pragma: no cover - executed in Blender
    try:
        from sbstudio.plugin.panels import SafetyCheckPanel
    except Exception:
        return
    SafetyCheckPanel.append(_draw_auto_proximity_toggle)


def unpatch_safety_check_panel():  # pragma: no cover - executed in Blender
    try:
        from sbstudio.plugin.panels import SafetyCheckPanel
    except Exception:
        return
    try:
        SafetyCheckPanel.remove(_draw_auto_proximity_toggle)
    except Exception:
        pass


@persistent
def _auto_run_proximity_check(scene, _depsgraph):  # pragma: no cover - Blender handler
    if not getattr(scene, "auto_proximity_check", False):
        return
    op = bpy.ops
    for part in _RUN_FULL_PROXIMITY_OP.split("."):
        op = getattr(op, part)
    try:
        op("INVOKE_DEFAULT")
    except Exception:
        pass


def try_patch():
    global _PATCHED
    try:
        if _PATCHED:
            bpy.app.timers.unregister(try_patch)
            return None
        patch_recalculate_operator()
        patch_storyboard_panel_extras()
        #recalculate_transitions_patch.patch_recalculate_transitions()
        formation_patch.patch_create_formation_operator()
        storyboard_patch.patch_storyboard_entry_removal()
        light_effects_patch.patch_light_effect_collection()
        light_effects_patch.patch_light_effect_class()
        light_effects_patch.patch_light_effects_panel()
        light_effects_result_patch.patch_light_effect_results()
        patch_safety_check_panel()
        print("patch success!")
    except Exception as e:
        print(e)
        return 0.5
    _PATCHED = True
    for area in bpy.context.screen.areas:
        area.tag_redraw()
    bpy.app.timers.unregister(try_patch)
    return None

# --- ファイル読み込み後にも再適用したい場合（原状復帰→再適用が安全） ---
def _on_load_post(_dummy):
    # 原状復帰→再適用（複数回読み込みに強くする）
    _restore_originals()
    # 遅延パッチも再スケジュール
    bpy.app.timers.register(try_patch, first_interval=0.1)

def _restore_originals():
    try:
        unpatch_recalculate_operator()
        #recalculate_transitions_patch.unpatch_recalculate_transitions()
        formation_patch.unpatch_create_formation_operator()
        storyboard_patch.unpatch_storyboard_entry_removal()
        unpatch_storyboard_panel_extras()
        light_effects_patch.unpatch_light_effect_collection()
        light_effects_patch.unpatch_light_effects_panel()
        light_effects_patch.unpatch_light_effect_class()
        light_effects_result_patch.unpatch_light_effect_results()
        unpatch_safety_check_panel()
    except:
        return
# -------------------------------
# UIパネル
# -------------------------------
class DRONE_PT_KeyTransfer(Panel):
    bl_label = "Drone Transfer"
    bl_idname = "DRONE_PT_key_transfer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        """Construct the key transfer panel layout.

        Displays the file-name property, buttons for saving and loading
        data, the time-bind entry list, and controls for editing the
        selected entry.
        """
        layout = self.layout
        tb = context.scene.time_bind

        # File name property
        layout.prop(context.scene.drone_key_props, "file_name")

        # Operators
        layout.operator("drone.save_keys", text="Save")
        layout.operator("drone.save_single_keys", text="Key Save")
        layout.operator("drone.save_light_effects", text="Light Save")
        layout.operator("drone.load_light_effects", text="Light Load")
        layout.operator("drone.load_keys", text="Load")
        layout.operator("drone.load_all_keys", text="All Load")
        layout.operator("drone.append_assets", text="Append Assets")
        layout.operator("drone.add_prefix", text="Add Prefix")
        layout.operator("drone.shift_coll_frame", text="Shift Collection")
        layout.operator_menu_enum(
            "drone.recalc_transitions_with_keys",
            "scope",
            text="Recalc Reload",
        )
        layout.operator("timebind.goto_startframe", text="Goto Start")

        # Refresh button
        layout.operator("timebind.refresh", text="Refresh", icon='FILE_REFRESH')

        # Entry list and controls
        row = layout.row()
        row.template_list(
            "TIMEBIND_UL_entries", "",
            tb, "entries",
            tb, "active_index"
        )

        # Control buttons
        col = row.column(align=True)
        col.operator("timebind.entry_add", icon='ADD', text="")
        col.operator("timebind.entry_remove", icon='REMOVE', text="")
        col.separator()
        col.operator("timebind.deselect", icon='RESTRICT_SELECT_ON', text="")
        col.separator()
        col.operator("timebind.entry_move", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("timebind.entry_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

        # Entry editing
        if tb.entries and tb.active_index >= 0:
            entry = tb.entries[tb.active_index]
            box = layout.box()
            box.prop(entry, "Prefix")
            box.prop(entry, "StartFrame")

        # Shift prefix list and controls
        sp = context.scene.shift_prefix_list
        row = layout.row()
        row.template_list(
            "TIMEBIND_UL_shift_prefixes", "",
            sp, "entries",
            sp, "active_index",
        )
        col = row.column(align=True)
        col.operator("timebind.add_shift_prefix", icon='ADD', text="")
        col.operator("timebind.remove_shift_prefix", icon='REMOVE', text="")

        layout.prop(sp, "shift_amount")
        layout.operator("timebind.shift_prefixes", text="Shift")
        layout.operator(
            "timebind.create_animated_collection", text="Create Animated Collection"
        )

# -------------------------------
# Utility panel
# -------------------------------


class DRONE_PT_Utilities(Panel):
    bl_label = "Drone Utilities"
    bl_idname = "DRONE_PT_utilities"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        layout = self.layout
        # Some Blender builds complain about specific icon enums; fall back gracefully
        try:
            layout.operator(
                "drone.use_new_drone_spec",
                text="Convert to New Drone Spec",
                icon='NODETREE',
            )
        except TypeError:
            layout.operator("drone.use_new_drone_spec", text="Convert to New Drone Spec")
        if getattr(context.scene, "sbutil_use_patched_light_effects", False):
            layout.label(text="Patched light effects enabled", icon='CHECKMARK')
        layout.separator()
        layout.prop(context.scene, "proximity_limit_mode", text="Clamp Region")
        layout.prop(context.scene, "proximity_skip_influence_keys")
        layout.operator(
            "drone.apply_proximity_limit", text="Apply Proximity Limit"
        )
        layout.operator(
            "drone.remove_proximity_limit", text="Remove Proximity Limit"
        )
        layout.prop(context.scene, "copyloc_handle_frames", slider=True)
        layout.operator(
            "drone.linearize_copyloc_influence_active_formation",
            text="Linearize CopyLoc (Formation)",
        )
        layout.operator(
            "drone.linearize_copyloc_influence", text="Linearize CopyLoc"
        )
        layout.prop(context.scene, "copyloc_stagger_frames")
        layout.prop(context.scene, "copyloc_stagger_layers")
        layout.operator(
            "drone.stagger_copyloc_influence", text="Stagger CopyLoc Transition"
        )
        layout.operator(
            "drone.restore_copyloc_influence", text="Restore CopyLoc Keys"
        )
        layout.operator("mesh.reflow_vertices", text="Reflow Vertices")
        layout.operator("mesh.repel_from_neighbors", text="Repel From Neighbors")
        layout.separator()
        layout.operator("drone.apply_drone_check_gn", text="Apply Drone Check GN")
        row = layout.row(align=True)
        row.operator(
            "drone.enable_drone_check_circle",
            text="Enable Check Circle",
        )
        row.operator(
            "drone.disable_drone_check_circle",
            text="Disable Check Circle",
        )
        layout.operator("drone.remove_drone_check_gn", text="Remove Drone Check GN")
        layout.separator()
        if hasattr(context.scene, "sbutil_camera_margin"):
            layout.prop(context.scene, "sbutil_camera_margin", slider=True)
        row = layout.row(align=True)
        row.operator("sbutil.setup_glare_compositor", text="Setup Glare")
        row.operator("sbutil.frame_from_neg_y", text="Frame Camera")


class SBUTIL_PT_StoryboardBatchExport(Panel):
    bl_label = "Storyboard CSV Batch"
    bl_idname = "SBUTIL_PT_storyboard_batch_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.sbutil_storyboard_export

        row = layout.row(align=True)
        row.operator(SBUTIL_OT_LoadStoryboardBatch.bl_idname, icon='FILE_REFRESH')
        row.prop(settings, "include_transitions", text="Transitions")

        header = layout.row(align=True)
        header.label(text="")
        header.label(text="Name")
        header.label(text="Start")
        header.label(text="End")
        header.label(text="Dur")

        layout.template_list(
            SBUTIL_UL_StoryboardBatch.bl_idname,
            "",
            settings,
            "entries",
            settings,
            "active_index",
        )

        layout.label(text="Frame rate fixed to 24 FPS; render range updates per item.")
        layout.operator(
            SBUTIL_OT_ExportStoryboardBatch.bl_idname,
            text="Export Checked Entries",
            icon='EXPORT',
        )

# -------------------------------
# Add-on Preferences
# -------------------------------

class SBUTIL_AddonPreferences(AddonPreferences):
    bl_idname = __name__
    bl_label = "SkyBrush Util Preferences"

    def draw(self, context):
        layout = self.layout
        layout.operator("drone.update_addon", text="Update Add-on")

# -------------------------------
# 登録
# -------------------------------
classes = (
    DroneKeyTransferProperties,
    DRONE_OT_shift_collecion,
    DRONE_OT_SaveKeys,
    DRONE_OT_SaveSignleKeys,
    DRONE_OT_SaveLightEffects,
    DRONE_OT_LoadLightEffects,
    DRONE_OT_LoadKeys,
    SBUTIL_AddonPreferences,
    DRONE_OT_UpdateAddon,
    DRONE_PT_KeyTransfer,
    DRONE_PT_Utilities,
    DRONE_OT_LoadAllKeys,
    DRONE_OT_append_assets,
    DRONE_OT_RecalcTransitionsWithKeys,
    DRONE_OT_UseNewDroneSpec,
    DRONE_OT_ApplyProximityLimit,
    DRONE_OT_RemoveProximityLimit,
    DRONE_OT_LinearizeCopyLocationInfluence,
    DRONE_OT_LinearizeCopyLocationInfluenceActiveFormation,
    DRONE_OT_ShiftVatStartFrames,
    DRONE_OT_StaggerCopyLocationInfluence,
    DRONE_OT_RestoreCopyLocationInfluence,
    LIGHTEFFECT_OTadd_prefix_le_tex,
    TIMEBIND_OT_goto_startframe,
    TimeBindEntry,
    ShiftPrefixEntry,
    ShiftPrefixList,
    TimeBindCollection,
    TIMEBIND_UL_entries,
    TIMEBIND_UL_shift_prefixes,
    TIMEBIND_OT_entry_add,
    TIMEBIND_OT_entry_remove,
    TIMEBIND_OT_entry_move,
    TIMEBIND_OT_deselect,
    TIMEBIND_OT_refresh,
    TIMEBIND_OT_add_shift_prefix,
    TIMEBIND_OT_remove_shift_prefix,
    TIMEBIND_OT_create_animated_collection,
    TIMEBIND_OT_shift_prefixes,
    SBUTIL_StoryboardBatchItem,
    SBUTIL_StoryboardBatchSettings,
    SBUTIL_UL_StoryboardBatch,
    SBUTIL_OT_LoadStoryboardBatch,
    SBUTIL_OT_ExportStoryboardBatch,
    SBUTIL_OT_SetRenderRangeFromStoryboard,
    SBUTIL_OT_IsolateActiveFormation,
    SBUTIL_PT_StoryboardBatchExport,
)
_PATCHED = False
_ORIGINALS = {}  # {("module.path","attr_name"): original_obj}

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.drone_key_props = bpy.props.PointerProperty(type=DroneKeyTransferProperties)
    bpy.types.Scene.time_bind = bpy.props.PointerProperty(type=TimeBindCollection)
    bpy.types.Scene.shift_prefix_list = bpy.props.PointerProperty(type=ShiftPrefixList)
    bpy.types.Scene.sbutil_storyboard_export = bpy.props.PointerProperty(
        type=SBUTIL_StoryboardBatchSettings
    )
    bpy.types.Scene.sbutil_use_patched_light_effects = BoolProperty(
        name="Use Patched Light Effects",
        description=(
            "Keep using the patched light effect updater after converting drones to the new specification"
        ),
        default=False,
    )
    bpy.types.Scene.sbutil_update_light_effects = BoolProperty(
        name="Update Light Effects",
        description="Run the SBUtil light effect updater on frame changes",
        default=True,
    )
    bpy.types.Scene.auto_proximity_check = BoolProperty(
        name="Auto Proximity Check",
        description="Run full proximity check when frame changes",
        default=False,
    )
    bpy.types.Scene.proximity_limit_mode = bpy.props.EnumProperty(
        name="Proximity Clamp Region",
        description="Clamp region used by generated proximity Limit Distance constraints",
        items=[
            (
                'LIMITDIST_ONSURFACE',
                'On Surface',
                'Keep objects exactly on the surface of the defined distance',
            ),
            (
                'LIMITDIST_OUTSIDE',
                'Outside',
                'Keep objects outside the defined distance',
            ),
        ],
        default='LIMITDIST_ONSURFACE',
    )
    bpy.types.Scene.proximity_skip_influence_keys = bpy.props.BoolProperty(
        name="Skip Influence Keys",
        description="Do not insert influence keyframes when applying proximity limits",
        default=False,
    )
    bpy.types.Scene.handle_keys_on_recalculate = BoolProperty(
        name="Export && Apply Keys on Recalc",
        description=(
            "Export keyframes before recalculation and re-apply them afterwards"
        ),
        default=False,
    )
    bpy.types.Scene.vat_start_shift_frames = bpy.props.IntProperty(
        name="VAT Start Offset",
        description=(
            "Offset added to storyboard start when shifting VAT geometry node start frames"
        ),
        default=0,
    )
    bpy.types.Scene.copyloc_handle_frames = bpy.props.FloatProperty(
        name="CopyLoc Handle Frames",
        description=(
            "Absolute frame distance to move Copy Location influence handles; "
            "applies to first/last keys"
        ),
        default=5.0,
        min=0.0,
        soft_max=50.0,
        step=1,
    )
    bpy.types.Scene.copyloc_stagger_frames = bpy.props.IntProperty(
        name="CopyLoc Offset Frames",
        description=(
            "Maximum number of frames to shift Copy Location influence keys "
            "when staggering transitions"
        ),
        default=2,
        min=0,
    )
    bpy.types.Scene.copyloc_stagger_layers = bpy.props.IntProperty(
        name="CopyLoc Layers",
        description=(
            "Number of offset layers to cycle when staggering Copy Location "
            "influence keys"
        ),
        default=2,
        min=1,
    )
    light_effects_patch.register()
    CSV2Vertex.register()
    reflow_vertex.register()
    drone_check_gn.register()
    view_setup.register()
    bpy.app.timers.register(try_patch)
    bpy.app.handlers.load_post.append(_on_load_post)
    if _auto_run_proximity_check not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_auto_run_proximity_check)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.drone_key_props
    del bpy.types.Scene.time_bind
    del bpy.types.Scene.shift_prefix_list
    if hasattr(bpy.types.Scene, "sbutil_storyboard_export"):
        del bpy.types.Scene.sbutil_storyboard_export
    if hasattr(bpy.types.Scene, "sbutil_use_patched_light_effects"):
        del bpy.types.Scene.sbutil_use_patched_light_effects
    if hasattr(bpy.types.Scene, "sbutil_update_light_effects"):
        del bpy.types.Scene.sbutil_update_light_effects
    if hasattr(bpy.types.Scene, "auto_proximity_check"):
        del bpy.types.Scene.auto_proximity_check
    if hasattr(bpy.types.Scene, "proximity_limit_mode"):
        del bpy.types.Scene.proximity_limit_mode
    if hasattr(bpy.types.Scene, "proximity_skip_influence_keys"):
        del bpy.types.Scene.proximity_skip_influence_keys
    if hasattr(bpy.types.Scene, "handle_keys_on_recalculate"):
        del bpy.types.Scene.handle_keys_on_recalculate
    if hasattr(bpy.types.Scene, "vat_start_shift_frames"):
        del bpy.types.Scene.vat_start_shift_frames
    if hasattr(bpy.types.Scene, "copyloc_handle_frames"):
        del bpy.types.Scene.copyloc_handle_frames
    if hasattr(bpy.types.Scene, "copyloc_stagger_frames"):
        del bpy.types.Scene.copyloc_stagger_frames
    if hasattr(bpy.types.Scene, "copyloc_stagger_layers"):
        del bpy.types.Scene.copyloc_stagger_layers
    light_effects_patch.unregister()
    CSV2Vertex.unregister()
    reflow_vertex.unregister()
    drone_check_gn.unregister()
    view_setup.unregister()

    # ハンドラ除去（存在チェック）
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _auto_run_proximity_check in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_auto_run_proximity_check)

    # 元に戻す
    _restore_originals()
    if bpy.app.timers.is_registered(try_patch):
        bpy.app.timers.unregister(try_patch)

if __name__ == "__main__":
    register()
