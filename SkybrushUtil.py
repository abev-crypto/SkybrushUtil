bl_info = {
    "name": "SkyBrushUtil",
    "author": "ABEYUYA",
    "version": (2, 3, 1),
    "blender": (4, 3, 2),
    "location": "3D View > Sidebar > SBUtil",
    "description": "SkybrushTransfarUtil",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty
from bpy.types import Panel, Operator, PropertyGroup, AddonPreferences
from mathutils import Vector
import json, os, shutil, tempfile, urllib.request
from sbstudio.plugin.operators import RecalculateTransitionsOperator
from sbstudio.plugin.operators.base import StoryboardOperator
from sbutil import formation_patch
from sbutil import light_effects as light_effects_patch
from sbutil import CSV2Vertex
from sbutil import reflow_vertex
from sbutil import drone_check_gn
from sbutil import view_setup

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
    data = []
    if texture and hasattr(texture, "color_ramp"):
        color_ramp = texture.color_ramp
        for elem in color_ramp.elements:
            data.append({
                "position": elem.position,
                "color": [round(c, 3) for c in elem.color],
            })
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
        data["color_ramp"] = convert_color_ramp(texture)

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

        prev_start = entries[0].frame_start if entries else 0
        next_start = scene.frame_end
        for entry in entries:
            if entry.frame_start <= curr_frame:
                prev_start = entry.frame_start
            elif entry.frame_start > curr_frame:
                next_start = entry.frame_start
                break

        depsgraph = context.evaluated_depsgraph_get()
        objects = list(drones_collection.objects)
        pair_count = 0

        for i, obj1 in enumerate(objects):
            loc1 = obj1.evaluated_get(depsgraph).matrix_world.translation
            for obj2 in objects[i + 1:]:
                loc2 = obj2.evaluated_get(depsgraph).matrix_world.translation
                if (loc1 - loc2).length < threshold:
                    pair_count += 1
                    cname = f"Limit_{obj2.name}"
                    const = obj1.constraints.get(cname)
                    if not const or const.type != 'LIMIT_DISTANCE':
                        const = obj1.constraints.new('LIMIT_DISTANCE')
                        const.name = cname
                        const.limit_mode = 'LIMITDIST_ONSURFACE'
                    const.target = obj2
                    const.distance = threshold
                    start, end = prev_start, next_start
                    if end - start >= 2:
                        const.influence = 0.0
                        const.keyframe_insert('influence', frame=start)
                        const.influence = 1.0
                        const.keyframe_insert('influence', frame=start + 1)
                        const.keyframe_insert('influence', frame=end - 1)
                        const.influence = 0.0
                        const.keyframe_insert('influence', frame=end)
                    else:
                        const.influence = 1.0
                        const.keyframe_insert('influence', frame=start)
                        const.keyframe_insert('influence', frame=end)

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

        selected = [
            obj for obj in context.selected_objects if obj in drones_collection.objects
        ]
        targets = selected if selected else list(drones_collection.objects)

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

class DRONE_OT_LinearizeCopyLocationInfluence(Operator):
    """Set Copy Location constraint influence keys to linear"""

    bl_idname = "drone.linearize_copyloc_influence"
    bl_label = "Linearize CopyLoc Influence"
    bl_description = (
        "Set Copy Location constraint influence keyframe handles to linear for"
        " selected drones or all drones when none are selected"
    )

    def execute(self, context):
        drones_collection = bpy.data.collections.get("Drones")
        if not drones_collection:
            self.report({'ERROR'}, "Drones collection not found")
            return {'CANCELLED'}

        selected = [
            obj for obj in context.selected_objects if obj in drones_collection.objects
        ]
        targets = selected if selected else drones_collection.objects

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
                for key in fcurve.keyframe_points:
                    key.interpolation = 'LINEAR'
                    key.handle_left_type = 'VECTOR'
                    key.handle_right_type = 'VECTOR'
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
    for key, value in data.items():
        if key not in getattr(pg, "__annotations__", {}) and not hasattr(pg, key):
            continue  # JSONに余分なキーがあっても無視

        # 空文字や None はスキップ
        if value == "" or value is None:
            continue

        # texture / mesh は名前から検索して代入
        if key == "texture":
            tex = bpy.data.textures.get(value)
            if tex:
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
        if "EnumProperty" in str(type(pg.__annotations__[key])):
            enum_items = pg.__annotations__[key].keywords["items"]
            if value not in [item[0] for item in enum_items]:
                continue
            setattr(pg, key, value)
            continue

        # 通常プロパティ
        setattr(pg, key, value)

def apply_color_ramp(texture, ramp_data):
    """ColorRamp情報をTextureに適用"""
    if not texture or not hasattr(texture, "color_ramp"):
        return
    if texture and texture.animation_data and texture.animation_data.action:
        return

    color_ramp = texture.color_ramp

    # 1つだけ残して全削除
    while len(color_ramp.elements) > 1:
        color_ramp.elements.remove(color_ramp.elements[-1])

    # 残した要素を最初のJSONデータで上書き
    if ramp_data:
        color_ramp.elements[0].position = ramp_data[0]["position"]
        color_ramp.elements[0].color = ramp_data[0]["color"]

    # 2番目以降の要素を追加
    for point in ramp_data[1:]:
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

        # ColorRamp復元
        if effect.type == "COLOR_RAMP" and color_ramp_data:
            apply_color_ramp(effect.texture, color_ramp_data)

        # プロパティをセット
        set_propertygroup_from_dict(effect, effect_data)

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

        tb = context.scene.time_bind
        original_index = tb.active_index
        tb.active_index = -1

        targets = _entries_for_scope(storyboard, context.scene, self.scope)
        prefixes = [sb.name.split("_")[0] for sb in targets]
        for pref in prefixes:
            print(pref)
            export_key(context, blend_dir, pref)

        tb.active_index = original_index
        result = self._original_execute(context)

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

def unpatch_recalculate_operator():
    if RecalculateTransitionsOperator._original_execute:
        RecalculateTransitionsOperator.execute = RecalculateTransitionsOperator._original_execute
        RecalculateTransitionsOperator._original_execute = None


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
        formation_patch.patch_create_formation_operator()
        light_effects_patch.patch_light_effect_collection()
        light_effects_patch.patch_light_effect_class()
        light_effects_patch.patch_light_effects_panel()
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
        formation_patch.unpatch_create_formation_operator()
        light_effects_patch.unpatch_light_effect_collection()
        light_effects_patch.unpatch_light_effects_panel()
        light_effects_patch.unpatch_light_effect_class()
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
        layout.operator(
            "drone.apply_proximity_limit", text="Apply Proximity Limit"
        )
        layout.operator(
            "drone.remove_proximity_limit", text="Remove Proximity Limit"
        )
        layout.operator(
            "drone.linearize_copyloc_influence", text="Linearize CopyLoc"
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
    DRONE_OT_LoadKeys,
    SBUTIL_AddonPreferences,
    DRONE_OT_UpdateAddon,
    DRONE_PT_KeyTransfer,
    DRONE_PT_Utilities,
    DRONE_OT_LoadAllKeys,
    DRONE_OT_append_assets,
    DRONE_OT_RecalcTransitionsWithKeys,
    DRONE_OT_ApplyProximityLimit,
    DRONE_OT_RemoveProximityLimit,
    DRONE_OT_LinearizeCopyLocationInfluence,
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
)
_PATCHED = False
_ORIGINALS = {}  # {("module.path","attr_name"): original_obj}

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.drone_key_props = bpy.props.PointerProperty(type=DroneKeyTransferProperties)
    bpy.types.Scene.time_bind = bpy.props.PointerProperty(type=TimeBindCollection)
    bpy.types.Scene.shift_prefix_list = bpy.props.PointerProperty(type=ShiftPrefixList)
    bpy.types.Scene.auto_proximity_check = BoolProperty(
        name="Auto Proximity Check",
        description="Run full proximity check when frame changes",
        default=False,
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
    if hasattr(bpy.types.Scene, "auto_proximity_check"):
        del bpy.types.Scene.auto_proximity_check
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
