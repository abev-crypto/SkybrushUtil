bl_info = {
    "name": "SkyBrushUtil",
    "author": "ABEYUYA",
    "version": (1, 9),
    "blender": (4, 3, 0),
    "location": "3D View > Sidebar > SBUtil",
    "description": "SkybrushTransfarUtil",
    "category": "Animation",
}

import bpy
from bpy.props import StringProperty
from bpy.types import Panel, Operator, PropertyGroup
from mathutils import Vector
import json, os
from sbstudio.plugin.operators import RecalculateTransitionsOperator
from sbstudio.plugin.operators.base import StoryboardOperator


KeydataStr = "_KeyData.json"
LightdataStr = "_LightData.json"

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
    for prop in pg.__annotations__.keys():
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

        A nested ``check_file`` helper scans a directory for files that
        start with a given prefix. The method uses this helper on each
        storyboard entry to decide whether to apply the stored keyframes
        and light effect definitions. It also updates textures and
        time-bind properties and collects the prefixes of successfully
        processed entries to include in the report.
        """
        def check_file(dir, name):
            # ディレクトリ内の全ファイルを取得
            for file_name in os.listdir(dir):
                if file_name.startswith(name):
                    return True  # そのPrefixのファイルが存在
            return False  # 存在しない場合
        scene = context.scene
        storyboard = scene.skybrush.storyboard
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        flog = []
        for sb_entry in storyboard.entries:
            pref = sb_entry.name.split("_")[0]
            if check_file(blend_dir, pref):
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
        if key not in pg.__annotations__:
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
    def find_nearest_object(location):
        nearest_obj = None
        min_dist = float('inf')
        for obj in available_objects:
            dist = (Vector(location) - obj.matrix_world.translation).length
            if dist < min_dist:
                min_dist = dist
                nearest_obj = obj
        return nearest_obj
    
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
        nearest_obj = find_nearest_object(Vector(data["location"]))
        mat = nearest_obj.active_material
        for node in mat.node_tree.nodes:
            if not node.inputs or node.inputs[0].type != 'RGBA':
                continue
            # キー移植
            for channel, keyframes in data["keys"].items():
                for frame, value in keyframes:
                    node.inputs[0].default_value[channel] = value
                    node.inputs[0].keyframe_insert(
                        "default_value",
                        frame=frame_offset + frame,
                        index=channel
                    )
            break
        available_objects.remove(nearest_obj)

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
_original_execute = None


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
    if _original_execute is None:
        RecalculateTransitionsOperator._original_execute = RecalculateTransitionsOperator.execute
        RecalculateTransitionsOperator.execute = Patched_RTOP.execute
        print("patch success!")


def unpatch_recalculate_operator():
    if _original_execute:
        RecalculateTransitionsOperator.execute = _original_execute
        RecalculateTransitionsOperator._original_execute = None


def try_patch():
    try:
        patch_recalculate_operator()
    except Exception:
        return 0.5
    bpy.app.timers.unregister(try_patch)
    return None


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
# 登録
# -------------------------------
classes = (
    DroneKeyTransferProperties,
    DRONE_OT_shift_collecion,
    DRONE_OT_SaveKeys,
    DRONE_OT_SaveSignleKeys,
    DRONE_OT_SaveLightEffects,
    DRONE_OT_LoadKeys,
    DRONE_PT_KeyTransfer,
    DRONE_OT_LoadAllKeys,
    DRONE_OT_RecalcTransitionsWithKeys,
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

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.drone_key_props = bpy.props.PointerProperty(type=DroneKeyTransferProperties)
    bpy.types.Scene.time_bind = bpy.props.PointerProperty(type=TimeBindCollection)
    bpy.types.Scene.shift_prefix_list = bpy.props.PointerProperty(type=ShiftPrefixList)
    bpy.app.timers.register(try_patch)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.drone_key_props
    del bpy.types.Scene.time_bind
    del bpy.types.Scene.shift_prefix_list
    if bpy.app.timers.is_registered(try_patch):
        bpy.app.timers.unregister(try_patch)
    unpatch_recalculate_operator()

if __name__ == "__main__":
    register()
