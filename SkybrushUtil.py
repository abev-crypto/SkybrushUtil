bl_info = {
    "name": "SkyBrushUtil",
    "author": "ABEYUYA",
    "version": (1, 7),
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


KeydataStr = "_KeyData.json"
LightdataStr = "_LightData.json"

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

# コレクション全体の管理用
class TimeBindCollection(bpy.types.PropertyGroup):
    entries : bpy.props.CollectionProperty(type=TimeBindEntry)
    active_index : bpy.props.IntProperty()

class TIMEBIND_UL_entries(bpy.types.UIList):
    """TimeBind entriesを表示するUIList"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item: TimeBindEntry
        split = layout.split(factor=0.5)
        split.label(text=item.Prefix if item.Prefix else "-")

# -------------------------------
# 抽出（保存）オペレーター
# -------------------------------
class DRONE_OT_SaveKeys(Operator):
    bl_idname = "drone.save_keys"
    bl_label = "Save Keys"

    def execute(self, context):
        """
        Workflow: prepare paths, export key data, export light effects,
        and report completion.
        """
        def convert_value(value):
            """Blender固有型やPropertyGroupをJSON化可能な値に変換"""
            if hasattr(value, "__annotations__"):
                return propertygroup_to_dict(value)

            if isinstance(value, bpy.types.bpy_prop_collection):
                return [convert_value(item) for item in value]

            if isinstance(value, bpy.types.ID):
                # Imageならパス優先、なければ名前
                if isinstance(value, bpy.types.Image):
                    return value.filepath_raw or value.name
                # 他のBlenderデータは名前のみ
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
                        "color": [round(c, 3) for c in elem.color]  # RGBAを0〜1で小数3桁
                    })
            return data

        def propertygroup_to_dict(pg):
            """PropertyGroupを辞書化（name含める）"""
            data = {}

            # まず name を追加（UIリストでの表示名）
            data["name"] = pg.name

            # 登録されているプロパティを辞書化
            for prop in pg.__annotations__.keys():
                val = getattr(pg, prop)
                data[prop] = convert_value(val)

            # type == COLOR_RAMP の場合、color_ramp情報を追加
            if getattr(pg, "type", None) == "COLOR_RAMP":
                texture = getattr(pg, "texture", None)
                data["color_ramp"] = convert_color_ramp(texture)

            return data

        def export_light_effects_to_json(filepath):
            scene = bpy.context.scene
            light_effects = scene.skybrush.light_effects.entries
            
            effects_data = [propertygroup_to_dict(effect) for effect in light_effects]

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(effects_data, f, ensure_ascii=False, indent=4)
                props = context.scene.drone_key_props
        
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        props = context.scene.drone_key_props
        # Ensure textures and materials have prefixed names for export
        add_prefix_le_tex(context)
        # Export keyframe data
        export_key(context, blend_dir, props.file_name)
        # Export light effect definitions
        export_light_effects_to_json(os.path.join(blend_dir, props.file_name + LightdataStr))
        self.report({'INFO'}, f"Keys saved: {blend_dir}")
        return {'FINISHED'}
    
class DRONE_OT_SaveSignleKeys(Operator):
    bl_idname = "drone.save_single_keys"
    bl_label = "Save Single Keys"

    def execute(self, context):
        """Save keyframes for the current selection using the configured file name.

        The keyframes are written to the directory of the active blend file and the
        resulting path is reported.
        """
        props = context.scene.drone_key_props
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        export_key(context, blend_dir, props.file_name)
        self.report({'INFO'}, f"Keys saved: {blend_dir}")
        return {'FINISHED'}

# -------------------------------
# 移植（ロード）オペレーター
# -------------------------------
class DRONE_OT_LoadKeys(Operator):
    bl_idname = "drone.load_keys"
    bl_label = "Load Keys"

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
  
class LIGHTEFFECT_OTadd_prefix_le_tex(bpy.types.Operator):
    bl_idname = "drone.add_prefix"
    bl_label = "Add Prefix"

    def execute(self, context):
        add_prefix_le_tex(context)
        return {'FINISHED'}
    
class DRONE_OT_shift_collecion(bpy.types.Operator):
    bl_idname = "drone.shift_coll_frame"
    bl_label = "ShiftCollectionFrame"

    def execute(self, context):
        """Create an "_Animated" collection, link selected objects, shift keys, and finish."""
        # 新しいコレクションを作成
        shift_collection_name = context.scene.drone_key_props.file_name + "_Animated"
        shift_collection = bpy.data.collections.new(shift_collection_name)
        bpy.context.scene.collection.children.link(shift_collection)
        
        # Link selected objects to the new collection (do not move them)
        selected_objects = bpy.context.selected_objects
        for obj in selected_objects:
            # Link instead of move
            if obj.name not in shift_collection.objects:
                shift_collection.objects.link(obj)
        shift_collection_key(shift_collection)
        return {'FINISHED'}

class TIMEBIND_OT_entry_add(bpy.types.Operator):
    bl_idname = "timebind.entry_add"
    bl_label = "Add TimeBind Entry"

    def execute(self, context):
        add_timebind_prop(context, "ANYPREFIX", 0)
        return {'FINISHED'}
    
class TIMEBIND_OT_goto_startframe(bpy.types.Operator):
    bl_idname = "timebind.goto_startframe"
    bl_label = "Goto StartFrame"

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

    def execute(self, context):
        tb = context.scene.time_bind
        if tb.entries and tb.active_index >= 0:
            tb.entries.remove(tb.active_index)
            tb.active_index = min(tb.active_index, len(tb.entries) - 1)
        return {'FINISHED'}
    
class TIMEBIND_OT_deselect(bpy.types.Operator):
    bl_idname = "timebind.deselect"
    bl_label = "Deselect"

    def execute(self, context):
        context.scene.time_bind.active_index = -1
        return {'FINISHED'}

class TIMEBIND_OT_entry_move(bpy.types.Operator):
    """Move entry up or down"""
    bl_idname = "timebind.entry_move"
    bl_label = "Move Entry"
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
                        shift_collection_key(bpy.data.collections.get(bind_entry.Prefix + "_Animated"))
                    # Drones内のオブジェクトキーを移動
                    if drones_collection:
                        self.move_material_keys(drones_collection.objects,
                                                bind_entry.StartFrame,
                                                sb_entry.duration,  # Duration使う
                                                diff)
                    # TimeBindをStoryboardに同期
                    bind_entry.StartFrame = sb_entry.frame_start
        return {'FINISHED'}

    def move_material_keys(self, objects, start_frame, duration, diff):
        end_frame = start_frame + duration

        for obj in objects:
            # オブジェクトにマテリアルがなければスキップ
            if not obj.material_slots:
                continue

            for slot in obj.material_slots:
                mat = slot.material
                if not mat or not mat.node_tree:
                    continue

                # マテリアル内のノード探索（RGBA入力を持つノード）
                for node in mat.node_tree.nodes:
                    if not node.inputs or node.inputs[0].type != 'RGBA':
                        continue

                    # アクション(F-Curve)があるかチェック
                    anim = mat.node_tree.animation_data
                    if not anim or not anim.action:
                        continue

                    for fcurve in anim.action.fcurves:
                        # default_value だけ対象（R,G,B,Aそれぞれindex別）
                        if "default_value" not in fcurve.data_path:
                            continue

                        for keyframe in fcurve.keyframe_points:
                            frame = keyframe.co.x
                            if start_frame <= frame <= end_frame:
                                keyframe.co.x += diff
                                keyframe.handle_left.x += diff
                                keyframe.handle_right.x += diff

def shift_collection_key(shift_collection):
    # === 設定 ===
    shift_amount = bpy.context.scene.frame_current  # 現在フレーム分シフト

        # --- 汎用キーシフト関数 ---
    def shift_keyframes(anim_data, amount):
        if anim_data and anim_data.action:
            for fcurve in anim_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.co.x += amount
                    keyframe.handle_left.x += amount
                    keyframe.handle_right.x += amount

    # --- コレクション内すべてのオブジェクト処理 ---
    for obj in shift_collection.all_objects:
        # オブジェクト本体
        shift_keyframes(obj.animation_data, shift_amount)

        # シェイプキー
        if obj.data and hasattr(obj.data, "shape_keys") and obj.data.shape_keys:
            shift_keyframes(obj.data.shape_keys.animation_data, shift_amount)

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

def update_texture_key(prefix, diff):
    # 全テクスチャをチェック
    for tex in bpy.data.textures:
        # 名前が prefix で始まる場合のみ処理
        if tex.name.startswith(prefix):
            if tex.animation_data and tex.animation_data.action:
                action = tex.animation_data.action
                for fcurve in action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        # フレーム位置を +CurrentFrame
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
    if duration is not 0:
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
    tb_entries = context.scene.time_bind.entries
    index = context.scene.time_bind.active_index
    if 0 <= index < len(tb_entries) and tb_entries:
        pref = tb_entries[index].Prefix
    for sb_entry in storyboard.entries:
        if pref == sb_entry.name.split("_")[0]:
            frame_start = sb_entry.frame_start
            duration = sb_entry.duration
            extract()
            return
    extract()



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
        layout = self.layout
        tb = context.scene.time_bind

        layout.prop(context.scene.drone_key_props, "file_name")
        layout.operator("drone.save_keys", text="Save")
        layout.operator("drone.save_single_keys", text="Key Save")
        layout.operator("drone.load_keys", text="Load")
        layout.operator("drone.load_all_keys", text="All Load")
        layout.operator("drone.add_prefix", text="Add Prefix")
        layout.operator("drone.shift_coll_frame", text="Shift Collection")
        layout.operator("timebind.goto_startframe", text="Goto Start")

                # Refreshボタン
        layout.operator("timebind.refresh", text="Refresh", icon='FILE_REFRESH')

        # UIList
        row = layout.row()
        row.template_list(
            "TIMEBIND_UL_entries", "", 
            tb, "entries", 
            tb, "active_index"
        )

        # 右側の操作ボタン
        col = row.column(align=True)
        col.operator("timebind.entry_add", icon='ADD', text="")
        col.operator("timebind.entry_remove", icon='REMOVE', text="")
        col.separator()
        col.operator("timebind.deselect", icon='RESTRICT_SELECT_ON', text="")
        col.separator()
        col.operator("timebind.entry_move", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("timebind.entry_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

        # 詳細編集
        if tb.entries and tb.active_index >= 0:
            entry = tb.entries[tb.active_index]
            box = layout.box()
            box.prop(entry, "Prefix")
            box.prop(entry, "StartFrame")

# -------------------------------
# 登録
# -------------------------------
classes = (
    DroneKeyTransferProperties,
    DRONE_OT_shift_collecion,
    DRONE_OT_SaveKeys,
    DRONE_OT_SaveSignleKeys,
    DRONE_OT_LoadKeys,
    DRONE_PT_KeyTransfer,
    DRONE_OT_LoadAllKeys,
    LIGHTEFFECT_OTadd_prefix_le_tex,
    TIMEBIND_OT_goto_startframe,
    TimeBindEntry,
    TimeBindCollection,
    TIMEBIND_UL_entries,
    TIMEBIND_OT_entry_add,
    TIMEBIND_OT_entry_remove,
    TIMEBIND_OT_entry_move,
    TIMEBIND_OT_deselect,
    TIMEBIND_OT_refresh,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.drone_key_props = bpy.props.PointerProperty(type=DroneKeyTransferProperties)
    bpy.types.Scene.time_bind = bpy.props.PointerProperty(type=TimeBindCollection)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.drone_key_props
    del bpy.types.Scene.time_bind

if __name__ == "__main__":
    register()
