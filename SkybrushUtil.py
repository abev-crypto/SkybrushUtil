bl_info = {
    "name": "SkyBrushUtil",
    "author": "ABEYUYA",
    "version": (1, 1),
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

# -------------------------------
# プロパティ
# -------------------------------
class DroneKeyTransferProperties(PropertyGroup):
    file_name: StringProperty(
        name="File Name",
        default="color_key_data",
        description="Save/Load JSON file name"
    )

# TimeBind用のPropertyGroup
class TimeBindEntry(bpy.types.PropertyGroup):
    StoryBordName: bpy.props.StringProperty(name="Storyboard Name")
    Prefix: bpy.props.StringProperty(name="Prefix")
    StartFrame: bpy.props.IntProperty(name="Start Frame")

# コレクション全体の管理用
class TimeBindCollection(bpy.types.PropertyGroup):
    entries: bpy.props.CollectionProperty(type=TimeBindEntry)
    active_index: bpy.props.IntProperty()

class TIMEBIND_UL_entries(bpy.types.UIList):
    """TimeBind entriesを表示するUIList"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item: TimeBindEntry
        split = layout.split(factor=0.5)
        split.label(text=item.StoryBordName if item.StoryBordName else "-")
        split.label(text=item.Prefix if item.Prefix else "-")

# -------------------------------
# 抽出（保存）オペレーター
# -------------------------------
class DRONE_OT_SaveKeys(Operator):
    bl_idname = "drone.save_keys"
    bl_label = "Save Keys"

    def execute(self, context):
        props = context.scene.drone_key_props
        file_name = props.file_name + ".json"

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
        prefix_light_effect_names(props.le_prefix)
        for tex in bpy.data.textures:
            # すでにプレフィックスが付いていない場合のみ追加
            if not tex.name.startswith(props.le_prefix):
                tex.name = props.le_prefix + tex.name

        # 保存先（Blendファイルと同じ場所）
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        save_path = os.path.join(blend_dir, file_name)

        drones_collection = bpy.data.collections.get("Drones")
        data = []

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
                    keys[fc.array_index].append((kp.co.x, kp.co.y))

            data.append({
                "name": obj.name,
                "location": list(obj.location),
                "keys": keys
            })

        with open(save_path, "w") as f:
            json.dump(data, f)
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
        file_name = props.file_name + "_LE.json"
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        load_path = os.path.join(blend_dir, file_name)
        export_light_effects_to_json(load_path)
        self.report({'INFO'}, f"Keys saved: {save_path}")
        return {'FINISHED'}


# -------------------------------
# 移植（ロード）オペレーター
# -------------------------------
class DRONE_OT_LoadKeys(Operator):
    bl_idname = "drone.load_keys"
    bl_label = "Load Keys"

    def execute(self, context):
        props = context.scene.drone_key_props
        file_name = props.file_name + ".json"

        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        load_path = os.path.join(blend_dir, file_name)

        with open(load_path, "r") as f:
            color_key_data = json.load(f)

        # JSONキーをintに変換
        for d in color_key_data:
            d["keys"] = {int(k): v for k, v in d["keys"].items()}

        drones_collection = bpy.data.collections.get("Drones")
        current_frame = bpy.context.scene.frame_current
        available_objects = list(drones_collection.objects)

        def find_nearest_object(location):
            nearest_obj = None
            min_dist = float('inf')
            for obj in available_objects:
                dist = (Vector(location) - obj.location).length
                if dist < min_dist:
                    min_dist = dist
                    nearest_obj = obj
            return nearest_obj

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
                            frame=current_frame + frame,
                            index=channel
                        )
                break

            available_objects.remove(nearest_obj)
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

        def import_light_effects_from_json(filepath, frame_offset=0):
            """
            JSONからLightEffectsを復元する
            frame_offset: frame_start / frame_end に加算するオフセット
            """
            scene = bpy.context.scene
            entries = scene.skybrush.light_effects.entries

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

                # プロパティをセット
                set_propertygroup_from_dict(effect, effect_data)

                # frame_offsetを適用
                if hasattr(effect, "frame_start"):
                    effect.frame_start += frame_offset
                if hasattr(effect, "frame_end"):
                    effect.frame_end += frame_offset

                # ColorRamp復元
                if effect.type == "COLOR_RAMP" and color_ramp_data:
                    apply_color_ramp(effect.texture, color_ramp_data)
        props = context.scene.drone_key_props
        file_name = props.file_name + "_LE.json"
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        load_path = os.path.join(blend_dir, file_name)
        import_light_effects_from_json(load_path)
        # 全テクスチャをチェック
        for tex in bpy.data.textures:
            # 名前が prefix で始まる場合のみ処理
            if tex.name.startswith(props.le_prefix):
                if tex.animation_data and tex.animation_data.action:
                    action = tex.animation_data.action
                    for fcurve in action.fcurves:
                        for keyframe in fcurve.keyframe_points:
                            # フレーム位置を +CurrentFrame
                            keyframe.co.x += current_frame
                            keyframe.handle_left.x += current_frame
                            keyframe.handle_right.x += current_frame

                    # 更新を通知
                    for fcurve in action.fcurves:
                        fcurve.update()
        self.report({'INFO'}, f"Keys loaded: {load_path}")
        return {'FINISHED'}
  
class TIMEBIND_OT_entry_add(bpy.types.Operator):
    bl_idname = "timebind.entry_add"
    bl_label = "Add TimeBind Entry"

    def execute(self, context):
        tb = context.scene.time_bind
        new_entry = tb.entries.add()
        new_entry.StoryBordName = "New"
        new_entry.Prefix = ""
        new_entry.StartFrame = 0
        tb.active_index = len(tb.entries) - 1
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


class TIMEBIND_OT_entry_move(bpy.types.Operator):
    """Move entry up or down"""
    bl_idname = "timebind.entry_move"
    bl_label = "Move Entry"
    direction: bpy.props.EnumProperty(items=(('UP', "Up", ""), ('DOWN', "Down", "")))

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
    bl_idname = "timebind.refresh"
    bl_label = "Refresh TimeBind"

    def execute(self, context):
        scene = context.scene
        storyboard = scene.skybrush.storyboard
        light_effects = scene.skybrush.light_effects

        for bind_entry in scene.time_bind.entries:
            # Storyboardエントリ検索
            for sb_entry in storyboard.entries:
                if sb_entry.name == bind_entry.StoryBordName:
                    # 差分計算
                    diff = sb_entry.StartFrame - bind_entry.StartFrame

                    # light_effectsのPrefix一致（startswith）
                    for le_entry in light_effects.entries:
                        if le_entry.name.startswith(bind_entry.Prefix):
                            le_entry.StartFrame += diff

                    # TimeBindをStoryboardに同期
                    bind_entry.StartFrame = sb_entry.StartFrame

        return {'FINISHED'}
    
# -------------------------------
# UIパネル
# -------------------------------
class DRONE_PT_KeyTransfer(Panel):
    bl_label = "Drone Key Transfer"
    bl_idname = "DRONE_PT_key_transfer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        layout = self.layout
        tb = context.scene.time_bind

        layout.prop(context.scene.drone_key_props, "file_name")
        layout.operator("drone.save_keys", text="Save")
        layout.operator("drone.load_keys", text="Load")

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
        col.operator("timebind.entry_move", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("timebind.entry_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

        # 詳細編集
        if tb.entries and tb.active_index >= 0:
            entry = tb.entries[tb.active_index]
            box = layout.box()
            box.prop(entry, "StoryBordName")
            box.prop(entry, "Prefix")
            box.prop(entry, "StartFrame")

# -------------------------------
# 登録
# -------------------------------
classes = (
    DroneKeyTransferProperties,
    DRONE_OT_SaveKeys,
    DRONE_OT_LoadKeys,
    DRONE_PT_KeyTransfer,
    TimeBindEntry,
    TimeBindCollection,
    TIMEBIND_UL_entries,
    TIMEBIND_OT_entry_add,
    TIMEBIND_OT_entry_remove,
    TIMEBIND_OT_entry_move,
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
