import bpy

NODE_GROUP_NAME = "CamRing_GN"
TARGET_COLLECTION_NAME = "Drones"
SUPPORTED_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'POINTCLOUD'}


def ensure_node_group():
    """画像のジオメトリノードを 4.3 方式で新規作成して返す"""

    # 以前の失敗で同名の NodeGroup が残っている可能性があるので一度消す
    if NODE_GROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[NODE_GROUP_NAME], do_unlink=True)

    # 新規 Geometry Node Tree
    ng = bpy.data.node_groups.new(NODE_GROUP_NAME, 'GeometryNodeTree')

    # ───────────────── インターフェイス（Geometry 入出力） ─────────────────
    iface = ng.interface
    # 入力 Geometry
    iface.new_socket(
        name="Geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description=""
    )
    # 出力 Geometry
    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description=""
    )

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    interface = ng.interface
    c_cir = interface.new_socket(name="Check Circle", in_out='INPUT', socket_type='NodeSocketBool')
    c_cir.default_value = True  # デフォルトは従来通り 0〜1 グラデ

    # ───────────────── Group In / Out ─────────────────
    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-600, -100)

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (600, 100)

    # ───────────────── Curve 部分 ─────────────────
    # 小さい Curve Circle（画像左のやつ / 出力には未接続だが再現しておく）
    cc_small = nodes.new("GeometryNodeCurvePrimitiveCircle")
    cc_small.location = (-1000, 150)
    cc_small.mode = 'RADIUS'
    cc_small.inputs["Resolution"].default_value = 3
    cc_small.inputs["Radius"].default_value = 0.005

    # 大きい Curve Circle（こちらをメッシュ化に使用）
    cc_big = nodes.new("GeometryNodeCurvePrimitiveCircle")
    cc_big.location = (-780, 150)
    cc_big.mode = 'RADIUS'
    cc_big.inputs["Resolution"].default_value = 32
    cc_big.inputs["Radius"].default_value = 0.78

    # Curve → Mesh
    curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
    curve_to_mesh.location = (-520, 150)
    curve_to_mesh.inputs["Fill Caps"].default_value = False

    # Transform Geometry
    trans_geo = nodes.new("GeometryNodeTransform")
    trans_geo.location = (-250, 150)

    # ───────────────── Camera / Object Info / Align Rotation ─────────────────
    active_cam = nodes.new("GeometryNodeInputActiveCamera")
    active_cam.location = (-1050, -150)

    obj_info = nodes.new("GeometryNodeObjectInfo")
    obj_info.location = (-820, -150)
    obj_info.transform_space = 'RELATIVE'  # ノードの「Relative」

    align_rot = nodes.new("FunctionNodeAlignRotationToVector")
    align_rot.location = (-520, -150)
    align_rot.axis = 'Y'
    align_rot.pivot_axis = 'Y'
    align_rot.inputs["Factor"].default_value = 1.0
    align_rot.inputs["Vector"].default_value = (1.0, 0.0, 0.0)

    # ───────────────── Scale Elements / Join / Set Material Index ─────────────────
    scale_elem = nodes.new("GeometryNodeScaleElements")
    scale_elem.location = (-250, -120)
    scale_elem.domain = 'FACE'
    scale_elem.inputs["Scale"].default_value = 0.4

    join_geo = nodes.new("GeometryNodeJoinGeometry")
    join_geo.location = (100, 100)

    switch = nodes.new("GeometryNodeSwitch")
    switch.location = (100, 200)

    set_mat_index = nodes.new("GeometryNodeSetMaterialIndex")
    set_mat_index.location = (350, 100)
    set_mat_index.inputs["Material Index"].default_value = 1

    # ───────────────── リンク接続 ─────────────────

    # Active Camera → Object Info
    links.new(active_cam.outputs["Active Camera"], obj_info.inputs["Object"])

    # Object Info (Rotation) → Align Rotation
    links.new(obj_info.outputs["Rotation"], align_rot.inputs["Rotation"])

    # Align Rotation → Transform Geometry (Rotation)
    links.new(align_rot.outputs["Rotation"], trans_geo.inputs["Rotation"])

    # Curve Circle (大) → Curve to Mesh
    links.new(cc_big.outputs["Curve"], curve_to_mesh.inputs["Curve"])
    links.new(cc_small.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])

    # Curve to Mesh → Transform Geometry
    links.new(curve_to_mesh.outputs["Mesh"], trans_geo.inputs["Geometry"])

    # Group Input Geometry → Scale Elements → Join
    links.new(n_in.outputs["Geometry"], scale_elem.inputs["Geometry"])
    links.new(scale_elem.outputs["Geometry"], join_geo.inputs["Geometry"])

    # Transform Geometry → Join
    links.new(trans_geo.outputs["Geometry"], join_geo.inputs["Geometry"])

    links.new(join_geo.outputs["Geometry"], switch.inputs["True"])
    links.new(scale_elem.outputs["Geometry"], switch.inputs["False"])
    links.new(n_in.outputs['Check Circle'], switch.inputs["Switch"])

    # Join → Set Material Index → Group Output
    links.new(switch.outputs["Output"], set_mat_index.inputs["Geometry"])
    links.new(set_mat_index.outputs["Geometry"], n_out.inputs["Geometry"])

    return ng


def _iter_collection_objects(collection):
    """コレクション配下（入れ子含む）のオブジェクトを重複なく列挙"""

    seen = set()
    stack = [collection]
    while stack:
        col = stack.pop()
        for obj in col.objects:
            if obj.name not in seen:
                seen.add(obj.name)
                yield obj
        stack.extend(col.children)


def get_target_objects():
    """Drones コレクション配下の対象オブジェクトを返す"""

    collection = bpy.data.collections.get(TARGET_COLLECTION_NAME)
    if collection is None:
        return []
    return [obj for obj in _iter_collection_objects(collection) if obj.type in SUPPORTED_TYPES]


def add_modifier_to_drones():
    """Drones コレクション内のオブジェクトに Geometry Nodes モディファイアを追加"""

    objects = get_target_objects()
    if not objects:
        return 0

    ng = ensure_node_group()
    count = 0
    for obj in objects:
        existing = next(
            (
                mod
                for mod in obj.modifiers
                if mod.type == 'NODES'
                and (mod.name == NODE_GROUP_NAME or getattr(mod.node_group, "name", "") == NODE_GROUP_NAME)
            ),
            None,
        )
        if existing is None:
            existing = obj.modifiers.new(name=NODE_GROUP_NAME, type='NODES')
        existing.node_group = ng
        count += 1
    return count


def remove_modifier_from_drones():
    """Drones コレクション内から対象ジオメトリノードモディファイアを削除"""

    objects = get_target_objects()
    removed = 0
    for obj in objects:
        modifiers = [
            mod
            for mod in obj.modifiers
            if mod.type == 'NODES'
            and (mod.name == NODE_GROUP_NAME or getattr(mod.node_group, "name", "") == NODE_GROUP_NAME)
        ]
        for mod in modifiers:
            obj.modifiers.remove(mod)
            removed += 1

    # 使われなくなったノードグループがあれば削除
    if NODE_GROUP_NAME in bpy.data.node_groups:
        node_group = bpy.data.node_groups[NODE_GROUP_NAME]
        if not node_group.users:
            bpy.data.node_groups.remove(node_group, do_unlink=True)

    return removed


class DRONE_OT_apply_drone_check_gn(bpy.types.Operator):
    bl_idname = "drone.apply_drone_check_gn"
    bl_label = "Apply Drone Check GN"
    bl_description = "Dronesコレクション内のオブジェクトへ CamRing_GN モディファイアを追加"

    def execute(self, context):
        try:
            count = add_modifier_to_drones()
        except Exception as exc:  # pragma: no cover - Blender runtime error
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if count == 0:
            self.report({'WARNING'}, "対象オブジェクトが見つかりませんでした")
        else:
            self.report({'INFO'}, f"{count} 件のモディファイアを設定しました")
        return {'FINISHED'}


class DRONE_OT_remove_drone_check_gn(bpy.types.Operator):
    bl_idname = "drone.remove_drone_check_gn"
    bl_label = "Remove Drone Check GN"
    bl_description = "Dronesコレクション内から CamRing_GN モディファイアを削除"

    def execute(self, context):
        try:
            removed = remove_modifier_from_drones()
        except Exception as exc:  # pragma: no cover - Blender runtime error
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if removed == 0:
            self.report({'WARNING'}, "削除対象のモディファイアが見つかりませんでした")
        else:
            self.report({'INFO'}, f"{removed} 件のモディファイアを削除しました")
        return {'FINISHED'}


classes = (
    DRONE_OT_apply_drone_check_gn,
    DRONE_OT_remove_drone_check_gn,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    add_modifier_to_drones()
