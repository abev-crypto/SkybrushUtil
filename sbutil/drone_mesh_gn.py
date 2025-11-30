import bpy
from math import inf

# ======================================
# 設定
# ======================================

GN_GROUP_NAME = "GN_DroneInstances"
SPHERE_OBJECT_NAME = "DroneSphere"
MATERIAL_NAME = "M_DroneTexture"
DRONE_UV_ATTR_NAME = "drone_uv"  # Geometry Nodes / Shader で使う属性名
CHECK_CIRCLE_SOCKET_NAME = "Check Circle"

# ======================================
# ユーティリティ
# ======================================

def get_render_range(scene: bpy.types.Scene) -> int:
    """frame_start ~ frame_end からレンダーレンジの長さを返す"""
    return scene.frame_end - scene.frame_start + 1


def ensure_sphere_object(name: str) -> bpy.types.Object:
    """インスタンス用のUV Sphereオブジェクトを用意する"""
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        return obj

    # 既にあってもメッシュじゃなければ消す
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 新しく作成
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=8,
        radius=0.5,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "_Mesh"
    return obj


def ensure_system_object(name: str) -> bpy.types.Object:
    """GNをぶら下げるためのシステム用オブジェクトを用意"""
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        return obj

    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ======================================
# Geometry Nodes セットアップ
# ======================================

def create_gn_group(
    group_name: str,
    controller_collection: bpy.types.Collection,
    render_range: int,
    mat: bpy.types.Material,
) -> bpy.types.NodeTree:
    """コントローラーコレクションからSphereインスタンスを生やすGNグループを作成"""

    # 既存あれば消す（再生成）
    old = bpy.data.node_groups.get(group_name)
    if old and old.bl_idname == "GeometryNodeTree":
        bpy.data.node_groups.remove(old, do_unlink=True)

    ng = bpy.data.node_groups.new(group_name, "GeometryNodeTree")

    # --------------------------------------------------
    # 4.x 方式: interface で出力ソケット定義
    # --------------------------------------------------
    iface = ng.interface

    # 念のため既存をクリア
    for item in list(iface.items_tree):
        iface.items_tree.remove(item)

    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
    )

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    # Group Output
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (900, 0)

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-600, -100)

    c_cir = iface.new_socket(
    name=CHECK_CIRCLE_SOCKET_NAME,
    in_out='INPUT',
    socket_type='NodeSocketBool',
    )
    c_cir.default_value = True  # デフォルトは従来通り 0〜1 グラデ

        # ───────────────── Curve 部分 ─────────────────
    # 小さい Curve Circle（画像左のやつ / 出力には未接続だが再現しておく）
    cc_small = nodes.new("GeometryNodeCurvePrimitiveCircle")
    cc_small.location = (-1000, 150)
    cc_small.mode = 'RADIUS'
    cc_small.inputs["Resolution"].default_value = 2
    cc_small.inputs["Radius"].default_value = 0.005

    # 大きい Curve Circle（こちらをメッシュ化に使用）
    cc_big = nodes.new("GeometryNodeCurvePrimitiveCircle")
    cc_big.location = (-780, 150)
    cc_big.mode = 'RADIUS'
    cc_big.inputs["Resolution"].default_value = 24
    cc_big.inputs["Radius"].default_value = 0.78

    ico_sphere_node = nodes.new(type='GeometryNodeMeshIcoSphere')
    ico_sphere_node.location = (0, 0)
    # パラメーターを任意に設定
    ico_sphere_node.inputs['Radius'].default_value = 1
    ico_sphere_node.inputs['Subdivisions'].default_value = 2

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
    links.new(ico_sphere_node.outputs["Mesh"], scale_elem.inputs["Geometry"])
    links.new(scale_elem.outputs["Geometry"], join_geo.inputs["Geometry"])

    # Transform Geometry → Join
    links.new(trans_geo.outputs["Geometry"], join_geo.inputs["Geometry"])

    links.new(join_geo.outputs["Geometry"], switch.inputs["True"])
    links.new(ico_sphere_node.outputs["Mesh"], switch.inputs["False"])
    links.new(n_in.outputs[CHECK_CIRCLE_SOCKET_NAME], switch.inputs["Switch"])


    # --- Collection Info ---
    n_col = nodes.new("GeometryNodeCollectionInfo")
    n_col.location = (-800, 200)
    # 入力ソケットを index で扱う（名前は使わない）
    # 0: Collection, 1: Separate Children, 2: Reset Children, それ以降オプション
    n_col.inputs[0].default_value = controller_collection
    if len(n_col.inputs) > 1:
        n_col.inputs[1].default_value = True   # Separate Children
    if len(n_col.inputs) > 2:
        n_col.inputs[2].default_value = False  # Reset Children

    # Blender 4.x だと Transform / As Instance 等が後ろにぶら下がってることがある
    for sock in n_col.inputs:
        if sock.name.lower().startswith("transform"):
            sock.default_value = True
        if "instance" in sock.name.lower():
            sock.default_value = True

    # --- Realize Instances ---
    n_realize = nodes.new("GeometryNodeRealizeInstances")
    n_realize.location = (-550, 200)

    # --- Mesh to Points（頂点をポイント化）---
    n_mesh2pts = nodes.new("GeometryNodeMeshToPoints")
    n_mesh2pts.location = (-300, 200)
    try:
        n_mesh2pts.mode = 'VERTICES'
    except AttributeError:
        pass

    # Radius ソケットだけを安全に探して 0.0 を入れる
    for sock in n_mesh2pts.inputs:
        ident = getattr(sock, "identifier", "")
        name = getattr(sock, "name", "")
        if ident == "Radius" or name in {"Radius", "半径"}:
            sock.default_value = 0.0
            break

    # --- Domain Size（ポイント数を取得）---
    n_domain = nodes.new("GeometryNodeAttributeDomainSize")
    n_domain.location = (-300, -100)
    # Mesh to Points の出力は POINTCLOUD コンポーネント
    try:
        n_domain.component = 'POINTCLOUD'
    except AttributeError:
        pass

    # --- Index ---
    n_index = nodes.new("GeometryNodeInputIndex")
    n_index.location = (-100, -200)

    # Y = (Index + 0.5) / PointCount
    n_add = nodes.new("ShaderNodeMath")
    n_add.location = (100, -200)
    n_add.operation = 'ADD'
    n_add.inputs[1].default_value = 0.5

    n_div = nodes.new("ShaderNodeMath")
    n_div.location = (300, -200)
    n_div.operation = 'DIVIDE'

    # X = 1 / RenderRange（一定値）
    n_valx = nodes.new("ShaderNodeValue")
    n_valx.location = (100, 50)
    n_valx.outputs[0].default_value = 1.0 / float(render_range)

    # Combine XYZ -> UVベクトル
    n_combine = nodes.new("ShaderNodeCombineXYZ")
    n_combine.location = (550, 50)

    # Store Named Attribute : drone_uv (ベクトル・ポイント)
    n_store = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store.location = (650, 200)
    n_store.data_type = 'FLOAT_VECTOR'
    n_store.domain = 'POINT'
    n_store.inputs[2].default_value = DRONE_UV_ATTR_NAME  # Name ソケット（たいてい index 2）

    # Instance on Points
    n_inst = nodes.new("GeometryNodeInstanceOnPoints")
    n_inst.location = (850, 200)

    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.location = (400, 200)
    set_mat.inputs["Material"].default_value = mat

    # --- 接続（全部 index ベース） ---

    # Collection Info 出力0(Geometry) -> Realize Instances 入力0(Geometry)
    links.new(n_col.outputs[0], n_realize.inputs[0])

    # Realize 出力0(Geometry) -> Mesh to Points 入力0(Mesh)
    links.new(n_realize.outputs[0], n_mesh2pts.inputs[0])

    # Mesh to Points 出力0(Points) -> Domain Size 入力0(Geometry)
    links.new(n_mesh2pts.outputs[0], n_domain.inputs[0])

    # Index 出力0 -> Add 入力0
    links.new(n_index.outputs[0], n_add.inputs[0])

    # Add 出力 -> Divide 入力0
    links.new(n_add.outputs[0], n_div.inputs[0])

    # Domain Size 出力0(Point Count) -> Divide 入力1
    links.new(n_domain.outputs[0], n_div.inputs[1])

    # X=1/RenderRange, Y=Divide の結果
    links.new(n_valx.outputs[0], n_combine.inputs[0])   # X
    links.new(n_div.outputs[0], n_combine.inputs[1])    # Y

    # UVベクトルを named attribute として保存
    # Mesh to Points 出力0(Points) -> StoreNamedAttribute 入力0(Geometry)
    links.new(n_mesh2pts.outputs[0], n_store.inputs[0])
    # Combine XYZ 出力0(Vector) -> StoreNamedAttribute 入力3(Value)
    if len(n_store.inputs) > 3:
        links.new(n_combine.outputs[0], n_store.inputs[3])
    else:
        # もしソケット構成が違っていたら最後の入力に繋ぐ
        links.new(n_combine.outputs[0], n_store.inputs[-1])

    # インスタンス配置
    # StoreNamedAttribute 出力0(Geometry) -> Instance on Points 入力0(Points)
    links.new(n_store.outputs[0], n_inst.inputs[0])
    # Object Info 出力0(Geometry/Instance) -> Instance on Points 入力2(Instance)
    links.new(switch.outputs["Output"], n_inst.inputs[2])

    # 出力へ（Group Output の最初の入力ソケットに接続）
    if not group_out.inputs:
        iface.new_socket(
            name="Geometry",
            in_out='OUTPUT',
            socket_type='NodeSocketGeometry',
        )
    output_input = group_out.inputs[0]
    links.new(n_inst.outputs[0], set_mat.inputs["Geometry"])

    links.new(set_mat.outputs["Geometry"], output_input)

    return ng



# ======================================
# マテリアルセットアップ
# ======================================

def create_drone_material(mat_name: str, render_range: int) -> bpy.types.Material:
    """drone_uv 属性からUVを取り、X方向をフレームで動かすEmissionマテリアルを作る"""

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # 出力
    n_out = nodes.new("ShaderNodeOutputMaterial")
    n_out.location = (800, 0)

    # Emission
    n_em = nodes.new("ShaderNodeEmission")
    n_em.location = (550, 0)
    n_em.inputs["Strength"].default_value = 5.0

    # Attribute（drone_uv）
    n_attr = nodes.new("ShaderNodeAttribute")
    n_attr.location = (-400, 0)
    n_attr.attribute_name = DRONE_UV_ATTR_NAME

    # Separate XYZ
    n_sep = nodes.new("ShaderNodeSeparateXYZ")
    n_sep.location = (-150, 0)

    # FrameNorm（Valueノードにドライバーで frame / RenderRange を入れる）
    n_frame = nodes.new("ShaderNodeValue")
    n_frame.location = (-150, -200)
    n_frame.label = "FrameNorm"

    # X + FrameNorm
    n_add = nodes.new("ShaderNodeMath")
    n_add.location = (100, -100)
    n_add.operation = 'ADD'

    # Combine XYZ (X動かしたもの + Yそのまま)
    n_combine = nodes.new("ShaderNodeCombineXYZ")
    n_combine.location = (350, 0)

    # Image Texture（ここにアニメーションテクスチャを設定してもらう）
    n_tex = nodes.new("ShaderNodeTexImage")
    n_tex.location = (550, -200)
    n_tex.label = "DroneTexture"
    n_tex.extension = 'REPEAT'  # X方向にループさせたいので

    # 接続
    links.new(n_attr.outputs["Vector"], n_sep.inputs["Vector"])

    # X + FrameNorm
    links.new(n_sep.outputs["X"], n_add.inputs[0])
    links.new(n_frame.outputs[0], n_add.inputs[1])

    # 新しいUVベクトル
    links.new(n_add.outputs["Value"], n_combine.inputs["X"])
    links.new(n_sep.outputs["Y"], n_combine.inputs["Y"])

    # UV -> テクスチャ
    links.new(n_combine.outputs["Vector"], n_tex.inputs["Vector"])

    # テクスチャカラー -> Emission Color
    links.new(n_tex.outputs["Color"], n_em.inputs["Color"])

    # Emission -> 出力
    links.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

    # FrameNorm にドライバーを設定： frame / RenderRange
    data_path = f'nodes["{n_frame.name}"].outputs[0].default_value'
    fcurve = nt.driver_add(data_path)
    drv = fcurve.driver
    # Blenderにはデフォルトで 'frame' という変数があるのでそれを使う
    drv.expression = f"frame/{float(render_range)}"

    # 変数定義は不要（frameはビルトイン）だが、明示したいなら以下を追加してもよい
    # var = drv.variables.new()
    # var.name = "frame"
    # var.targets[0].id_type = 'SCENE'
    # var.targets[0].id = bpy.context.scene
    # var.targets[0].data_path = "frame_current"

    return mat


# ======================================
# エクスポートされたユーティリティ
# ======================================

def setup_for_collection(controller_collection: bpy.types.Collection) -> bpy.types.Object:
    """Set up the geometry node system for the given controller collection."""

    scene = bpy.context.scene
    render_range = get_render_range(scene)

    mat = create_drone_material(MATERIAL_NAME, render_range)
    gn_group = create_gn_group(GN_GROUP_NAME, controller_collection, render_range, mat)

    system_obj = ensure_system_object("DroneSystem")
    mod = system_obj.modifiers.get("DroneInstances")
    if mod is None:
        mod = system_obj.modifiers.new(name="DroneInstances", type='NODES')
    mod.node_group = gn_group

    return system_obj


# ======================================
# メイン処理
# ======================================

def main():
    scene = bpy.context.scene
    render_range = get_render_range(scene)

    active = bpy.context.active_object
    if active is None or not active.users_collection:
        raise RuntimeError("トランスフォーム用コレクション内のオブジェクトを1つアクティブにしてから実行してください。")

    controller_collection = active.users_collection[0]

    print(f"Using controller collection: {controller_collection.name}")
    print(f"RenderRange (frame_start={scene.frame_start}, frame_end={scene.frame_end}) -> {render_range} frames")

    system_obj = setup_for_collection(controller_collection)

    print("セットアップ完了！")
    print("・DroneSystem オブジェクトを選択してビューポートを再生すると、")
    print("  コントローラーコレクション内の各頂点位置に Sphere がインスタンスされ、")
    print("  マテリアル側で frame/RenderRange に応じてUVのXが動きます。")
    print("・Image Texture ノード(DroneTexture)に好きなテクスチャを割り当ててください。")


if __name__ == "__main__":
    main()
