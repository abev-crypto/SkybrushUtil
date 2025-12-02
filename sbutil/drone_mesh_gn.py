import bpy
from math import inf

# ======================================
# 設定
# ======================================

GN_GROUP_NAME = "GN_DroneInstances"
SPHERE_OBJECT_NAME = "DroneSphere"
MATERIAL_NAME = "M_DroneTexture"
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


def ensure_system_object(name: str, vertex_count: int) -> bpy.types.Object:
    """GNをぶら下げるためのシステム用オブジェクトを用意"""

    obj = bpy.data.objects.get(name)
    mesh = None

    if obj and obj.type == 'MESH':
        mesh = obj.data
    elif obj:
        bpy.data.objects.remove(obj, do_unlink=True)

    old_mesh = mesh if mesh and len(mesh.vertices) != vertex_count else None

    if mesh is None or len(mesh.vertices) != vertex_count:
        mesh = bpy.data.meshes.new(name + "_Mesh")
        verts = [(0.0, 0.0, 0.0)] * vertex_count
        mesh.from_pydata(verts, [], [])
        mesh.update()

    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        obj.data = mesh

    if old_mesh and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

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

    input_geom_socket = iface.new_socket(
        name="Base Geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
    )

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
    ico_sphere_node.inputs['Radius'].default_value = 0.5
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

    # Base geometry をコントローラーの座標に合わせる
    pos_node = nodes.new("GeometryNodeInputPosition")
    pos_node.location = (-100, 350)

    sample_pos = nodes.new("GeometryNodeSampleIndex")
    sample_pos.location = (150, 300)
    sample_pos.data_type = 'FLOAT_VECTOR'
    try:
        sample_pos.domain = 'POINT'
    except AttributeError:
        pass

    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (400, 250)

    # --- Index ---
    n_index = nodes.new("GeometryNodeInputIndex")
    n_index.location = (-250, -200)

    # Instance on Points
    n_inst = nodes.new("GeometryNodeInstanceOnPoints")
    n_inst.location = (850, 200)

    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.location = (400, 200)
    set_mat.inputs["Material"].default_value = mat

    # --- 接続（全部 index ベース） ---

    # 入力ジオメトリを Set Position に渡す
    links.new(n_in.outputs[input_geom_socket.name], set_pos.inputs["Geometry"])

    # Collection Info 出力0(Geometry) -> Realize Instances 入力0(Geometry)
    links.new(n_col.outputs[0], n_realize.inputs[0])

    # Realize 出力0(Geometry) -> Mesh to Points 入力0(Mesh)
    links.new(n_realize.outputs[0], n_mesh2pts.inputs[0])

    # Mesh to Points を元に位置をサンプリング
    links.new(n_mesh2pts.outputs[0], sample_pos.inputs["Geometry"])
    links.new(pos_node.outputs["Position"], sample_pos.inputs["Value"])
    links.new(n_index.outputs[0], sample_pos.inputs["Index"])
    links.new(sample_pos.outputs["Value"], set_pos.inputs["Position"])

    # インスタンス配置
    links.new(set_pos.outputs[0], n_inst.inputs[0])
    # Object Info 出力0(Geometry/Instance) -> Instance on Points 入力2(Instance)
    links.new(switch.outputs["Output"], n_inst.inputs[2])

    # Group Input の Check Circle をスイッチへ
    links.new(n_in.outputs[c_cir.name], switch.inputs["Switch"])

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
    """LEDColor頂点カラーを参照するEmissionマテリアルを作る"""

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

    # Vertex color (LEDColor)
    n_vcol = nodes.new("ShaderNodeVertexColor")
    n_vcol.location = (300, 0)
    n_vcol.layer_name = "LEDColor"

    # 接続
    links.new(n_vcol.outputs["Color"], n_em.inputs["Color"])
    links.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

    return mat


# ======================================
# エクスポートされたユーティリティ
# ======================================

def setup_for_collection(controller_collection: bpy.types.Collection) -> bpy.types.Object:
    """Set up the geometry node system for the given controller collection."""

    scene = bpy.context.scene
    render_range = get_render_range(scene)
    vertex_count = len(controller_collection.objects)

    mat = create_drone_material(MATERIAL_NAME, render_range)
    gn_group = create_gn_group(GN_GROUP_NAME, controller_collection, render_range, mat)

    system_obj = ensure_system_object("DroneSystem", vertex_count)
    mod = system_obj.modifiers.get("DroneInstances")
    if mod is None:
        mod = system_obj.modifiers.new(name="DroneInstances", type='NODES')
    mod.node_group = gn_group

    return system_obj


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
