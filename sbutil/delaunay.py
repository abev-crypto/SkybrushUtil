import bpy
import bmesh
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

# 現在フレームの評価済みワールド座標（アニメ/制約/親子反映）
def get_selected_evaluated_world_positions(include_hidden=False):
    scene = bpy.context.scene
    scene.frame_set(scene.frame_current)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objs = bpy.context.selected_objects
    if not include_hidden:
        objs = [o for o in objs if not o.hide_get()]
    return [o.evaluated_get(depsgraph).matrix_world.translation.copy() for o in objs]

def build_planar_mesh_from_points(points):
    if len(points) < 3:
        raise ValueError("少なくとも3つの点が必要です。")

    y_base = min(p.y for p in points)

    # 2D化（x,z）＋重複除去
    verts2d = [(p.x, p.z) for p in points]
    verts2d = list(dict.fromkeys(verts2d))  # 重複排除
    if len(verts2d) < 3:
        raise ValueError("有効頂点が3点未満（重複や同一点が多すぎます）。")

    # 2D Delaunay（三角形分割）
    # 戻り値は環境によって 3要素以上になることがあるので先頭3つだけ受け取る
    res = delaunay_2d_cdt(verts2d, [], [], 0, 1e-9)  # output_type=0
    v2, e2, f2 = res[:3]

    if not f2:
        raise ValueError("三角形が生成されませんでした（点がほぼ一直線の可能性）。")

    # 3D化（y=一定）
    verts3d = [(x, y_base, z) for (x, z) in v2]
    faces   = [tuple(face) for face in f2]

    mesh = bpy.data.meshes.new("PlaneFromPoints")
    mesh.from_pydata(verts3d, [], faces)
    mesh.update(calc_edges=True, calc_edges_loose=True)

    # 法線を +Y に概ね揃える
    mesh.flip_normals()
    if sum(p.normal.y for p in mesh.polygons) < 0:
        mesh.flip_normals()

    return mesh

def create_hull_plane_from_current_positions_y_thickness(thickness_default=0.2, centered_default=True):
    # 1) 現在フレームの評価済み座標
    pts = get_selected_evaluated_world_positions()
    if len(pts) < 3:
        raise ValueError("少なくとも3つのオブジェクトを選択してください。")

    # 2) 2D Delaunay で Plane メッシュ生成（ラミナ回避）
    mesh = build_planar_mesh_from_points(pts)

    # 3) オブジェクト作成
    obj = bpy.data.objects.new("PlaneTriangulatedObj", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # === 4) Geometry Nodes（Extrude + Centered） ===
    mod = obj.modifiers.new("YThickness", 'NODES')
    ng = bpy.data.node_groups.new("YThickness_FromTriangulated", 'GeometryNodeTree')
    mod.node_group = ng

    # 4.3+ のインターフェイスで I/O を定義
    ng.interface.new_socket(
        name="Geometry", in_out='INPUT',
        socket_type='NodeSocketGeometry', description="Input Geometry"
    )
    ng.interface.new_socket(
        name="Thickness", in_out='INPUT',
        socket_type='NodeSocketFloat', description="Extrude thickness (+Y)"
    )
    ng.interface.new_socket(
        name="Centered", in_out='INPUT',
        socket_type='NodeSocketBool', description="Center the thickness (±Thickness/2)"
    )
    ng.interface.new_socket(
        name="Geometry", in_out='OUTPUT',
        socket_type='NodeSocketGeometry', description="Output Geometry"
    )

    nodes = ng.nodes; links = ng.links
    nodes.clear()

    n_in = nodes.new("NodeGroupInput")
    n_out = nodes.new("NodeGroupOutput")

    n_extrude = nodes.new("GeometryNodeExtrudeMesh")     # Faces を押し出し
    n_setpos_top = nodes.new("GeometryNodeSetPosition")  # 押し出しで出来た Top だけ +Y
    n_vec_top = nodes.new("ShaderNodeCombineXYZ")        # (0, Thickness, 0)

    # Centered: 全体を -Thickness*0.5*Centered だけ Y に戻す
    n_setpos_center = nodes.new("GeometryNodeSetPosition")  # 全頂点
    n_half = nodes.new("ShaderNodeMath"); n_half.operation = 'MULTIPLY'     # Thickness * 0.5
    n_center_mul = nodes.new("ShaderNodeMath"); n_center_mul.operation = 'MULTIPLY'  # * Centered
    n_neg = nodes.new("ShaderNodeMath"); n_neg.operation = 'MULTIPLY'       # * (-1)
    n_vec_center = nodes.new("ShaderNodeCombineXYZ")      # (0, -Thickness*0.5*Centered, 0)

    # 位置
    n_in.location = (-800, 0)
    n_extrude.location = (-560, 0)
    n_setpos_top.location = (-300, 0)
    n_vec_top.location = (-320, -180)
    n_half.location = (-80, -180)
    n_center_mul.location = (120, -180)
    n_neg.location = (320, -180)
    n_vec_center.location = (520, -180)
    n_setpos_center.location = (60, 0)
    n_out.location = (260, 0)

    # 設定
    n_extrude.inputs["Offset Scale"].default_value = 0.0       # 位置移動は後段で
    n_extrude.inputs["Individual"].default_value = False

    # 入力→押し出し→Top移動
    links.new(n_in.outputs[0], n_extrude.inputs["Mesh"])
    links.new(n_extrude.outputs["Mesh"], n_setpos_top.inputs["Geometry"])
    links.new(n_extrude.outputs["Top"], n_setpos_top.inputs["Selection"])

    # Top を (0, Thickness, 0) へ
    links.new(n_in.outputs[1], n_vec_top.inputs["Y"])                  # Thickness
    links.new(n_vec_top.outputs["Vector"], n_setpos_top.inputs["Offset"])

    # Centered: -Thickness * 0.5 * Centered
    n_half.inputs[1].default_value = 0.5
    n_neg.inputs[1].default_value = -1.0
    links.new(n_in.outputs[1], n_half.inputs[0])                       # Thickness * 0.5
    links.new(n_half.outputs["Value"], n_center_mul.inputs[0])
    links.new(n_in.outputs[2], n_center_mul.inputs[1])                 # * Centered (0/1)
    links.new(n_center_mul.outputs["Value"], n_neg.inputs[0])          # * (-1)
    links.new(n_neg.outputs["Value"], n_vec_center.inputs["Y"])        # (0, neg, 0)

    # 全体オフセット適用
    links.new(n_setpos_top.outputs["Geometry"], n_setpos_center.inputs["Geometry"])
    links.new(n_vec_center.outputs["Vector"], n_setpos_center.inputs["Offset"])

    # 出力
    links.new(n_setpos_center.outputs["Geometry"], n_out.inputs[0])

    # モディファイア側の初期値（UI）
    mod["Input_2"] = float(thickness_default)  # Thickness
    mod["Input_3"] = bool(centered_default)    # Centered

    print("✅ 2D Delaunay で平面生成 → +Y 厚み（Centered対応）を作成しました。")
    return obj

# 実行例
create_hull_plane_from_current_positions_y_thickness(thickness_default=0.2, centered_default=True)
