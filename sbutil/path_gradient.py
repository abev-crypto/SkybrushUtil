import bpy

def create_curve_segment_cylinder_gn():
    tree_name = "CurveSegmentCylinder_VCol"
    ng = bpy.data.node_groups.new(tree_name, 'GeometryNodeTree')

    #----------------------------------------------------------
    # インターフェース定義（Blender 4.3 仕様）
    #----------------------------------------------------------
    interface = ng.interface
    sockets_in = []
    sockets_out = []

    sockets_in.append(interface.new_socket(
        name="Geometry", in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description="入力カーブ"))

    sockets_in.append(interface.new_socket(
        name="Segment Start", in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="トリム開始位置"))
    sockets_in[-1].default_value = 0.3
    sockets_in[-1].min_value = 0.0
    sockets_in[-1].max_value = 1.0

    sockets_in.append(interface.new_socket(
        name="Segment End", in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="トリム終了位置"))
    sockets_in[-1].default_value = 0.7
    sockets_in[-1].min_value = 0.0
    sockets_in[-1].max_value = 1.0

    sockets_in.append(interface.new_socket(
        name="Radius", in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="円柱半径"))
    sockets_in[-1].default_value = 0.05
    sockets_in[-1].min_value = 0.0

    sockets_in.append(interface.new_socket(
        name="Sample Count", in_out='INPUT',
        socket_type='NodeSocketInt',
        description="リサンプリング数"))
    sockets_in[-1].default_value = 64
    sockets_in[-1].min_value = 2

    sockets_out.append(interface.new_socket(
        name="Geometry", in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description="出力ジオメトリ"))

    #----------------------------------------------------------
    # ノード群
    #----------------------------------------------------------
    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    group_in = nodes.new('NodeGroupInput'); group_in.location = (-800, 0)
    group_out = nodes.new('NodeGroupOutput'); group_out.location = (800, 0)

    trim = nodes.new('GeometryNodeTrimCurve')
    trim.location = (-600, 0)
    trim.mode = 'FACTOR'

    resample = nodes.new('GeometryNodeResampleCurve')
    resample.location = (-400, 0)
    resample.mode = 'COUNT'

    spline_param = nodes.new('GeometryNodeSplineParameter')
    spline_param.location = (-400, -180)

    comb_color = nodes.new('FunctionNodeCombineColor')
    comb_color.location = (-200, -180)

    store_attr = nodes.new('GeometryNodeStoreNamedAttribute')
    store_attr.location = (0, 0)
    store_attr.data_type = 'FLOAT_COLOR'
    store_attr.domain = 'POINT'
    store_attr.inputs["Name"].default_value = "color"

    circle = nodes.new('GeometryNodeCurvePrimitiveCircle')
    circle.location = (-200, 200)

    curve_to_mesh = nodes.new('GeometryNodeCurveToMesh')
    curve_to_mesh.location = (200, 0)
    # ★ 両端をフィル
    curve_to_mesh.inputs["Fill Caps"].default_value = True

    # ★ Output の前に Realize Instances
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (500, 0)

    #----------------------------------------------------------
    # 接続
    #----------------------------------------------------------
    l = links.new

    # 入力 → Trim
    l(group_in.outputs['Geometry'], trim.inputs['Curve'])
    l(group_in.outputs['Segment Start'], trim.inputs['Start'])
    l(group_in.outputs['Segment End'], trim.inputs['End'])

    # Trim → Resample
    l(trim.outputs['Curve'], resample.inputs['Curve'])
    l(group_in.outputs['Sample Count'], resample.inputs['Count'])

    # Resample → Store Named Attribute
    l(resample.outputs['Curve'], store_attr.inputs['Geometry'])

    # Spline Parameter Factor → RGB 同じ値で頂点カラー
    l(spline_param.outputs['Factor'], comb_color.inputs['Red'])
    l(spline_param.outputs['Factor'], comb_color.inputs['Green'])
    l(spline_param.outputs['Factor'], comb_color.inputs['Blue'])
    l(comb_color.outputs['Color'], store_attr.inputs['Value'])

    # Radius → Circle
    l(group_in.outputs['Radius'], circle.inputs['Radius'])

    # StoreAttr → Curve to Mesh
    l(store_attr.outputs['Geometry'], curve_to_mesh.inputs['Curve'])

    # Circle → Curve to Mesh Profile
    l(circle.outputs['Curve'], curve_to_mesh.inputs['Profile Curve'])

    # Curve to Mesh → Realize Instances → Output
    l(curve_to_mesh.outputs['Mesh'], realize.inputs['Geometry'])
    l(realize.outputs['Geometry'], group_out.inputs['Geometry'])

    circle.inputs["Resolution"].default_value = 16

    return ng


def add_modifier_to_active_object(ng):
    obj = bpy.context.active_object
    if obj is None:
        print("アクティブオブジェクトがありません。")
        return
    mod = obj.modifiers.new(name=ng.name, type='NODES')
    mod.node_group = ng
    print(f"モディファイア '{ng.name}' を '{obj.name}' に追加しました。")


# 実行
# sort_vtx_and_create_curve()
# ng = create_curve_segment_cylinder_gn()
# add_modifier_to_active_object(ng)


import bmesh

def sort_vtx_and_create_curve():
    # -------------------------------------------------------
    # 前提チェック（編集モードのメッシュで実行）
    # -------------------------------------------------------
    obj = bpy.context.edit_object
    if obj is None or obj.type != 'MESH':
        raise Exception("メッシュオブジェクトの編集モードで実行してください。")

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # -------------------------------------------------------
    # 1. 選択頂点を「選択順」で取得
    # -------------------------------------------------------
    hist_verts = [ele for ele in bm.select_history if isinstance(ele, bmesh.types.BMVert)]

    if hist_verts:
        ordered_verts = hist_verts
    else:
        # 選択順が取れない場合のフォールバック
        ordered_verts = [v for v in bm.verts if v.select]
        ordered_verts.sort(key=lambda v: v.index)

    if len(ordered_verts) < 2:
        raise Exception("2つ以上の頂点を選択してください。")

    # ローカル座標（新メッシュ用）
    coords_local = [v.co.copy() for v in ordered_verts]
    # ワールド座標（カーブ用）
    coords_world = [obj.matrix_world @ v.co for v in ordered_verts]

    print("==== 選択順 → 元の頂点 index ====")
    for new_idx, v in enumerate(ordered_verts):
        print(f"新{new_idx}  <-  元{v.index}")

    # -------------------------------------------------------
    # 2. 選択順に頂点IDを並べ替えた新メッシュを作成
    #    → 頂点 index は 0,1,2,... がそのまま「選択順」になる
    # -------------------------------------------------------
    sorted_mesh = bpy.data.meshes.new(me.name + "_sorted")
    sorted_obj = bpy.data.objects.new(obj.name + "_SortedVerts", sorted_mesh)
    bpy.context.collection.objects.link(sorted_obj)

    # 頂点のみのメッシュ + 頂点間にエッジも張っておく
    edges = [(i, i + 1) for i in range(len(coords_local) - 1)]
    sorted_mesh.from_pydata(coords_local, edges, [])

    # 元オブジェクトと同じトランスフォームを持たせる
    sorted_obj.matrix_world = obj.matrix_world

    # -------------------------------------------------------
    # 3. 頂点順に沿ったパス（カーブ）を作成
    # -------------------------------------------------------
    curve_data = bpy.data.curves.new(name=obj.name + "_PathFromVerts", type='CURVE')
    curve_data.dimensions = '3D'

    spline = curve_data.splines.new(type='POLY')
    spline.points.add(len(coords_world) - 1)

    for i, co in enumerate(coords_world):
        spline.points[i].co = (co.x, co.y, co.z, 1.0)

    spline.use_cyclic_u = False  # ループさせたいなら True

    curve_obj = bpy.data.objects.new(obj.name + "_Path", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    print(f"{len(coords_local)} 個の頂点から SortedVerts / Path を作成しました。")
