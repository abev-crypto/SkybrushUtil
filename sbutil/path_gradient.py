import bpy


def create_curve_segment_cylinder_gn():
    """Return the geometry node group that generates a colored curve mesh."""

    tree_name = "CurveSegmentCylinder_VCol"
    if tree_name in bpy.data.node_groups:
        return bpy.data.node_groups[tree_name]

    ng = bpy.data.node_groups.new(tree_name, 'GeometryNodeTree')

    # ----------------------------------------------------------
    # インターフェース定義（Blender 4.x：interface.new_socket）
    # ----------------------------------------------------------
    interface = ng.interface

    # INPUTS
    s_geom = interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')

    s_start = interface.new_socket(name="Segment Start", in_out='INPUT', socket_type='NodeSocketFloat')
    s_start.default_value = 0.0
    s_start.min_value = 0.0
    s_start.max_value = 1.0

    s_end = interface.new_socket(name="Segment End", in_out='INPUT', socket_type='NodeSocketFloat')
    s_end.default_value = 1.0
    s_end.min_value = 0.0
    s_end.max_value = 1.0

    s_radius = interface.new_socket(name="Radius", in_out='INPUT', socket_type='NodeSocketFloat')
    s_radius.default_value = 1.0
    s_radius.min_value = 0.0

    s_count = interface.new_socket(name="Sample Count", in_out='INPUT', socket_type='NodeSocketInt')
    s_count.default_value = 64
    s_count.min_value = 2

    s_offset = interface.new_socket(name="Color Offset", in_out='INPUT', socket_type='NodeSocketFloat')

    # ★ 追加：絶対長さで頂点カラーを書くかどうか
    s_abs = interface.new_socket(name="Use Absolute Length", in_out='INPUT', socket_type='NodeSocketBool')
    s_abs.default_value = False  # デフォルトは従来通り 0〜1 グラデ

    s_cap = interface.new_socket(name="Fill Caps", in_out='INPUT', socket_type='NodeSocketBool')
    s_cap.default_value = True  # ★追加：フタを付けるか

    # OUTPUTS
    interface.new_socket(name="Geometry",  in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    # ----------------------------------------------------------
    # ノード作成
    # ----------------------------------------------------------
    group_in  = nodes.new('NodeGroupInput');           group_in.location  = (-900,   0)
    group_out = nodes.new('NodeGroupOutput');          group_out.location = ( 600,   0)

    # 0.3〜0.7 区間のトリム
    trim = nodes.new('GeometryNodeTrimCurve')
    trim.location = (-700, 0)
    trim.mode = 'FACTOR'

    # 均等サンプル
    resample = nodes.new('GeometryNodeResampleCurve')
    resample.location = (-500, 0)
    resample.mode = 'COUNT'

    # カーブ長（トリム＋リサンプル後）
    curve_length = nodes.new('GeometryNodeCurveLength')
    curve_length.location = (-500, -200)

    # 0〜1 の位置パラメータ
    spline_param = nodes.new('GeometryNodeSplineParameter')
    spline_param.location = (-500, -400)

    # 絶対長さ = Factor * Length 用 Math
    math_mul = nodes.new('ShaderNodeMath')
    math_mul.location = (-300, -300)
    math_mul.operation = 'MULTIPLY'

    math_add = nodes.new('ShaderNodeMath')
    math_add.location = (-300, -400)
    math_add.operation = 'ADD'

    # 正規化 or 絶対長さ の切り替え
    switch = nodes.new('GeometryNodeSwitch')
    switch.location = (-100, -300)
    switch.input_type = 'FLOAT'

    # グレースケール → Color
    comb_color = nodes.new('FunctionNodeCombineColor')
    comb_color.location = (100, -300)

    # 頂点カラー "color" を書き込む
    store_attr = nodes.new('GeometryNodeStoreNamedAttribute')
    store_attr.location = (100, 0)
    store_attr.data_type = 'FLOAT_COLOR'
    store_attr.domain = 'POINT'
    store_attr.inputs["Name"].default_value = "color"

    # 円柱断面
    circle = nodes.new('GeometryNodeCurvePrimitiveCircle')
    circle.location = (-100, 200)

    # カーブをチューブに
    curve_to_mesh = nodes.new('GeometryNodeCurveToMesh')
    curve_to_mesh.location = (350, 0)

    circle.inputs["Resolution"].default_value = 4

    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (700, 0)

    # ----------------------------------------------------------
    # 接続
    # ----------------------------------------------------------
    l = links.new

    # Group Input → Trim
    l(group_in.outputs['Geometry'],      trim.inputs['Curve'])
    l(group_in.outputs['Segment Start'], trim.inputs['Start'])
    l(group_in.outputs['Segment End'],   trim.inputs['End'])

    # Trim → Resample
    l(trim.outputs['Curve'], resample.inputs['Curve'])
    l(group_in.outputs['Sample Count'], resample.inputs['Count'])

    # Resample → Curve Length（長さ[m]）
    l(resample.outputs['Curve'], curve_length.inputs['Curve'])

    # Resample → Store Attribute（ジオメトリ本体）
    l(resample.outputs['Curve'], store_attr.inputs['Geometry'])

    # Spline Parameter (0〜1) → Math(入力1)
    l(spline_param.outputs['Factor'], math_mul.inputs[0])

    # Curve Length (m) → Math(入力2)
    l(curve_length.outputs['Length'], math_mul.inputs[1])

    # Switch: Bool → "Use Absolute Length"
    l(group_in.outputs['Use Absolute Length'], switch.inputs['Switch'])

    # Switch False（OFF時） = 0〜1 の Factor
    l(spline_param.outputs['Factor'], switch.inputs['False'])

    # Switch True（ON時） = 絶対長さ[m] = Factor * Length
    l(math_mul.outputs['Value'], switch.inputs['True'])

    l(switch.outputs['Output'], math_add.inputs[0])
    l(group_in.outputs['Color Offset'], math_add.inputs[1])

    # Switch 出力 → Combine Color (R,G,B 全て同じ値)
    l(math_add.outputs['Output'], comb_color.inputs['Red'])
    l(math_add.outputs['Output'], comb_color.inputs['Green'])
    l(math_add.outputs['Output'], comb_color.inputs['Blue'])

    # Combine Color → Store Named Attribute
    l(comb_color.outputs['Color'], store_attr.inputs['Value'])

    # Radius → Circle
    l(group_in.outputs['Radius'], circle.inputs['Radius'])

    # Store Named Attribute → Curve to Mesh（Curve）
    l(store_attr.outputs['Geometry'], curve_to_mesh.inputs['Curve'])

    # Circle → Curve to Mesh（Profile Curve）
    l(circle.outputs['Curve'], curve_to_mesh.inputs['Profile Curve'])
    l(group_in.outputs['Fill Caps'], curve_to_mesh.inputs['Fill Caps'])
    # Curve to Mesh → Group Output
    l(curve_to_mesh.outputs['Mesh'], realize.inputs['Geometry'])

    l(realize.outputs['Geometry'], group_out.inputs['Geometry'])

    return ng


def add_modifier_to_object(obj, ng):
    """Attach ``ng`` as a Geometry Nodes modifier to ``obj``."""

    if obj is None:
        raise ValueError("An object is required to add the modifier to.")

    modifier = None
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group == ng:
            modifier = mod
            break

    if modifier is None:
        modifier = obj.modifiers.new(name=ng.name, type='NODES')
        modifier.node_group = ng

    return modifier


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

    return sorted_obj, curve_obj


def ensure_gradient_geometry(curve_obj):
    """Ensure that ``curve_obj`` has the gradient geometry nodes modifier."""

    if curve_obj is None or curve_obj.type != 'CURVE':
        raise ValueError("A curve object is required.")
    node_group = create_curve_segment_cylinder_gn()
    add_modifier_to_object(curve_obj, node_group)
    return curve_obj

def create_gradient_curve_from_selection():
    """Create a curve with the geometry nodes modifier from the current selection."""

    _sorted_obj, curve_obj = sort_vtx_and_create_curve()
    return ensure_gradient_geometry(curve_obj)