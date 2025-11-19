import bpy


NODE_GROUP_NAME_L = "PathFollowLine"


def get_or_create_pfl_node_group():
    # 既存グループがあれば再利用
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME_L)
    if ng is not None:
        return ng

    # 新規 Geometry Node Group 作成
    ng = bpy.data.node_groups.new(NODE_GROUP_NAME_L, 'GeometryNodeTree')

    # --- インターフェイス定義 ---
    iface = ng.interface

    # 入力: Geometry (メッシュ)
    iface.new_socket(
        name="Geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description="Mesh with line vertices",
    )

    # 入力: CurveObject (追従するカーブオブジェクト)
    sock_curve_obj = iface.new_socket(
        name="CurveObject",
        in_out='INPUT',
        socket_type='NodeSocketObject',
        description="Curve object to follow",
    )

    # 入力: Speed
    sock_speed = iface.new_socket(
        name="Speed",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Speed along curve (units per second)",
    )
    sock_speed.default_value = 0.2

    # 入力: Offset
    sock_offset = iface.new_socket(
        name="Offset",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Additional offset along curve (0-1)",
    )
    sock_offset.default_value = 0.0

    # 出力: Geometry
    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description="Output geometry",
    )

    # カーブ入力ソケットの identifier をノードグループに保存しておく
    try:
        ng["curve_socket_identifier"] = sock_curve_obj.identifier
    except AttributeError:
        # もし identifier が無いバージョンでも、とりあえずスクリプトは落ちないように
        pass

    # --- ノード構築 ---
    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    # Group I/O
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1100, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (700, 0)

    # Set Position（頂点位置をカーブ上に置き換える）
    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (350, 0)

    # Sample Curve（カーブの t=0〜1 から位置サンプル）
    sample_curve = nodes.new("GeometryNodeSampleCurve")
    sample_curve.location = (150, 200)

    # Object Info（CurveObject → Geometry）
    obj_info = nodes.new("GeometryNodeObjectInfo")
    obj_info.location = (-200, 200)
    obj_info.transform_space = 'RELATIVE'

    # Domain Size（頂点数取得 → インデックス正規化用）
    domain_size = nodes.new("GeometryNodeAttributeDomainSize")
    domain_size.location = (-700, -220)

    # Index（頂点インデックス）
    index_node = nodes.new("GeometryNodeInputIndex")
    index_node.location = (-700, 0)

    # Scene Time（時間）
    scene_time = nodes.new("GeometryNodeInputSceneTime")
    scene_time.location = (-700, 220)

    # Math ノード群
    # count - 1
    math_sub = nodes.new("ShaderNodeMath")
    math_sub.operation = 'SUBTRACT'
    math_sub.location = (-500, -220)
    math_sub.inputs[1].default_value = 1.0

    # max(count-1, 1)
    math_max = nodes.new("ShaderNodeMath")
    math_max.operation = 'MAXIMUM'
    math_max.location = (-300, -220)
    math_max.inputs[1].default_value = 1.0

    # index / max(count-1, 1)
    math_div = nodes.new("ShaderNodeMath")
    math_div.operation = 'DIVIDE'
    math_div.location = (-100, -100)

    # time * speed
    math_mul = nodes.new("ShaderNodeMath")
    math_mul.operation = 'MULTIPLY'
    math_mul.location = (-500, 220)

    # time*speed + offset
    math_add_time = nodes.new("ShaderNodeMath")
    math_add_time.operation = 'ADD'
    math_add_time.location = (-300, 220)

    # base_factor + (time*speed+offset)
    math_add = nodes.new("ShaderNodeMath")
    math_add.operation = 'ADD'
    math_add.location = (-50, 180)

    # fract( base_factor + time_offset )
    math_frac = nodes.new("ShaderNodeMath")
    math_frac.operation = 'FRACT'
    math_frac.location = (150, 120)

    # --- GroupInput のソケット参照（インデックス） ---
    # 0: Geometry, 1: CurveObject, 2: Speed, 3: Offset
    gi_geo = group_in.outputs[0]
    gi_curve_obj = group_in.outputs[1]
    gi_speed = group_in.outputs[2]
    gi_offset = group_in.outputs[3]

    # GroupOutput も 0: Geometry のみ想定
    go_geo = group_out.inputs[0]

    # --- 配線 ---

    # Geometry 流れ: Group In -> Set Position -> Group Out
    links.new(gi_geo, set_pos.inputs["Geometry"])
    links.new(set_pos.outputs["Geometry"], go_geo)

    # Domain Size（頂点数）
    links.new(gi_geo, domain_size.inputs["Geometry"])
    links.new(domain_size.outputs["Point Count"], math_sub.inputs[0])

    # max(count-1, 1)
    links.new(math_sub.outputs[0], math_max.inputs[0])

    # index / max(...)
    links.new(index_node.outputs["Index"], math_div.inputs[0])
    links.new(math_max.outputs[0], math_div.inputs[1])

    # Scene time * Speed
    links.new(scene_time.outputs["Seconds"], math_mul.inputs[0])
    links.new(gi_speed, math_mul.inputs[1])

    # time*speed + Offset
    links.new(math_mul.outputs[0], math_add_time.inputs[0])
    links.new(gi_offset, math_add_time.inputs[1])

    # base_factor + time_offset
    links.new(math_div.outputs[0], math_add.inputs[0])
    links.new(math_add_time.outputs[0], math_add.inputs[1])

    # fract(...)
    links.new(math_add.outputs[0], math_frac.inputs[0])

    # Object Info: CurveObject → Geometry
    links.new(gi_curve_obj, obj_info.inputs["Object"])

    # Sample Curve inputs: 0: Curves, 2: Factor を想定
    sc_input_curves = sample_curve.inputs[0]
    sc_input_factor = sample_curve.inputs[2]

    # Sample Curve: Curves <- Object Info.Geometry
    links.new(obj_info.outputs["Geometry"], sc_input_curves)
    # Factor <- fract計算
    links.new(math_frac.outputs[0], sc_input_factor)

    # Set Position: Position = Sampled position
    links.new(sample_curve.outputs["Position"], set_pos.inputs["Position"])

    return ng


def setup_pfl_modifier_for_selection(mesh_obj, curve_obj):
    ng = get_or_create_pfl_node_group()

    # Geometry Nodes モディファイア追加
    mod = mesh_obj.modifiers.new(name="PathFollowLine", type='NODES')
    mod.node_group = ng

    # カーブ入力ソケットの identifier から、モディファイアのプロパティ名を探してセット
    curve_id = ng.get("curve_socket_identifier", None)
    assigned = False
    if curve_id is not None:
        try:
            if curve_id in mod:
                mod[curve_id] = curve_obj
                assigned = True
        except Exception:
            pass

    # うまくいかなかったときのフォールバック ("Input_2" など)
    if not assigned:
        for key in ("Input_2", "Input_3", "Socket_2", "Socket_3"):
            try:
                if key in mod:
                    mod[key] = curve_obj
                    print(f"CurveObject を {key} に割り当てました。")
                    assigned = True
                    break
            except Exception:
                continue

    if not assigned:
        print("自動で CurveObject を割り当てられませんでした。モディファイアのパネルから手動で設定してください。")

    # Speed / Offset はデフォルト値でも問題ないので、必須ではない
    print(f"PathFollowLine ノードグループを {mesh_obj.name} に設定しました。")
    print("モディファイアパネルで:")
    print(" - CurveObject にパス用カーブオブジェクトが入っているか確認（必要なら手動で選択）")
    print(" - Speed / Offset を調整してループ動作を確認してください。")


NODE_GROUP_NAME_C = "PathFollowCluster"

def create_or_reset_pfc_node_group():
    # 既存があれば中身をリセットして使い回す
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME_C)
    if ng is None:
        ng = bpy.data.node_groups.new(NODE_GROUP_NAME_C, 'GeometryNodeTree')
    else:
        ng.interface.clear()
        ng.nodes.clear()
        ng.links.clear()

    iface = ng.interface

    # 入出力ソケット定義
    iface.new_socket(
        name="Geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description="Mesh with line vertices",
    )

    sock_curve_obj = iface.new_socket(
        name="CurveObject",
        in_out='INPUT',
        socket_type='NodeSocketObject',
        description="Curve object to follow",
    )

    sock_offset = iface.new_socket(
        name="Offset",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="0: cluster tail at start, 1: head at end",
    )
    sock_offset.min_value = 0.0
    sock_offset.max_value = 1.0
    sock_offset.default_value = 0.0

    sock_span = iface.new_socket(
        name="Span",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Fraction of curve occupied by vertex cluster",
    )
    sock_span.min_value = 0.0
    sock_span.max_value = 1.0
    sock_span.default_value = 0.3  # デフォルトで全体の3割くらいを占有

    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description="Output geometry",
    )

    nodes = ng.nodes
    links = ng.links

    # Group I/O
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1100, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (700, 0)

    # Set Position
    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (350, 0)

    # Object Info（CurveObject → Geometry）
    obj_info = nodes.new("GeometryNodeObjectInfo")
    obj_info.location = (-300, 200)
    obj_info.transform_space = 'RELATIVE'

    # Sample Curve
    sample_curve = nodes.new("GeometryNodeSampleCurve")
    sample_curve.location = (150, 200)

    # Domain Size（頂点数取得）
    domain_size = nodes.new("GeometryNodeAttributeDomainSize")
    domain_size.location = (-900, -200)

    # Index
    index_node = nodes.new("GeometryNodeInputIndex")
    index_node.location = (-900, 0)

    # Math ノード群
    # count - 1
    math_sub = nodes.new("ShaderNodeMath")
    math_sub.operation = 'SUBTRACT'
    math_sub.location = (-700, -200)
    math_sub.inputs[1].default_value = 1.0

    # max(count-1, 1)
    math_max = nodes.new("ShaderNodeMath")
    math_max.operation = 'MAXIMUM'
    math_max.location = (-500, -200)
    math_max.inputs[1].default_value = 1.0

    # index_norm = index / max(count-1, 1)
    math_div = nodes.new("ShaderNodeMath")
    math_div.operation = 'DIVIDE'
    math_div.location = (-500, -20)

    # one_minus_span = 1 - Span
    math_one_minus_span = nodes.new("ShaderNodeMath")
    math_one_minus_span.operation = 'SUBTRACT'
    math_one_minus_span.location = (-500, 200)
    math_one_minus_span.inputs[0].default_value = 1.0  # 1 - Span

    # offset_scaled = Offset * (1 - Span)
    math_offset_scaled = nodes.new("ShaderNodeMath")
    math_offset_scaled.operation = 'MULTIPLY'
    math_offset_scaled.location = (-300, 200)

    # span_scaled = index_norm * Span
    math_span_scaled = nodes.new("ShaderNodeMath")
    math_span_scaled.operation = 'MULTIPLY'
    math_span_scaled.location = (-300, 20)

    # factor = offset_scaled + span_scaled
    math_factor = nodes.new("ShaderNodeMath")
    math_factor.operation = 'ADD'
    math_factor.location = (-100, 120)

    # --- Group I/O ソケット参照 ---
    # INPUT: 0 Geometry, 1 CurveObject, 2 Offset, 3 Span
    gi_geo = group_in.outputs[0]
    gi_curve_obj = group_in.outputs[1]
    gi_offset = group_in.outputs[2]
    gi_span = group_in.outputs[3]

    go_geo = group_out.inputs[0]

    # --- 配線 ---

    # Geometry フロー
    links.new(gi_geo, set_pos.inputs["Geometry"])
    links.new(set_pos.outputs["Geometry"], go_geo)

    # Domain Size で Point Count
    links.new(gi_geo, domain_size.inputs["Geometry"])
    links.new(domain_size.outputs["Point Count"], math_sub.inputs[0])
    links.new(math_sub.outputs[0], math_max.inputs[0])

    # index_norm
    links.new(index_node.outputs["Index"], math_div.inputs[0])
    links.new(math_max.outputs[0], math_div.inputs[1])

    # one_minus_span = 1 - Span
    links.new(gi_span, math_one_minus_span.inputs[1])

    # offset_scaled = Offset * (1 - Span)
    links.new(gi_offset, math_offset_scaled.inputs[0])
    links.new(math_one_minus_span.outputs[0], math_offset_scaled.inputs[1])

    # span_scaled = index_norm * Span
    links.new(math_div.outputs[0], math_span_scaled.inputs[0])
    links.new(gi_span, math_span_scaled.inputs[1])

    # factor = offset_scaled + span_scaled
    links.new(math_offset_scaled.outputs[0], math_factor.inputs[0])
    links.new(math_span_scaled.outputs[0], math_factor.inputs[1])

    # Object Info: CurveObject → Geometry
    links.new(gi_curve_obj, obj_info.inputs["Object"])

    # Sample Curve inputs: 0 Curves, 2 Factor
    sc_input_curves = sample_curve.inputs[0]
    sc_input_factor = sample_curve.inputs[2]

    links.new(obj_info.outputs["Geometry"], sc_input_curves)
    links.new(math_factor.outputs[0], sc_input_factor)

    # Set Position: Position = Sampled position
    links.new(sample_curve.outputs["Position"], set_pos.inputs["Position"])

    return ng


def setup_pfc_modifier_for_selection(mesh_obj, curve_obj):

    ng = create_or_reset_pfc_node_group()

    # Geometry Nodes モディファイア追加
    mod = mesh_obj.modifiers.new(name="PathFollowCluster", type='NODES')
    mod.node_group = ng

        # カーブ入力ソケットの identifier から、モディファイアのプロパティ名を探してセット
    curve_id = ng.get("curve_socket_identifier", None)
    assigned = False
    if curve_id is not None:
        try:
            if curve_id in mod:
                mod[curve_id] = curve_obj
                assigned = True
        except Exception:
            pass

    # うまくいかなかったときのフォールバック ("Input_2" など)
    if not assigned:
        for key in ("Input_2", "Input_3", "Socket_2", "Socket_3"):
            try:
                if key in mod:
                    mod[key] = curve_obj
                    print(f"CurveObject を {key} に割り当てました。")
                    assigned = True
                    break
            except Exception:
                continue

    if not assigned:
        print("自動で CurveObject を割り当てられませんでした。モディファイアのパネルから手動で設定してください。")

    # ここでは CurveObject は自動セットにはこだわらず、
    # モディファイア側で手動で選んでもらう前提とする
    print(f"PathFollowCluster ノードグループを {mesh_obj.name} に設定しました。")
    print("モディファイアパネルで:")
    print(" - CurveObject にカーブオブジェクトを指定")
    print(" - Span で頂点の塊の長さ（カーブ全体に対する割合）を調整")
    print(" - Offset を 0→1 にアニメーションさせると、塊が端から端までスライドします。")

def is_curve_cyclic(curve_obj):
    """カーブオブジェクトが閉じているか判定（1本でもループしていたら True）"""
    if curve_obj.type != "CURVE":
        return False
    for spline in curve_obj.data.splines:
        if spline.use_cyclic_u:
            return True
    return False

def setup_modifier_for_selection():
        # メッシュとカーブを選択しておく前提
    sel = bpy.context.selected_objects
    if len(sel) < 2:
        raise RuntimeError("メッシュとカーブの2つ以上のオブジェクトを選択してください。")

    mesh_obj = None
    curve_obj = None

    for obj in sel:
        if obj.type == 'MESH' and mesh_obj is None:
            mesh_obj = obj
        elif obj.type == 'CURVE' and curve_obj is None:
            curve_obj = obj

    if mesh_obj is None:
        raise RuntimeError("選択オブジェクトの中に MESH が見つかりません。")
    if curve_obj is None:
        raise RuntimeError("選択オブジェクトの中に CURVE が見つかりません。")
    
    if is_curve_cyclic(curve_obj):
        # ループカーブ用 GeometryNode を生成
        setup_pfl_modifier_for_selection(mesh_obj, curve_obj)
    else:
        # テレポ無しクラスター移動バージョン
        setup_pfc_modifier_for_selection(mesh_obj, curve_obj)