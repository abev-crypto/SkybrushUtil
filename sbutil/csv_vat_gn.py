import bpy
import csv
import os

# === 設定 ===
# CSV が入っているフォルダを指定（Blender ファイルからの相対パスも可）
CSV_FOLDER = r"F:\アタリ Dropbox\abe yuya\works\DroneTest\hato\07_hato_v04_200"  # ←ここを書き換える

# 画像名・GN名など
POS_IMAGE_NAME = "VAT_Pos"
COL_IMAGE_NAME = "VAT_Color"
OBJ_NAME = "DronePoints"
GN_NAME = "GN_DroneVAT"

# === CSV 読み込み & VAT データ作成 ===

def load_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort()  # Drone 1, Drone 2... のように名前順に並べる想定
    if not files:
        raise RuntimeError("指定フォルダに CSV がありません: " + folder)

    all_tracks = []  # [drone_index][frame_index] = (x, y, z, r, g, b)
    time_list = None

    for fname in files:
        path = os.path.join(folder, fname)
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            track = []
            local_time_list = []
            for row in reader:
                t_ms = float(row["Time [msec]"])
                x = float(row["x [m]"])
                y = float(row["y [m]"])
                z = float(row["z [m]"])
                r = float(row["Red"])
                g = float(row["Green"])
                b = float(row["Blue"])
                track.append((x, y, z, r, g, b))
                local_time_list.append(t_ms)

        if time_list is None:
            time_list = local_time_list
        else:
            # 簡易チェック：行数が同じかどうかだけ見る
            if len(time_list) != len(local_time_list):
                raise RuntimeError(
                    f"CSV 間で行数が違います: {fname} ({len(local_time_list)} vs {len(time_list)})"
                )

        all_tracks.append(track)

    return time_list, all_tracks


def compute_min_max(all_tracks):
    # 全ドローン・全フレームの x,y,z の min/max
    xs, ys, zs = [], [], []
    for track in all_tracks:
        for (x, y, z, r, g, b) in track:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        raise RuntimeError("CSV データが空です")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def create_image(name, width, height):
    # 既存があれば再利用／上書き
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name=name, width=width, height=height, float_buffer=True)
    return img


def build_vat_images(time_list, all_tracks):
    frame_count = len(time_list)
    drone_count = len(all_tracks)

    pos_min, pos_max = compute_min_max(all_tracks)
    min_x, min_y, min_z = pos_min
    max_x, max_y, max_z = pos_max

    # range が 0 にならないように保険
    rx = max_x - min_x or 1.0
    ry = max_y - min_y or 1.0
    rz = max_z - min_z or 1.0

    # 画像作成
    pos_img = create_image(POS_IMAGE_NAME, frame_count, drone_count)
    col_img = create_image(COL_IMAGE_NAME, frame_count, drone_count)

    pos_pixels = [0.0] * (frame_count * drone_count * 4)
    col_pixels = [0.0] * (frame_count * drone_count * 4)

    # Blender の Image は (x, y) = (0, 0) が左下
    # x = frame index, y = drone index として詰める
    for drone_idx, track in enumerate(all_tracks):
        if len(track) != frame_count:
            raise RuntimeError(f"トラック {drone_idx} のフレーム数が不一致")
        for frame_idx, (x, y, z, r, g, b) in enumerate(track):
            # 0..1 正規化
            nx = (x - min_x) / rx
            ny = (y - min_y) / ry
            nz = (z - min_z) / rz

            # 色もとりあえず 0..255 想定で 0..1 に正規化
            cr = r / 255.0 if r > 1.0 else r
            cg = g / 255.0 if g > 1.0 else g
            cb = b / 255.0 if b > 1.0 else b

            idx = (drone_idx * frame_count + frame_idx) * 4

            pos_pixels[idx + 0] = nx
            pos_pixels[idx + 1] = ny
            pos_pixels[idx + 2] = nz
            pos_pixels[idx + 3] = 1.0

            col_pixels[idx + 0] = cr
            col_pixels[idx + 1] = cg
            col_pixels[idx + 2] = cb
            col_pixels[idx + 3] = 1.0

    pos_img.pixels[:] = pos_pixels
    col_img.pixels[:] = col_pixels

    return pos_img, col_img, pos_min, pos_max, frame_count, drone_count


# === Geometry ノード構築 ===

def create_drone_points_object(drone_count):
    # 既存オブジェクトを消して作り直し
    obj = bpy.data.objects.get(OBJ_NAME)
    if obj is not None:
        if obj.data:
            bpy.data.meshes.remove(obj.data)
        bpy.data.objects.remove(obj)

    mesh = bpy.data.meshes.new(OBJ_NAME + "_Mesh")
    verts = [(0.0, 0.0, 0.0)] * drone_count
    edges = []
    faces = []
    mesh.from_pydata(verts, edges, faces)
    mesh.update()

    obj = bpy.data.objects.new(OBJ_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_gn_vat_group(pos_img, col_img, pos_min, pos_max, frame_count, drone_count):
    # 既存の NodeGroup を削除
    ng = bpy.data.node_groups.get(GN_NAME)
    if ng is not None:
        bpy.data.node_groups.remove(ng)

    ng = bpy.data.node_groups.new(GN_NAME, 'GeometryNodeTree')

    # --- インターフェイス (Blender 4.x スタイル) ---
    iface = ng.interface

    # INPUTS
    geo_in = iface.new_socket(
        name="Geometry", in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description=""
    )
    posmin_in = iface.new_socket(
        name="Pos Min", in_out='INPUT',
        socket_type='NodeSocketVector',
        description="最小位置 (x,y,z)"
    )
    posmax_in = iface.new_socket(
        name="Pos Max", in_out='INPUT',
        socket_type='NodeSocketVector',
        description="最大位置 (x,y,z)"
    )
    startframe_in = iface.new_socket(
        name="Start Frame", in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="アニメーション開始フレーム"
    )
    framecount_in = iface.new_socket(
        name="Frame Count", in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="フレーム数"
    )
    dronecount_in = iface.new_socket(
        name="Drone Count", in_out='INPUT',
        socket_type='NodeSocketInt',
        description="ドローン数"
    )

    # OUTPUTS
    geo_out = iface.new_socket(
        name="Geometry", in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description=""
    )

    # デフォルト値設定（Vector は 3要素だけ渡す）
    posmin_in.default_value = pos_min
    posmax_in.default_value = pos_max
    startframe_in.default_value = float(bpy.context.scene.frame_start)
    framecount_in.default_value = float(frame_count)
    dronecount_in.default_value = int(drone_count)

    # --- ノード作成 ---

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    n_input = nodes.new("NodeGroupInput")
    n_input.location = (-900, 0)

    n_output = nodes.new("NodeGroupOutput")
    n_output.location = (500, 0)

    # Scene Time
    n_time = nodes.new("GeometryNodeInputSceneTime")
    n_time.location = (-700, 200)

    # Index
    n_index = nodes.new("GeometryNodeInputIndex")
    n_index.location = (-700, -50)

    # Math: (Frame - StartFrame)
    n_sub = nodes.new("ShaderNodeMath")
    n_sub.operation = 'SUBTRACT'
    n_sub.location = (-500, 200)

    # Math: (Frame - StartFrame) / (FrameCount - 1)
    n_div = nodes.new("ShaderNodeMath")
    n_div.operation = 'DIVIDE'
    n_div.use_clamp = True  # 0..1 にクランプ
    n_div.location = (-300, 200)

    # Math: FrameCount - 1
    n_fc_minus1 = nodes.new("ShaderNodeMath")
    n_fc_minus1.operation = 'SUBTRACT'
    n_fc_minus1.location = (-500, 50)

    # Math: DroneCount - 1
    n_dc_minus1 = nodes.new("ShaderNodeMath")
    n_dc_minus1.operation = 'SUBTRACT'
    n_dc_minus1.location = (-500, -250)

    # Math: Index / (DroneCount - 1)
    n_div_index = nodes.new("ShaderNodeMath")
    n_div_index.operation = 'DIVIDE'
    n_div_index.use_clamp = True
    n_div_index.location = (-300, -250)

    # Combine XYZ → UV
    n_combine_uv = nodes.new("ShaderNodeCombineXYZ")
    n_combine_uv.location = (-100, 0)

    # Image Texture (Pos) - Geometry Nodes 用
    n_tex_pos = nodes.new("GeometryNodeImageTexture")
    n_tex_pos.location = (100, 150)
    n_tex_pos.interpolation = 'Linear'
    n_tex_pos.extension = 'EXTEND'
    # ここが重要：Image ソケットの default_value に渡す
    n_tex_pos.inputs["Image"].default_value = pos_img

    # Image Texture (Color) - Geometry Nodes 用
    n_tex_col = nodes.new("GeometryNodeImageTexture")
    n_tex_col.location = (100, -50)
    n_tex_col.interpolation = 'Linear'
    n_tex_col.extension = 'EXTEND'
    n_tex_col.inputs["Image"].default_value = col_img

    # VectorMath: PosMax - PosMin
    n_vsub = nodes.new("ShaderNodeVectorMath")
    n_vsub.operation = 'SUBTRACT'
    n_vsub.location = (300, 250)

    # VectorMath: encoded * (PosMax - PosMin)
    n_vmul = nodes.new("ShaderNodeVectorMath")
    n_vmul.operation = 'MULTIPLY'
    n_vmul.location = (500, 150)

    # VectorMath: PosMin + ...
    n_vadd = nodes.new("ShaderNodeVectorMath")
    n_vadd.operation = 'ADD'
    n_vadd.location = (700, 150)

    # Set Position
    n_setpos = nodes.new("GeometryNodeSetPosition")
    n_setpos.location = (900, 100)

    # Store Named Attribute (Color)
    n_store_col = nodes.new("GeometryNodeStoreNamedAttribute")
    n_store_col.location = (1100, 0)
    n_store_col.data_type = 'FLOAT_COLOR'
    n_store_col.domain = 'POINT'
    n_store_col.name = "vat_color"

    # --- 接続 ---

    # Geometry パス
    links.new(n_input.outputs["Geometry"], n_setpos.inputs["Geometry"])
    links.new(n_setpos.outputs["Geometry"], n_store_col.inputs["Geometry"])
    links.new(n_store_col.outputs["Geometry"], n_output.inputs["Geometry"])

    # Scene Time → Subtract
    links.new(n_time.outputs["Frame"], n_sub.inputs[0])
    links.new(n_input.outputs["Start Frame"], n_sub.inputs[1])

    # FrameCount - 1
    links.new(n_input.outputs["Frame Count"], n_fc_minus1.inputs[0])
    n_fc_minus1.inputs[1].default_value = 1.0

    # (Frame - StartFrame) / (FrameCount - 1)
    links.new(n_sub.outputs[0], n_div.inputs[0])
    links.new(n_fc_minus1.outputs[0], n_div.inputs[1])

    # U
    links.new(n_div.outputs[0], n_combine_uv.inputs[0])

    # DroneCount - 1
    links.new(n_input.outputs["Drone Count"], n_dc_minus1.inputs[0])
    n_dc_minus1.inputs[1].default_value = 1.0

    # Index / (DroneCount - 1)
    links.new(n_index.outputs["Index"], n_div_index.inputs[0])
    links.new(n_dc_minus1.outputs[0], n_div_index.inputs[1])

    # V
    links.new(n_div_index.outputs[0], n_combine_uv.inputs[1])

    # UV → Textures
    links.new(n_combine_uv.outputs["Vector"], n_tex_pos.inputs["Vector"])
    links.new(n_combine_uv.outputs["Vector"], n_tex_col.inputs["Vector"])

    # PosMax - PosMin
    links.new(n_input.outputs["Pos Max"], n_vsub.inputs[0])
    links.new(n_input.outputs["Pos Min"], n_vsub.inputs[1])

    # encoded * range
    links.new(n_tex_pos.outputs["Color"], n_vmul.inputs[0])
    links.new(n_vsub.outputs["Vector"], n_vmul.inputs[1])

    # PosMin + ...
    links.new(n_input.outputs["Pos Min"], n_vadd.inputs[0])
    links.new(n_vmul.outputs["Vector"], n_vadd.inputs[1])

    # Set Position (absolute)
    links.new(n_vadd.outputs["Vector"], n_setpos.inputs["Position"])

    # Store Named Attribute (Color)
    links.new(n_tex_col.outputs["Color"], n_store_col.inputs["Value"])

    return ng


def apply_gn_to_object(obj, node_group):
    # 既存の NODES モディファイアは消しておく
    for m in list(obj.modifiers):
        if m.type == 'NODES':
            obj.modifiers.remove(m)
    mod = obj.modifiers.new(name="Drone VAT", type='NODES')
    mod.node_group = node_group
    return mod


# === 実行 ===

def main():
    folder = bpy.path.abspath(CSV_FOLDER)
    if not os.path.isdir(folder):
        raise RuntimeError("CSV_FOLDER がディレクトリではありません: " + folder)

    time_list, all_tracks = load_csv_files(folder)
    pos_img, col_img, pos_min, pos_max, frame_count, drone_count = build_vat_images(time_list, all_tracks)

    obj = create_drone_points_object(drone_count)
    node_group = create_gn_vat_group(pos_img, col_img, pos_min, pos_max, frame_count, drone_count)
    apply_gn_to_object(obj, node_group)

    print("VAT 作成完了")
    print(f"フレーム数: {frame_count}, ドローン数: {drone_count}")
    print(f"PosMin: {pos_min}, PosMax: {pos_max}")

main()
