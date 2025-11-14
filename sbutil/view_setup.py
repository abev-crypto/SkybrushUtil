import bpy
import math
from mathutils import Vector

def setup_glare_compositor(scene=None):
    if scene is None:
        scene = bpy.context.scene

    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    # ----- Render Layers -----
    n_rl = nodes.new("CompositorNodeRLayers")
    n_rl.location = (-400, 0)

    # View Layer 名は view_layers[0].name から取得
    default_view_layer_name = scene.view_layers[0].name
    n_rl.layer = default_view_layer_name

    # ----- Glare -----
    n_glare = nodes.new("CompositorNodeGlare")
    n_glare.location = (0, 0)
    n_glare.glare_type = 'BLOOM'
    n_glare.quality = 'MEDIUM'
    n_glare.mix = 0.0
    n_glare.threshold = 0.2
    n_glare.size = 5

    # ----- Composite -----
    n_comp = nodes.new("CompositorNodeComposite")
    n_comp.location = (400, 0)
    n_comp.use_alpha = True

    # ----- Links -----
    links.new(n_rl.outputs["Image"], n_glare.inputs["Image"])
    links.new(n_glare.outputs["Image"], n_comp.inputs["Image"])

setup_glare_compositor()

def ensure_camera():
    """シーンにカメラがなければ新規作成して返す"""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new("AutoFramingCamera")
        cam = bpy.data.objects.new("AutoFramingCamera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    return cam


def get_world_bbox_of_objects(objects):
    """複数オブジェクトのワールド座標BB(min_vec, max_vec)を返す"""
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, w.x)
            min_v.y = min(min_v.y, w.y)
            min_v.z = min(min_v.z, w.z)
            max_v.x = max(max_v.x, w.x)
            max_v.y = max(max_v.y, w.y)
            max_v.z = max(max_v.z, w.z)

    return min_v, max_v


def frame_selection_from_neg_y(margin_scale=1.2):
    """
    選択オブジェクトが画面に収まるように、
    Y- 側にカメラを置き +Y 方向を向けて配置する
    （画面上の横 = X、縦 = Z としてサイズ計算）
    """
    scene = bpy.context.scene
    selected = [o for o in bpy.context.selected_objects if o.type != 'CAMERA']

    if not selected:
        print("オブジェクトが選択されていません。")
        return

    cam_obj = ensure_camera()
    cam_data = cam_obj.data

    # 解像度をHDに設定
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100

    # BB取得
    bb_min, bb_max = get_world_bbox_of_objects(selected)

    center = (bb_min + bb_max) * 0.5
    size = bb_max - bb_min

    # 画面に映る平面は X(横) / Z(縦)
    width = size.x * margin_scale
    height = size.z * margin_scale
    depth_y = size.y

    # カメラFOV
    angle_x = cam_data.angle_x
    angle_y = cam_data.angle_y

    half_w = width * 0.5
    half_h = height * 0.5

    dist_x = half_w / math.tan(angle_x * 0.5) if angle_x > 0 else 0.0
    dist_y = half_h / math.tan(angle_y * 0.5) if angle_y > 0 else 0.0

    # FOVから必要距離
    distance_fov = max(dist_x, dist_y)

    # オブジェクトのY方向厚み分だけは必ず前に置く（カメラがめり込まないように）
    half_depth_y = depth_y * 0.5
    distance = max(distance_fov, half_depth_y + 0.5)  # 0.5m 余裕

    # カメラ位置：Y- 側（center.y より手前の負方向）
    cam_location = Vector((center.x, center.y - distance, center.z))
    cam_obj.location = cam_location

    # カメラの向き：中心を見るように回転（-Z 軸がターゲットを向き、Y が上）
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # クリップ面
    cam_data.clip_start = 0.1
    cam_data.clip_end = max(1000.0, distance * 3.0)

    print("Y- 側からフレーミングするカメラを再配置しました。")


# 実行
frame_selection_from_neg_y(margin_scale=1.4)
