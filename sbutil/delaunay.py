import bpy
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

# 現在フレームの評価済みワールド座標（アニメ/制約/親子反映）
def get_evaluated_world_positions(objects, include_hidden: bool = False):
    """Return evaluated world positions of ``objects`` at the current frame."""

    scene = bpy.context.scene
    scene.frame_set(scene.frame_current)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    positions: list[Vector] = []
    for obj in objects:
        if obj is None:
            continue
        try:
            hidden = obj.hide_get()
        except Exception:
            hidden = False
        if hidden and not include_hidden:
            continue
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            positions.append(eval_obj.matrix_world.translation.copy())
        except Exception:
            continue
    return positions


def get_selected_evaluated_world_positions(include_hidden=False):
    objects = bpy.context.selected_objects
    return get_evaluated_world_positions(objects, include_hidden=include_hidden)

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

    # （法線は既に+Y揃え済みだが、念のため）
    obj.data.flip_normals()
    if sum(p.normal.y for p in obj.data.polygons) < 0:
        obj.data.flip_normals()

    # === 4) Solidifyモディファイアで ± センター厚み＆フタ ===
    solid = obj.modifiers.new("SolidifyY", type='SOLIDIFY')
    solid.thickness = float(thickness_default)  # 例: 0.2（総厚み）
    solid.offset = 0.0                          # 中心±（=センター）
    solid.use_rim = True                        # フタ有効
    solid.use_even_offset = True                # 均一厚み

    # ある環境では追加で有効（あればON）
    for attr in ("use_quality_normals", "nonmanifold_boundary_mode"):
        if hasattr(solid, attr):
            try:
                setattr(solid, attr, True if isinstance(getattr(solid, attr), bool) else getattr(solid, attr))
            except Exception:
                pass

    print("✅ 2D Delaunay で平面生成 → +Y 厚み（Centered対応）を作成しました。")
    return obj

if __name__ == "__main__":  # pragma: no cover - manual testing helper
    create_hull_plane_from_current_positions_y_thickness(
        thickness_default=0.2, centered_default=True
    )
