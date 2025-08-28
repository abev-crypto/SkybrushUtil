import bpy
import bmesh
from mathutils import Vector

# --------- ユーティリティ ---------
def project_axis_limit(vec: Vector, axis_limit: str):
    v = Vector(vec)
    if axis_limit == "XY": v.z = 0
    elif axis_limit == "XZ": v.y = 0
    elif axis_limit == "YZ": v.x = 0
    elif axis_limit == "X": v.y = 0; v.z = 0
    elif axis_limit == "Y": v.x = 0; v.z = 0
    elif axis_limit == "Z": v.x = 0; v.y = 0
    return v

def build_selected_graph(bm_verts):
    # 隣接（選択内の辺）グラフを作る
    index_map = {v: i for i, v in enumerate(bm_verts)}
    adj = {i: set() for i in range(len(bm_verts))}
    for e in [edge for v in bm_verts for edge in v.link_edges]:
        v0, v1 = e.verts
        if v0 in index_map and v1 in index_map:
            i0, i1 = index_map[v0], index_map[v1]
            adj[i0].add(i1)
            adj[i1].add(i0)
    return adj, index_map

def endpoints_edge_path(bm_verts):
    adj, index_map = build_selected_graph(bm_verts)
    if not any(adj.values()):
        return None  # 辺つながりが無い
    deg1 = [i for i, nbrs in adj.items() if len(nbrs) == 1]
    if len(deg1) >= 2:
        # 端の2つ（もし3つ以上あれば最遠の組を選ぶ）
        pts = [bm_verts[i].co for i in deg1]
        best = (0, deg1[0], deg1[1])
        for i in range(len(deg1)):
            for j in range(i+1, len(deg1)):
                d = (pts[i] - pts[j]).length_squared
                if d > best[0]:
                    best = (d, deg1[i], deg1[j])
        return best[1], best[2]
    return None

def endpoints_farthest(bm_verts):
    best = (0.0, 0, 1)
    for i in range(len(bm_verts)):
        ci = bm_verts[i].co
        for j in range(i+1, len(bm_verts)):
            d2 = (ci - bm_verts[j].co).length_squared
            if d2 > best[0]:
                best = (d2, i, j)
    return best[1], best[2]

def endpoints_active_farthest(bm_verts, active_vert):
    if active_vert not in bm_verts:
        return endpoints_farthest(bm_verts)
    i0 = bm_verts.index(active_vert)
    c0 = active_vert.co
    j_best, d_best = 0, -1.0
    for j, v in enumerate(bm_verts):
        d2 = (c0 - v.co).length_squared
        if d2 > d_best:
            d_best, j_best = d2, j
    return i0, j_best

def order_by_edge_path(bm_verts, i_start, i_end):
    # BFSで最短経路（辺ベース）を復元
    adj, _ = build_selected_graph(bm_verts)
    if not any(adj.values()):
        return None
    from collections import deque
    prev = {i_start: -1}
    dq = deque([i_start])
    while dq:
        u = dq.popleft()
        if u == i_end: break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                dq.append(v)
    if i_end not in prev:
        return None
    path = []
    cur = i_end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def order_by_projection(bm_verts, i_start, i_end):
    a, b = bm_verts[i_start].co, bm_verts[i_end].co
    ab = b - a
    if ab.length < 1e-12:
        return list(range(len(bm_verts)))
    return sorted(range(len(bm_verts)),
                  key=lambda i: (bm_verts[i].co - a).dot(ab) / ab.length)

def catmull_rom(p0, p1, p2, p3, t):
    # 標準 Catmull–Rom（張力=0.5）
    t2, t3 = t*t, t*t*t
    return ( ( -0.5*t3 + t2 - 0.5*t) * p0 +
             (  1.5*t3 - 2.5*t2 + 1.0) * p1 +
             ( -1.5*t3 + 2.0*t2 + 0.5*t) * p2 +
             (  0.5*t3 - 0.5*t2      ) * p3 )

def polyline_resample(points, count):
    # 弧長等間隔で count 個にサンプリング
    import math
    seglen = [ (points[i+1]-points[i]).length for i in range(len(points)-1) ]
    L = sum(seglen)
    if L < 1e-12 or count <= 1:
        return [points[0]]*count
    targets = [L * i/(count-1) for i in range(count)]
    res = []
    acc = 0.0
    i = 0
    cur = points[0]
    for tL in targets:
        while i < len(seglen)-1 and acc + seglen[i] < tL:
            acc += seglen[i]; i += 1
        remain = tL - acc
        if seglen[i] < 1e-12:
            res.append(points[i])
        else:
            alpha = remain / seglen[i]
            res.append(points[i].lerp(points[i+1], alpha))
    return res

# --------- オペレーター ---------
class MESH_OT_reflow_vertices(bpy.types.Operator):
    bl_idname = "mesh.reflow_vertices"
    bl_label = "Reflow Vertices"
    bl_options = {'REGISTER', 'UNDO'}

    flow_mode: bpy.props.EnumProperty(
        name="Flow",
        items=[
            ("LINEAR", "Linear (Polyline)", "折れ線→等間隔"),
            ("SPLINE", "Spline (Catmull-Rom)", "スプライン→等間隔"),
            ("REPULSION", "Repulsion", "最小距離を保つ反発"),
        ],
        default="SPLINE"
    )

    endpoint_mode: bpy.props.EnumProperty(
        name="Endpoints",
        items=[
            ("EDGE_PATH", "Edge Path", "選択辺の端(次数1)を端点に"),
            ("AUTO_FARTHEST", "Auto Farthest", "最遠ペア"),
            ("ACTIVE_FARTHEST", "Active → Farthest", "アクティブから最遠"),
        ],
        default="EDGE_PATH"
    )

    min_distance: bpy.props.FloatProperty(name="Min Distance (Repulsion)", default=0.1, min=0.0)
    iterations: bpy.props.IntProperty(name="Iterations (Repulsion)", default=10, min=1)

    axis_limit: bpy.props.EnumProperty(
        name="Axis Limit",
        items=[("XYZ","XYZ",""),("XY","XY",""),("XZ","XZ",""),("YZ","YZ",""),("X","X",""),("Y","Y",""),("Z","Z","")],
        default="XY"
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "flow_mode")
        layout.prop(self, "endpoint_mode")
        if self.flow_mode == "REPULSION":
            layout.prop(self, "min_distance")
            layout.prop(self, "iterations")
        layout.prop(self, "axis_limit")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Mesh object required")
            return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        sel = [v for v in bm.verts if v.select]
        if len(sel) < 2:
            self.report({'WARNING'}, "Select at least two vertices")
            return {'CANCELLED'}

        # ---- REPULSION モード ----
        if self.flow_mode == "REPULSION":
            for _ in range(self.iterations):
                for i, v1 in enumerate(sel):
                    for v2 in sel[i+1:]:
                        d = v1.co - v2.co
                        L = d.length
                        if L < self.min_distance and L > 1e-12:
                            move = d.normalized() * (self.min_distance - L) * 0.5
                            move = project_axis_limit(move, self.axis_limit)
                            v1.co += move
                            v2.co -= move
            bmesh.update_edit_mesh(obj.data)
            return {'FINISHED'}

        # ---- 端点決定 ----
        active_bmv = bm.select_history.active
        if not isinstance(active_bmv, bmesh.types.BMVert):
            active_bmv = None
        if self.endpoint_mode == "EDGE_PATH":
            ep = endpoints_edge_path(sel)
            if ep is None:
                ep = endpoints_farthest(sel)
        elif self.endpoint_mode == "ACTIVE_FARTHEST":
            ep = endpoints_active_farthest(sel, active_bmv)
        else:
            ep = endpoints_farthest(sel)
        i_start, i_end = ep

        # ---- 並び順確定 ----
        order = order_by_edge_path(sel, i_start, i_end)
        if order is None:
            order = order_by_projection(sel, i_start, i_end)
        ordered_pts = [sel[i].co.copy() for i in order]

        # ---- 基準ライン（折れ線 or スプライン）を高密度サンプルに ----
        dense = []
        if self.flow_mode == "LINEAR" or len(ordered_pts) < 4:
            dense = ordered_pts[:]  # 既存折れ線
        else:
            # Catmull-Rom：端を延長して C1 を保つ
            P = ordered_pts
            Pext = [P[0] + (P[0]-P[1])] + P + [P[-1] + (P[-1]-P[-2])]
            seg_samples = 16  # セグメントあたりのサンプル密度
            for i in range(len(Pext)-3):
                p0,p1,p2,p3 = Pext[i],Pext[i+1],Pext[i+2],Pext[i+3]
                for s in range(seg_samples):
                    t = s/seg_samples
                    dense.append(catmull_rom(p0,p1,p2,p3,t))
            dense.append(Pext[-2].copy())

        # ---- 弧長等間隔で選択数にリサンプル ----
        target_count = len(sel)
        resampled = polyline_resample(dense, target_count)

        # ---- 座標を適用（軸制限を適用した差分だけ動かす）----
        for idx, v in enumerate(order):
            cur = sel[v].co
            tgt = resampled[idx]
            delta = project_axis_limit(tgt - cur, self.axis_limit)
            sel[v].co = cur + delta

        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


class MESH_OT_repel_from_neighbors(bpy.types.Operator):
    bl_idname = "mesh.repel_from_neighbors"
    bl_label = "Repel From Neighbors"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = (
        "Move selected vertices away from surrounding vertices to ensure"
        " a minimum proximity distance"
    )

    min_distance: bpy.props.FloatProperty(name="Min Distance", default=0.1, min=0.0)
    iterations: bpy.props.IntProperty(name="Iterations", default=10, min=1)
    axis_limit: bpy.props.EnumProperty(
        name="Axis Limit",
        items=[
            ("XYZ", "XYZ", ""),
            ("XY", "XY", ""),
            ("XZ", "XZ", ""),
            ("YZ", "YZ", ""),
            ("X", "X", ""),
            ("Y", "Y", ""),
            ("Z", "Z", ""),
        ],
        default="XY",
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "object", None)
        return obj is not None and obj.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "min_distance")
        layout.prop(self, "iterations")
        layout.prop(self, "axis_limit")

    def invoke(self, context, event):
        # Prefill from Skybrush safety threshold if present
        try:
            t = context.scene.skybrush.safety_check.proximity_warning_threshold
            if t and t > 0:
                self.min_distance = float(t)
        except Exception:
            pass
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Mesh object required")
            return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        selected = [v for v in bm.verts if v.select]
        if not selected:
            self.report({'WARNING'}, "Select at least one vertex")
            return {'CANCELLED'}
        others = [v for v in bm.verts if not v.select]
        if not others:
            self.report({'INFO'}, "No unselected vertices to repel from")
            return {'CANCELLED'}
        min_d = max(self.min_distance, 0.0)
        for _ in range(self.iterations):
            disps = {v: Vector((0.0, 0.0, 0.0)) for v in selected}
            for v in selected:
                acc = disps[v]
                vc = v.co
                for u in others:
                    d = vc - u.co
                    L = d.length
                    if L < 1e-12:
                        continue
                    if L < min_d:
                        acc += d.normalized() * (min_d - L) * 0.5
            # apply
            for v, move in disps.items():
                move = project_axis_limit(move, self.axis_limit)
                v.co += move
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


classes = (MESH_OT_reflow_vertices, MESH_OT_repel_from_neighbors)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
