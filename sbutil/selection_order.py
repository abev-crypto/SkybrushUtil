import bpy

TAG = "selection_order"

def get_ordered_selected_objects():
    """仕込んだインデックス順で選択中を返す（未タグは後ろ）"""
    sel = [o for o in bpy.context.selected_objects]
    return sorted(sel, key=lambda o: o.get(TAG, 10**9))

def _clear_all_tags():
    for o in bpy.data.objects:
        if TAG in o:
            del o[TAG]

def _update_selection_order():
    ctx = bpy.context
    if ctx.mode != "OBJECT":
        return
    sel = list(ctx.selected_objects)
    if not sel:
        _clear_all_tags()
        return
    # 既存の順序を保持しつつ、新規選択分だけ末尾にタグ付け
    current = sorted((o for o in sel if TAG in o), key=lambda o: o[TAG])
    idx = 0
    for o in current:
        o[TAG] = idx; idx += 1
    for o in sel:
        if TAG not in o:
            o[TAG] = idx; idx += 1
    # 選択解除されたオブジェクトのタグは消す
    for o in bpy.data.objects:
        if o not in sel and TAG in o:
            del o[TAG]

def _selection_change_handler(_scene):
    # 依存グラフの更新のうち、選択更新っぽい場合のみ反応
    dg = bpy.context.view_layer.depsgraph
    is_sel_update = any(
        not u.is_updated_geometry and not u.is_updated_transform and not u.is_updated_shading
        for u in dg.updates
    )
    if is_sel_update:
        _update_selection_order()

def register_selection_order_tracker():
    # 二重登録防止
    for f in list(bpy.app.handlers.depsgraph_update_post):
        if getattr(f, "__name__", "") == "_selection_change_handler":
            bpy.app.handlers.depsgraph_update_post.remove(f)
    bpy.app.handlers.depsgraph_update_post.append(_selection_change_handler)

def unregister_selection_order_tracker():
    for f in list(bpy.app.handlers.depsgraph_update_post):
        if getattr(f, "__name__", "") == "_selection_change_handler":
            bpy.app.handlers.depsgraph_update_post.remove(f)
    _clear_all_tags()

# 使い方:
# register_selection_order_tracker() を一度実行 → 普段通り選択
# 必要なときに get_ordered_selected_objects() で順序付きリスト取得
