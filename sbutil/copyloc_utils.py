def shape_copyloc_influence_curve(fcurve, handle_frames: float, key_filter=None) -> bool:
    """Shape Copy Location influence handles with absolute offsets per key.

    Every key's handles are pushed horizontally by ``handle_frames`` (clamped so
    they never cross neighbouring keys). Supports optional ``key_filter`` to
    limit which keys are updated. Returns True when any handles were changed.
    """

    keys = list(getattr(fcurve, "keyframe_points", []))
    if len(keys) < 2:
        return False

    keys.sort(key=lambda k: k.co.x)
    eps = 1e-4
    updated = False

    for idx, key in enumerate(keys):
        if key_filter and not key_filter(key):
            continue

        prev_x = keys[idx - 1].co.x if idx > 0 else None
        next_x = keys[idx + 1].co.x if idx < len(keys) - 1 else None

        key.interpolation = 'BEZIER'
        key.handle_left_type = 'FREE'
        key.handle_right_type = 'FREE'

        target_left = key.co.x - handle_frames
        target_right = key.co.x + handle_frames

        if prev_x is not None:
            min_left = prev_x + eps
            key.handle_left.x = max(target_left, min_left)
            key.handle_left.y = key.co.y
        else:
            key.handle_left.x = target_left
            key.handle_left.y = key.co.y

        if next_x is not None:
            max_right = next_x - eps
            key.handle_right.x = min(target_right, max_right)
            key.handle_right.y = key.co.y
        else:
            key.handle_right.x = target_right
            key.handle_right.y = key.co.y

        updated = True

    if updated:
        fcurve.keyframe_points.update()
    return updated
