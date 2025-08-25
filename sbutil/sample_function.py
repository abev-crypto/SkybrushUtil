SPEED = 2.0  # アニメーション速度
BASE_COLOR = (1.0, 0.0, 0.0, 1.0)  # ベースカラー(RGBA)

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    import math
    intensity = (math.sin(time_fraction * SPEED * math.tau) + 1.0) / 2.0
    return tuple(intensity * c for c in BASE_COLOR)
