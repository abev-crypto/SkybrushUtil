import math, colorsys

# 中心の想定位置（必要なら調整）
CENTER = (0.0, 0.0)
SPEED_TURNS_PER_DURATION = 0.6  # 1アニメーション内での回転数

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    x, y, z = position
    cx, cy = CENTER
    ang = math.atan2(y - cy, x - cx) / (2 * math.pi)  # -0.5〜0.5
    hue = (ang + time_fraction * SPEED_TURNS_PER_DURATION) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (r, g, b, 1.0)