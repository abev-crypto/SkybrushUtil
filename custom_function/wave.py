import math, colorsys

WAVELENGTH = 3.2    # 波長（m）
SPEED = 1.0         # 1アニメーションあたりの進行量
SAT = 1.0

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    x, y, z = position
    d = math.sqrt(x*x + y*y + z*z)   # 原点からの距離（m）

    phase = (d / WAVELENGTH - time_fraction * SPEED) % 1.0  # 0〜1
    # 明るさは位相の山で最大
    bright = 0.5 + 0.5 * math.cos(2 * math.pi * phase)      # 0〜1
    hue = (1.0 - phase) % 1.0                                # 色も位相で回す

    r, g, b = colorsys.hsv_to_rgb(hue, SAT, bright)
    return (r, g, b, 1.0)
