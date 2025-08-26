import math

PLANE_AMPLITUDE = 6.0   # スキャン面の上下振幅（m）
PLANE_THICKNESS = 1.6   # 面の厚み（m）
CYCLES_PER_DURATION = 1.0
BASE = (0.05, 0.08, 0.15)  # ベース色（暗めの青）

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    x, y, z = position
    # 走査面の高さ（-AMPLITUDE〜+AMPLITUDEを往復）
    z0 = math.sin(2 * math.pi * time_fraction * CYCLES_PER_DURATION) * PLANE_AMPLITUDE

    d = abs(z - z0)
    u = max(0.0, 1.0 - d / PLANE_THICKNESS)  # 面に近いほど1
    u = u * u

    r = BASE[0] + u * (1.0 - BASE[0])
    g = BASE[1] + u * (1.0 - BASE[1])
    b = BASE[2] + u * (1.0 - BASE[2])
    a = 0.4 + 0.6 * u
    return (r, g, b, a)
