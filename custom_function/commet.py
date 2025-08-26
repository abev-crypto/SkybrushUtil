import math

TAIL_FRACTION = 0.18   # 全体に対して尾の長さ（割合）
REV_PER_DURATION = 1.2 # 1アニメーションで何周するか
HEAD_COLOR = (1.0, 1.0, 1.0)
TAIL_COLOR = (0.25, 0.55, 1.0)

def _lerp(a, b, t): return a + (b - a) * t

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    n = max(1, drone_count)
    head = (time_fraction * REV_PER_DURATION % 1.0) * n

    # 循環距離（0〜n/2）
    d1 = (drone_index - head) % n
    d2 = (head - drone_index) % n
    d = min(d1, d2)

    u = max(0.0, 1.0 - d / (TAIL_FRACTION * n))  # 0〜1
    u = u * u  # 尾の先端を滑らかに

    r = _lerp(TAIL_COLOR[0], HEAD_COLOR[0], u)
    g = _lerp(TAIL_COLOR[1], HEAD_COLOR[1], u)
    b = _lerp(TAIL_COLOR[2], HEAD_COLOR[2], u)
    a = 0.2 + 0.8 * u
    return (r, g, b, a)
