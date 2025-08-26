import math

def _frac(x): return x - math.floor(x)
def _hash01(x): return _frac(math.sin(x) * 43758.5453)

BASE = (0.20, 0.22, 0.5)

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    x, y, z = position
    # 各ドローンの固有シード
    seed = _hash01(drone_index * 1.618 + x * 0.11 + y * 0.17 + z * 0.13)
    freq = 0.8 + 2.2 * seed
    phase = seed * 6.28318530718

    tw = 0.5 + 0.5 * math.sin(2 * math.pi * time_fraction * freq + phase)  # 0〜1

    # たまに強いスパーク
    spark = 1.0 if (tw > 0.95 and _hash01(drone_index * 9.13 + frame * 0.07) > 0.85) else 0.0

    r = min(1.0, BASE[0] + 0.8 * tw + 0.25 * spark)
    g = min(1.0, BASE[1] + 0.7 * tw + 0.25 * spark)
    b = min(1.0, BASE[2] + 0.9 * tw + 0.35 * spark)
    a = min(1.0, 0.45 + 0.55 * tw + 0.4 * spark)
    return (r, g, b, a)
