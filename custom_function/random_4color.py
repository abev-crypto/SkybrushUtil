import random

# 4色のパレット (RGBA)
R1_COLOR = (1.0, 1.0, 1.0, 1.0)
R2_COLOR = (1.0, 1.0, 1.0, 1.0)
R3_COLOR = (1.0, 1.0, 1.0, 1.0)
R4_COLOR = (1.0, 1.0, 1.0, 1.0)

def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    """
    各ドローンに、4色の中から「固定の色」を割り当てて返す。
    frame や time_fraction に依存しないので、時間が進んでも色は変わらない。
    """
    # formation が複数ある場合も一意になるように seed を決める
    seed = drone_index + formation_index * 10000

    # グローバルな乱数状態を汚さないように、ローカル RNG を使う
    rng = random.Random(seed)

    # 4色の中から決定的に1色を選ぶ
    ca = [R1_COLOR, R2_COLOR, R3_COLOR, R4_COLOR]
    color = ca[rng.randrange(4)]
    return color
