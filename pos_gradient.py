FIRST_COLOR = (1.0, 1.0, 1.0, 1.0)
END_COLOR = (1.0, 1.0, 1.0, 1.0)
START_POS = (0.0, 0.0, 0.0)
END_POS = (0.0, 0.0, 0.0)
START_OFFSET = (0.0, 0.0, 0.0)
END_OFFSET = (0.0, 0.0, 0.0)


def color_function(frame, time_fraction, drone_index, formation_index, position, drone_count):
    """Return a color based on the position along a gradient."""
    offset = tuple(
        START_OFFSET[i] + (END_OFFSET[i] - START_OFFSET[i]) * time_fraction
        for i in range(3)
    )
    start = tuple(START_POS[i] + offset[i] for i in range(3))
    end = tuple(END_POS[i] + offset[i] for i in range(3))
    line = tuple(end[i] - start[i] for i in range(3))
    line_len_sq = sum(l * l for l in line)
    if line_len_sq > 0:
        t = sum((position[i] - start[i]) * line[i] for i in range(3)) / line_len_sq
        t = max(0.0, min(1.0, t))
    else:
        t = 0.0
    return tuple(
        FIRST_COLOR[i] + (END_COLOR[i] - FIRST_COLOR[i]) * t
        for i in range(4)
    )
