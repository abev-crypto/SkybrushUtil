"""Sampling helpers."""

from collections.abc import Iterable
from typing import Optional

import bpy
from bpy.types import Context


def each_frame_in(
    frames: Iterable[int], *, redraw: bool = False, context: Optional[Context] = None
) -> Iterable[tuple[int, float]]:
    """Yield frames and their time in seconds."""

    assert context is not None

    scene = context.scene
    fps = scene.render.fps
    for frame in frames:
        scene.frame_set(frame)
        if redraw:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=0)

        time = frame / fps
        yield frame, time
