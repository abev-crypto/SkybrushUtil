"""Sampling helpers with loop-tail support."""

from collections.abc import Iterable
from typing import Optional

import bpy
from bpy.types import Context


def each_frame_in(
    frames: Iterable[int], *, redraw: bool = False, context: Optional[Context] = None
) -> Iterable[tuple[int, float]]:
    """Yield frames and their time in seconds, appending the first two at the end."""

    assert context is not None

    scene = context.scene
    fps = scene.render.fps
    first_frames: list[int] = []

    for frame in frames:
        if len(first_frames) < 2:
            first_frames.append(frame)
        scene.frame_set(frame)
        if redraw:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=0)

        time = frame / fps
        yield frame, time

    if not first_frames:
        return

    # Repeat the first two frames to close the loop.
    for frame in first_frames:
        scene.frame_set(frame)
        if redraw:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=0)

        time = frame / fps
        yield frame, time
