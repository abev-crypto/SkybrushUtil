"""Monkey patches for Skybrush light effect result handling.

This module mirrors the updated behavior of Skybrush Studio where light effect
colors are baked into an image buffer. The patched functions only activate
when the Skybrush plug-in is available, keeping the rest of the add-on usable
on its own.
"""


import bpy
from bpy.types import Image
from typing import Iterator, Optional, TYPE_CHECKING
from bpy.app.handlers import persistent

from sbstudio.model import color as _colors_mod
from sbstudio.plugin.constants import Collections
from sbstudio.plugin.tasks import light_effects as _light_effects_mod
from sbstudio.plugin.utils.evaluator import get_position_of_object
from sbstudio.plugin.materials import get_led_light_color, set_led_light_color

from sbstudio.model.types import MutableRGBAColor, RGBAColor

def find_image_by_name(name: str) -> Image | None:
    """Searches for an image in bpy.data.images by its name.

    Args:
        name: The name of the image to find.

    Returns:
        the image object, or None if not found.
    """
    for img in bpy.data.images:
        if img.name == name:
            return img

__all__ = (
    "patch_light_effect_results",
    "unpatch_light_effect_results",
)

_ORIGINALS: dict[str, object] = {}


def _patched_get_color_of_drone(drone) -> RGBAColor:
    """Retrieve the color of a drone, reading from the result image when present."""

    color_from_image = _get_color_from_result_image(drone)
    if color_from_image is not None:
        return color_from_image

    if getattr(drone, "color", None) is not None:
        return drone.color  # type: ignore[return-value]

    return (0.0, 0.0, 0.0, 0.0)


def _get_color_from_result_image(drone) -> RGBAColor | None:
    scene = bpy.context.scene
    image = bpy.data.images.get("Light effects result")
    if image is None:
        return None

    width, height = image.size
    if width == 0 or height == 0:
        return None

    column = scene.frame_current - scene.frame_start
    if column < 0 or column >= width:
        return None

    drones = list(Collections.find_drones().objects)
    try:
        row = drones.index(drone)
    except ValueError:
        return None

    if row < 0 or row >= height:
        return None

    offset = (column + row * width) * 4
    return tuple(image.pixels[offset : offset + 4])  # type: ignore[return-value]


def _is_any_viewport_wireframe() -> bool:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return False

    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                shading = getattr(space, "shading", None)
                if shading and getattr(shading, "type", None) == 'WIREFRAME':
                    return True
    return False


def _patched_update_light_effects(scene, depsgraph):
    _suspension_counter = getattr(_light_effects_mod, "_suspension_counter", 0)
    if _suspension_counter > 0:
        return

    light_effects = scene.skybrush.light_effects
    if not light_effects:
        return

    frame = scene.frame_current
    frame_start = scene.frame_start

    drones = Collections.find_drones().objects if Collections is not None else []
    mapping = scene.skybrush.storyboard.get_mapping_at_frame(frame)
    height = len(mapping) if mapping is not None else len(drones)

    if not drones or height == 0:
        return

    changed = False
    colors: list[MutableRGBAColor] | None = None
    positions = None
    random_seq = scene.skybrush.settings.random_sequence_root

    for effect in light_effects.iter_active_effects_in_frame(frame):
        if colors is None:
            positions = [get_position_of_object(drone) for drone in drones]
            colors = [[0.0, 0.0, 0.0, 0.0] for _ in drones]
            changed = True

        effect.apply_on_colors(
            drones=drones,
            colors=colors,
            positions=positions,
            mapping=mapping,
            frame=frame,
            random_seq=random_seq,
        )
    if changed and colors is not None:
        _write_column_to_cache(frame - frame_start, colors)
        _write_led_color_attribute(colors)
    else:
        cached = _copy_previous_column_to_cache(frame - frame_start)
        if cached is not None:
            _write_led_color_attribute(cached)

_base_color_cache: dict[int, RGBAColor] = {}

_color_column_cache: dict[int, list[MutableRGBAColor]] = {}
"""Caches columns of colors by frame index to avoid touching the result image."""

_last_frame: Optional[int] = None
"""Number of the last frame that was evaluated with `update_light_effects()`"""

_suspension_counter: int = 0
"""Suspension counter. Dynamic light effect evaluation is suspended if this
counter is positive.
"""

WHITE: RGBAColor = (1, 1, 1, 1)
"""White color, used as a base color when no info is available for a newly added
drone.
"""

@persistent
def c_update_light_effects(scene):
    global _last_frame, _base_color_cache, _suspension_counter, WHITE

    if _is_any_viewport_wireframe():
        return

    if not getattr(scene, "sbutil_update_light_effects", True):
        return

    if getattr(scene, "sbutil_use_patched_light_effects", False):
        _patched_update_light_effects(scene, None)
        return

    if _suspension_counter > 0:
        return

    light_effects = scene.skybrush.light_effects
    if not light_effects:
        return

    random_seq = scene.skybrush.settings.random_sequence_root

    frame = scene.frame_current
    drones = None

    if _last_frame != frame:
        
        _last_frame = frame
        _base_color_cache.clear()

    changed = False

    for effect in light_effects.iter_active_effects_in_frame(frame):
        if drones is None:
            
            drones = Collections.find_drones().objects
            positions = [get_position_of_object(drone) for drone in drones]
            mapping = scene.skybrush.storyboard.get_mapping_at_frame(frame)
            if not _base_color_cache:
                
                
                colors: list[MutableRGBAColor] = []
                for drone in drones:
                    color = list(get_led_light_color(drone))
                    colors.append(color)
                    _base_color_cache[id(drone)] = color
            else:
                
                colors = [
                    _base_color_cache.get(id(drone)) or list(WHITE) for drone in drones
                ]

            changed = True

        effect.apply_on_colors(
            drones=drones,
            colors=colors,
            positions=positions,
            mapping=mapping,
            frame=frame,
            random_seq=random_seq,
        )

    
    
    
    
    
    if not changed:
        if _base_color_cache:
            drones = Collections.find_drones().objects
            colors = [
                _base_color_cache.get(id(drone)) or list(WHITE) for drone in drones
            ]
            _base_color_cache.clear()
            changed = True

    if changed:
        assert drones is not None
        for drone, color in zip(drones, colors):
            set_led_light_color(drone, color)


def _copy_previous_column_to_cache(column: int) -> list[MutableRGBAColor] | None:
    if column <= 0:
        return _color_column_cache.get(0)

    cached = _color_column_cache.get(column)
    if cached is not None:
        return cached

    prev = _color_column_cache.get(column - 1)
    if prev is None:
        return None

    copied = [list(color) for color in prev]
    _color_column_cache[column] = copied
    return copied


def _get_base_colors_for_frame(
    image, frame: int, frame_start: int, drones
) -> list[MutableRGBAColor]:
    column = frame - frame_start
    prev = _color_column_cache.get(column - 1)
    if prev is not None:
        return [list(color) for color in prev]

    return [list(_patched_get_color_of_drone(drone)) for drone in drones]

result_image: Image | None = None
stored_range: tuple[int, int] | None = None

def _get_or_create_result_image(width: int, height: int, render_range: tuple[int, int]):
    global result_image
    global stored_range
    global _color_column_cache

    image = result_image
    if image is None or image.name not in bpy.data.images:
        image = find_image_by_name("Light effects result") if find_image_by_name else None

    if (
        image is None
        or image.size[0] != width
        or image.size[1] != height
        or stored_range != render_range
    ):
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name="Light effects result", width=width, height=height)
        _color_column_cache.clear()

    result_image = image
    stored_range = render_range
    return image


def _write_column_to_cache(column: int, colors: list[MutableRGBAColor]) -> None:
    _color_column_cache[column] = [list(color) for color in colors]


def _write_led_color_attribute(colors: list[MutableRGBAColor]) -> None:
    system_obj = bpy.data.objects.get("DroneSystem")
    if system_obj is None or system_obj.type != 'MESH':
        return

    mesh = system_obj.data
    if mesh is None:
        return

    point_count = len(colors)

    if len(mesh.vertices) != point_count:
        mesh.clear_geometry()
        mesh.vertices.add(point_count)

    attr = mesh.color_attributes.get("LEDColor")
    if attr is None or attr.domain != 'POINT' or attr.data_type != 'BYTE_COLOR':
        if attr is not None:
            mesh.color_attributes.remove(attr)
        attr = mesh.color_attributes.new(
            name="LEDColor", domain='POINT', type='BYTE_COLOR'
        )

    for idx, color in enumerate(colors):
        if idx >= len(attr.data):
            break
        attr.data[idx].color = color

    mesh.update()


def _reregister_update_light_effects() -> None:
    bpy.app.handlers.depsgraph_update_post.remove(_light_effects_mod.update_light_effects)
    bpy.app.handlers.frame_change_post.remove(_light_effects_mod.update_light_effects)
    bpy.app.handlers.depsgraph_update_post.append(c_update_light_effects)
    bpy.app.handlers.frame_change_post.append(c_update_light_effects)
    print("Reregistered patched update_light_effects.")

def patch_light_effect_results():
    """Apply the monkey patches when Skybrush is available."""


    #_colors_mod.get_color_of_drone = _patched_get_color_of_drone

    
    _reregister_update_light_effects()
    #_light_effects_mod.update_light_effects = update_light_effects


def unpatch_light_effect_results():
    """Revert the monkey patches if they were applied."""

    if not _ORIGINALS:
        return

    _colors_mod.get_color_of_drone = _ORIGINALS.get(  # type: ignore[assignment]
        "get_color_of_drone", _colors_mod.get_color_of_drone
    )
    _light_effects_mod.update_light_effects = _ORIGINALS.get(
        "update_light_effects", _light_effects_mod.update_light_effects
    )

    _ORIGINALS.clear()



