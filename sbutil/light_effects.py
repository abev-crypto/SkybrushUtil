"""Light effect patches and UI extensions.

This module patches the light effect property group and its UI panel in the
`sbstudio` plug‑in.  The goal is to add loop configuration controls for
ColorRamp based effects while keeping the patch optional – when the original
plug‑in is not available, the module simply does nothing.
"""

import bpy
from bpy.props import (
    EnumProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
import importlib
from os.path import abspath, basename

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from operator import itemgetter
from typing import cast, Optional

from mathutils import Vector
from mathutils.bvhtree import BVHTree

from sbstudio.math.colors import blend_in_place, BlendMode
from sbstudio.math.rng import RandomSequence
from sbstudio.model.plane import Plane
from sbstudio.model.types import Coordinate3D, MutableRGBAColor
from sbstudio.plugin.constants import DEFAULT_LIGHT_EFFECT_DURATION
from sbstudio.plugin.meshes import use_b_mesh
from sbstudio.plugin.model.pixel_cache import PixelCache
from sbstudio.plugin.utils import remove_if_unused, with_context
from sbstudio.plugin.utils.collections import pick_unique_name
from sbstudio.plugin.utils.color_ramp import update_color_ramp_from
from sbstudio.plugin.utils.evaluator import get_position_of_object
from sbstudio.plugin.utils.image import convert_from_srgb_to_linear
from sbstudio.utils import constant, distance_sq_of, load_module, negate

from sbstudio.plugin.model.light_effects import OUTPUT_TYPE_TO_AXIS_SORT_KEY

from sbstudio.plugin.model.light_effects import LightEffect
from sbstudio.plugin.panels.light_effects import LightEffectsPanel

from sbstudio.plugin.model.light_effects import (
    effect_type_supports_randomization,
    output_type_supports_mapping_mode,
)
from sbstudio.plugin.operators import (
    CreateLightEffectOperator,
    DuplicateLightEffectOperator,
    MoveLightEffectDownOperator,
    MoveLightEffectUpOperator,
    RemoveLightEffectOperator,
)


# ---------------------------------------------------------------------------
# Property patching
# ---------------------------------------------------------------------------


class PosGradientProperties(bpy.types.PropertyGroup):  # pragma: no cover - Blender only
    first_color: FloatVectorProperty(
        name="First Color", size=4, subtype="COLOR", min=0.0, max=1.0, default=(1.0, 1.0, 1.0, 1.0)
    )
    end_color: FloatVectorProperty(
        name="End Color", size=4, subtype="COLOR", min=0.0, max=1.0, default=(1.0, 1.0, 1.0, 1.0)
    )
    start_pos: FloatVectorProperty(name="Start Pos", size=3, default=(0.0, 0.0, 0.0))
    end_pos: FloatVectorProperty(name="End Pos", size=3, default=(0.0, 0.0, 0.0))
    start_offset: FloatVectorProperty(name="Start Offset", size=3, default=(0.0, 0.0, 0.0))
    end_offset: FloatVectorProperty(name="End Offset", size=3, default=(0.0, 0.0, 0.0))


class SavePosGradientOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.save_pos_gradient"
    bl_label = "Embed Properties"
    bl_description = "Write PosGradient properties into the function file"

    filepath: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        props = entry.pos_gradient
        path = abspath(self.filepath)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:  # pragma: no cover - file I/O
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        def set_line(name, value):
            line = f"{name} = {tuple(value)}\n"
            for idx, existing in enumerate(lines):
                if existing.startswith(name + " ="):
                    lines[idx] = line
                    return
            lines.insert(0, line)

        set_line("FIRST_COLOR", props.first_color)
        set_line("END_COLOR", props.end_color)
        set_line("START_POS", props.start_pos)
        set_line("END_POS", props.end_pos)
        set_line("START_OFFSET", props.start_offset)
        set_line("END_OFFSET", props.end_offset)

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
        except OSError as exc:  # pragma: no cover - file I/O
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        return {"FINISHED"}

class PatchedLightEffect(PropertyGroup):
    type = EnumProperty(
        name="Effect Type",
        description=(
            "Type of the light effect: color ramp-based, image-based or"
            " custom function"
        ),
        items=[("COLOR_RAMP", "Color ramp", "", 1), ("IMAGE", "Image", "", 2), ("FUNCTION", "Function", "", 3)],
        default="COLOR_RAMP",
    )
    loop_count = IntProperty(name="Loop Count", default=0, min=0)
    loop_method = EnumProperty(
        name="Loop Method",
        items=[("FORWARD", "Forward", ""), ("REVERSE", "Reverse", ""), ("PINGPONG", "Ping-Pong", "")],
        default="FORWARD",
    )
    pos_gradient = PointerProperty(type=PosGradientProperties)
    def apply_on_colors(
        self,
        colors: Sequence[MutableRGBAColor],
        positions: Sequence[Coordinate3D],
        mapping: Optional[list[int]],
        *,
        frame: int,
        random_seq: RandomSequence,
    ) -> None:
        def get_output_based_on_output_type(output_type, mapping_mode, output_function):
            outputs = None
            common_output = None
            order = None
            if output_type == "FIRST_COLOR":
                common_output = 0.0
            elif output_type == "LAST_COLOR":
                common_output = 1.0
            elif output_type == "TEMPORAL":
                common_output = time_fraction
            elif output_type_supports_mapping_mode(output_type):
                proportional = mapping_mode == "PROPORTIONAL"
                if output_type == "DISTANCE":
                    if self.mesh:
                        position_of_mesh = get_position_of_object(self.mesh)
                        sort_key = lambda idx: distance_sq_of(positions[idx], position_of_mesh)
                    else:
                        sort_key = None
                else:
                    query_axes = (
                        OUTPUT_TYPE_TO_AXIS_SORT_KEY.get(output_type)
                        or OUTPUT_TYPE_TO_AXIS_SORT_KEY["default"]
                    )
                    if proportional:
                        sort_key = lambda idx: query_axes(positions[idx])[0]
                    else:
                        sort_key = lambda idx: query_axes(positions[idx])
                outputs = [1.0] * num_positions
                order = list(range(num_positions))
                if num_positions > 1:
                    if proportional and sort_key is not None:
                        evaluated_sort_keys = [sort_key(i) for i in order]
                        min_value, max_value = (min(evaluated_sort_keys), max(evaluated_sort_keys))
                        diff = max_value - min_value
                        if diff > 0:
                            outputs = [(value - min_value) / diff for value in evaluated_sort_keys]
                    else:
                        if sort_key is not None:
                            order.sort(key=sort_key)
                        assert outputs is not None
                        for u, v in enumerate(order):
                            outputs[v] = u / (num_positions - 1)
            elif output_type == "INDEXED_BY_DRONES":
                if num_positions > 1:
                    np_m1 = num_positions - 1
                    outputs = [i / np_m1 for i in range(num_positions)]
                else:
                    outputs = [0.0]
            elif output_type == "INDEXED_BY_FORMATION":
                if mapping is not None:
                    assert num_positions == len(mapping)
                    if None in mapping:
                        sorted_valid_mapping = sorted(
                            x for x in mapping if x is not None
                        )
                        np_m1 = max(len(sorted_valid_mapping) - 1, 1)
                        outputs = [
                            None if x is None else sorted_valid_mapping.index(x) / np_m1
                            for x in mapping
                        ]
                    else:
                        np_m1 = max(num_positions - 1, 1)
                        outputs = [None if x is None else x / np_m1 for x in mapping]
                else:
                    outputs = [None] * num_positions  
            elif output_type == "GROUP":
                outputs = [output_function(idx, 0, 0) for idx in range(num_positions)]
            elif output_type == "CUSTOM":
                absolute_path = abspath(output_function.path)
                module = load_module(absolute_path) if absolute_path else None
                if self.output_function.name:
                    fn = getattr(module, self.output_function.name)
                    outputs = [
                        fn(
                            frame=frame,
                            time_fraction=time_fraction,
                            drone_index=index,
                            formation_index=(
                                mapping[index] if mapping is not None else None
                            ),
                            position=positions[index],
                            drone_count=num_positions,
                        )
                        for index in range(num_positions)
                    ]
                else:
                    common_output = 1.0
            else:
                common_output = 1.0
            return outputs, common_output, order
        if not self.enabled or not self.contains_frame(frame):
            return
        time_fraction = (frame - self.frame_start) / max(self.duration - 1, 1)
        num_positions = len(positions)
        color_ramp = self.color_ramp
        color_image = self.color_image
        color_function_ref = self.color_function_ref
        new_color = [0.0] * 4
        outputs_x, common_output_x = get_output_based_on_output_type(
            self.output, self.output_mapping_mode, self.output_function
        )
        if color_image is not None:
            outputs_y, common_output_y = get_output_based_on_output_type(
                self.output_y, self.output_mapping_mode_y, self.output_function_y
            )
        condition = self._get_spatial_effect_predicate()
        for index, position in enumerate(positions):
            color = colors[index]
            if common_output_x is not None:
                output_x = common_output_x
            else:
                assert outputs_x is not None
                if outputs_x[index] is None:
                    continue
                output_x = outputs_x[index]
            assert isinstance(output_x, float)
            if color_image is not None:
                if common_output_y is not None:
                    output_y = common_output_y
                else:
                    assert outputs_y is not None
                    if outputs_y[index] is None:
                        continue
                    output_y = outputs_y[index]
                assert isinstance(output_y, float)
            if self.randomness != 0:
                offset_x = (random_seq.get_float(index) - 0.5) * self.randomness
                output_x = (offset_x + output_x) % 1.0
                if color_image is not None:
                    offset_y = (random_seq.get_float(index) - 0.5) * self.randomness
                    output_y = (offset_y + output_y) % 1.0
            alpha = max(
                min(self._evaluate_influence_at(position, frame, condition), 1.0), 0.0
            )
            if color_function_ref is not None:
                try:
                    new_color[:] = color_function_ref(
                        frame=frame,
                        time_fraction=time_fraction,
                        drone_index=index,
                        formation_index=(
                            mapping[index] if mapping is not None else None
                        ),
                        position=position,
                        drone_count=num_positions,
                    )
                except Exception as exc:
                    raise RuntimeError("ERROR_COLOR_FUNCTION") from exc
            elif color_image is not None:
                width, height = color_image.size
                pixels = self.get_image_pixels()
                x = int((width - 1) * output_x)
                y = int((height - 1) * output_y)
                offset = (x + y * width) * 4
                pixel_color = pixels[offset : offset + 4]
                if len(pixel_color) == len(new_color):
                    new_color[:] = convert_from_srgb_to_linear(pixel_color)
            elif color_ramp:
                loops = max(self.loop_count, 1)
                if loops > 1 or self.loop_method != "FORWARD":
                    t = output_x * loops
                    idx = min(int(t), loops - 1)
                    pos = t - idx
                    if self.loop_method == "PINGPONG":
                        forward = idx % 2 == 0
                    elif self.loop_method == "REVERSE":
                        forward = False
                    else:
                        forward = True
                    if not forward:
                        pos = 1.0 - pos
                    output_x = pos
                new_color[:] = color_ramp.evaluate(output_x)
            else:
                new_color[:] = (1.0, 1.0, 1.0, 1.0)
            new_color[3] *= alpha
            blend_in_place(new_color, color, BlendMode[self.blend_mode])

def patch_light_effect_class():
    """Inject loop properties into ``LightEffect`` using monkey patching."""
    if LightEffect is None:  # pragma: no cover - only runs inside Blender
        return
            
    LightEffect._original_type = getattr(LightEffect, "type", None)
    LightEffect._original_apply_on_colors = getattr(LightEffect, "apply_on_colors", None)
    LightEffect.type = PatchedLightEffect.type
    LightEffect.loop_count = PatchedLightEffect.loop_count
    LightEffect.loop_method = PatchedLightEffect.loop_method
    LightEffect.pos_gradient = PatchedLightEffect.pos_gradient
    LightEffect.apply_on_colors = PatchedLightEffect.apply_on_colors
    LightEffect.__annotations__["type"] = LightEffect.type
    LightEffect.__annotations__["loop_count"] = LightEffect.loop_count
    LightEffect.__annotations__["loop_method"] = LightEffect.loop_method
    LightEffect.__annotations__["pos_gradient"] = LightEffect.pos_gradient

def _unpatch_light_effect_class():
    if LightEffect is None or not hasattr(LightEffect, "_original_type"):  # pragma: no cover
        return
    bpy.utils.unregister_class(LightEffect)
    LightEffect.type = LightEffect._original_type
    LightEffect.__annotations__["type"] = LightEffect._original_type
    for attr in ("loop_count", "loop_method", "pos_gradient"):
        if hasattr(LightEffect, attr):
            delattr(LightEffect, attr)
            LightEffect.__annotations__.pop(attr, None)
    if getattr(LightEffect, "_original_apply_on_colors", None) is not None:
        LightEffect.apply_on_colors = LightEffect._original_apply_on_colors
        LightEffect._original_apply_on_colors = None
    LightEffect._original_type = None
    bpy.utils.register_class(LightEffect)
# UI patching
# ---------------------------------------------------------------------------


class BakeColorRampOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.bake_color_ramp"
    bl_label = "Bake Ramp"
    bl_description = "Bake looped ColorRamp into ramp and reset loop settings"

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry and entry.type == "COLOR_RAMP" and entry.texture

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        ramp = entry.texture.color_ramp
        loops = max(entry.loop_count, 1)
        src = [(e.position, list(e.color)) for e in ramp.elements]
        baked = []
        for i in range(loops):
            if entry.loop_method == "PINGPONG":
                forward = i % 2 == 0
            elif entry.loop_method == "REVERSE":
                forward = False
            else:
                forward = True
            for pos, col in src:
                new_pos = (i + (pos if forward else 1 - pos)) / loops
                baked.append((new_pos, col[:]))
        baked.sort(key=lambda x: x[0])
        elems = ramp.elements
        elems[0].position, elems[0].color = baked[0]
        while len(elems) > 1:
            elems.remove(elems[-1])
        for pos, col in baked[1:]:
            elem = elems.new(pos)
            elem.color = col
        entry.loop_count = 0
        entry.loop_method = "FORWARD"
        return {'FINISHED'}

class PatchedLightEffectsPanel(Panel):  # pragma: no cover - Blender UI code
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        light_effects = scene.skybrush.light_effects
        if not light_effects:
            return

        row = layout.row()

        col = row.column()
        col.template_list(
            "SKYBRUSH_UL_lightfxlist",
            "OBJECT_PT_skybrush_light_effects_panel",
            light_effects,
            "entries",
            light_effects,
            "active_entry_index",
        )

        col = row.column(align=True)
        col.operator(CreateLightEffectOperator.bl_idname, icon="ADD", text="")
        col.operator(RemoveLightEffectOperator.bl_idname, icon="REMOVE", text="")
        col.separator()
        col.operator(
            DuplicateLightEffectOperator.bl_idname, icon="DUPLICATE", text=""
        )
        col.separator()
        col.operator(MoveLightEffectUpOperator.bl_idname, icon="TRIA_UP", text="")
        col.operator(
            MoveLightEffectDownOperator.bl_idname, icon="TRIA_DOWN", text=""
        )

        entry = light_effects.active_entry
        if entry is not None:
            layout.prop(entry, "type")

            if entry.texture:
                if entry.type == "COLOR_RAMP":
                    row = layout.box()
                    row.template_color_ramp(entry.texture, "color_ramp")
                elif entry.type == "IMAGE":
                    row = layout.row()

                    col = row.column()
                    col.prop_search(entry.texture, "image", bpy.data, "images", text="")

                    col = row.column(align=True)
                    col.operator("image.open", icon="FILE_FOLDER", text="")
                elif entry.type == "FUNCTION":
                    box = self.layout.box()
                    box.prop(entry.color_function, "path", text="")
                    box.prop(entry.color_function, "name", text="")
                    if basename(abspath(entry.color_function.path)) == "pos_gradient.py":
                        pg = entry.pos_gradient
                        box.prop(pg, "first_color")
                        box.prop(pg, "end_color")
                        box.prop(pg, "start_pos")
                        box.prop(pg, "end_pos")
                        box.prop(pg, "start_offset")
                        box.prop(pg, "end_offset")
                        op = box.operator(SavePosGradientOperator.bl_idname, text="Embed Properties")
                        op.filepath = entry.color_function.path
                else:
                    row = layout.box()
                    row.alert = True
                    row.label(text="Invalid light effect type", icon="ERROR")
                    layout.separator()

            col = layout.column()
            col.prop(entry, "frame_start")
            col.prop(entry, "duration")
            col.prop(entry, "frame_end")
            col.separator()
            col.prop(entry, "fade_in_duration")
            col.prop(entry, "fade_out_duration")
            col.separator()
            col.prop(entry, "mesh")
            col.separator()
            if entry.type == "COLOR_RAMP":
                col.prop(entry, "loop_count")
                col.prop(entry, "loop_method")
                col.operator(BakeColorRampOperator.bl_idname, text="Bake Ramp")
            if entry.type == "COLOR_RAMP" or entry.type == "IMAGE":
                col.prop(entry, "output")
                if entry.output == "CUSTOM":
                    col.prop(entry.output_function, "path", text="Fn file")
                    col.prop(entry.output_function, "name", text="Fn name")
            if output_type_supports_mapping_mode(entry.output):
                col.prop(entry, "output_mapping_mode")
            if entry.type == "IMAGE":
                col.prop(entry, "output_y")
                if entry.output_y == "CUSTOM":
                    col.prop(entry.output_function_y, "path", text="Fn file")
                    col.prop(entry.output_function_y, "name", text="Fn name")
                if output_type_supports_mapping_mode(entry.output_y):
                    col.prop(entry, "output_mapping_mode_y")
            col.prop(entry, "target")
            col.prop(entry, "invert_target")
            col.prop(entry, "blend_mode")
            col.prop(entry, "influence", slider=True)

            if effect_type_supports_randomization(entry.type):
                col.prop(entry, "randomness", slider=True)

def patch_light_effects_panel():
    LightEffectsPanel._original_draw = LightEffectsPanel.draw
    LightEffectsPanel.draw = PatchedLightEffectsPanel.draw


def _unpatch_light_effects_panel():
    LightEffectsPanel.draw = LightEffectsPanel._original_draw
    LightEffectsPanel._original_draw = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register():  # pragma: no cover - executed in Blender
    bpy.utils.register_class(BakeColorRampOperator)
    bpy.utils.register_class(PosGradientProperties)
    bpy.utils.register_class(SavePosGradientOperator)


def unregister():  # pragma: no cover - executed in Blender
    _unpatch_light_effects_panel()
    _unpatch_light_effect_class()
    bpy.utils.unregister_class(SavePosGradientOperator)
    bpy.utils.unregister_class(PosGradientProperties)
    bpy.utils.unregister_class(BakeColorRampOperator)

