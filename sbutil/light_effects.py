"""Light effect patches and UI extensions.

This module patches the light effect property group and its UI panel in the
`sbstudio` plug‑in.  The goal is to add loop configuration controls for
ColorRamp based effects while keeping the patch optional – when the original
plug‑in is not available, the module simply does nothing.
"""

import bpy
import bmesh
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from os.path import abspath, basename, join

from bpy.app.handlers import persistent

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from operator import itemgetter
from typing import cast, Optional
import hashlib
import json
from uuid import uuid4

from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from sbstudio.math.colors import blend_in_place, BlendMode
from sbstudio.math.rng import RandomSequence
from sbstudio.model.plane import Plane
from sbstudio.model.types import Coordinate3D, MutableRGBAColor
from sbstudio.plugin.constants import DEFAULT_LIGHT_EFFECT_DURATION
try:  # pragma: no cover - optional dependency
    from sbstudio.plugin.meshes import use_b_mesh as _use_b_mesh
except Exception:  # pragma: no cover - fallback when plugin missing
    _use_b_mesh = None
from sbstudio.plugin.model.pixel_cache import PixelCache
from sbstudio.plugin.utils import remove_if_unused, with_context
from sbstudio.plugin.utils.collections import pick_unique_name
from sbstudio.plugin.utils.color_ramp import update_color_ramp_from
from sbstudio.plugin.utils.evaluator import get_position_of_object
from sbstudio.plugin.utils.image import convert_from_srgb_to_linear
from sbstudio.utils import constant, distance_sq_of, load_module, negate
from types import ModuleType
from contextlib import contextmanager

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
from sbutil import path_gradient

@contextmanager
def use_b_mesh(mesh):  # pragma: no cover - Blender integration
    """Return a ``BMesh`` for ``mesh`` regardless of plug-in version."""
    if _use_b_mesh is not None:
        try:  # new signature accepting the mesh directly
            with _use_b_mesh(mesh) as bm:
                yield bm
            return
        except TypeError:
            try:  # original helper provided only a bare BMesh
                with _use_b_mesh() as bm:
                    bm.from_mesh(mesh)
                    yield bm
                    bm.to_mesh(mesh)
                return
            except Exception:
                pass
        except Exception:
            pass
    bm = bmesh.new()
    bm.from_mesh(mesh)
    try:
        yield bm
        bm.to_mesh(mesh)
    finally:
        bm.free()

_state: dict[int, dict] = {}


def _as_dict(value):
    """Convert ID property containers to plain Python dictionaries."""
    if hasattr(value, "items"):
        return {k: _as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_dict(v) for v in value]
    return value


def get_state(pg) -> dict:
    st = _state.setdefault(pg.as_pointer(), {})
    if "config_schema" not in st and "_config_schema" in pg:
        st["config_schema"] = _as_dict(pg["_config_schema"])
    if "text_hash" not in st and "_text_hash" in pg:
        st["text_hash"] = pg["_text_hash"]
    if "absolute_path" not in st and "_absolute_path" in pg:
        st["absolute_path"] = pg["_absolute_path"]
    return st


def _barycentric_weights(point: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float]:
    """Return barycentric weights of ``point`` on triangle ``(a, b, c)``."""

    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def _get_color_attribute(mesh) -> Optional[object]:
    """Return the active vertex color attribute from ``mesh`` if present."""

    layer = None
    if hasattr(mesh, "color_attributes"):
        layer = mesh.color_attributes.get("color")
        if layer is None:
            try:
                layer = mesh.color_attributes.active_color
            except AttributeError:
                layer = None
    if layer is None and hasattr(mesh, "vertex_colors"):
        vcols = mesh.vertex_colors
        if vcols:
            layer = vcols.get("color") or getattr(vcols, "active", None)
    return layer


def sample_vertex_color_factors(mesh_obj, positions: Sequence[Coordinate3D]) -> Optional[list[Optional[float]]]:
    """Sample vertex color values from ``mesh_obj`` at ``positions``."""

    if mesh_obj is None or not positions:
        return None

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=True)
    try:
        color_layer = _get_color_attribute(eval_mesh)
        if color_layer is None:
            return None
        domain = getattr(color_layer, "domain", None)
        if domain not in {"POINT", "CORNER"}:
            return None
        eval_mesh.calc_loop_triangles()
        if not eval_mesh.loop_triangles:
            return None
        tree = BVHTree.FromMesh(eval_mesh, epsilon=0.0)
        if tree is None:
            return None
        inv_world = mesh_obj.matrix_world.inverted()
        outputs: list[Optional[float]] = []
        for pos in positions:
            co = Vector(pos) if not isinstance(pos, Vector) else pos.copy()
            local = inv_world @ co
            nearest = tree.find_nearest(local)
            if nearest is None:
                outputs.append(None)
                continue
            location, _normal, tri_index, _dist = nearest
            if tri_index is None:
                outputs.append(None)
                continue
            tri = eval_mesh.loop_triangles[tri_index]
            verts = tri.vertices
            loops = tri.loops
            v0 = eval_mesh.vertices[verts[0]].co
            v1 = eval_mesh.vertices[verts[1]].co
            v2 = eval_mesh.vertices[verts[2]].co
            w0, w1, w2 = _barycentric_weights(location, v0, v1, v2)
            if domain == "POINT":
                c0 = color_layer.data[verts[0]].color
                c1 = color_layer.data[verts[1]].color
                c2 = color_layer.data[verts[2]].color
            else:  # CORNER
                c0 = color_layer.data[loops[0]].color
                c1 = color_layer.data[loops[1]].color
                c2 = color_layer.data[loops[2]].color
            value = (c0[0] * w0 + c1[0] * w1 + c2[0] * w2) % 1.0
            outputs.append(float(value))
        return outputs
    finally:
        eval_obj.to_mesh_clear()
def _get_object_property(obj, prop_type):
    """Return dynamic property of ``obj`` based on ``prop_type``."""
    if prop_type == "POS":
        return tuple(get_position_of_object(obj))
    if prop_type == "ROT":
        return tuple(getattr(obj, "rotation_euler", (0.0, 0.0, 0.0)))
    if prop_type == "SCL":
        return tuple(getattr(obj, "scale", (1.0, 1.0, 1.0)))
    if prop_type == "MAT":
        mat = None
        if getattr(obj, "material_slots", None):
            slot = obj.material_slots[0]
            mat = getattr(slot, "material", None)
        if mat is None:
            mat = getattr(obj, "active_material", None)
        if mat is None:
            return None
        color = None
        if getattr(mat, "use_nodes", False) and getattr(mat, "node_tree", None):
            node = next(
                (n for n in mat.node_tree.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"),
                None,
            )
            if node is not None:
                inp = node.inputs.get("Base Color")
                if inp is not None:
                    color = tuple(inp.default_value[:4])
        if color is None:
            diff = getattr(mat, "diffuse_color", None)
            if diff is not None:
                color = tuple(diff[:4])
        return color
    return None


def initialize_color_function(pg) -> None:
    """Load the color function module and create config schema.

    Constants defined in the module are exposed as configuration variables.
    Constants whose lowercase names clash with existing RNA properties of the
    light effect (e.g. ``name`` or ``type``) are ignored to avoid overwriting
    those properties; these names are therefore reserved for Blender.
    """
    if pg.type != "FUNCTION":
        return

    st = get_state(pg)

    # Cache existing dynamic property values so they can be restored later if
    # the state needs to be reset and the new schema still contains them.
    cached_values = {
        name: _as_dict(pg[name])
        for name in pg.get("_config_schema", {})
        if name in pg
    }
    brna = getattr(pg, "bl_rna", None)
    reserved = {p.identifier.lower() for p in getattr(brna, "properties", [])}

    def reset_state():
        for an in list(pg.get("_config_schema", {})):
            if an in pg:
                del pg[an]
        if "_config_schema" in pg:
            del pg["_config_schema"]
        st.clear()

    if getattr(pg, "color_function_text", None):
        text = pg.color_function_text
        source = text.as_string()
        text_hash = hashlib.sha256(source.encode()).hexdigest()
        if text_hash != st.get("text_hash"):
            reset_state()
            st["text_hash"] = text_hash
            pg["_text_hash"] = text_hash
        if "module" not in st:
            module = ModuleType(text.name)
            exec(source, module.__dict__)
            st["module"] = module
        module = st["module"]
        if "config_schema" not in st:
            schema: dict[str, dict] = {}
            for name in dir(module):
                if not name.isupper():
                    continue
                attr_name = name.lower()
                if attr_name in reserved:
                    continue
                value = getattr(module, name)
                if isinstance(value, (int, float)):
                    if attr_name not in pg:
                        pg[attr_name] = value
                    schema[attr_name] = {"default": value}
                elif (
                    isinstance(value, (tuple, list))
                    and all(isinstance(v, (int, float)) for v in value)
                ):
                    list_value = list(value)
                    if attr_name not in pg:
                        pg[attr_name] = list_value
                    meta = {"default": list_value}
                    if name.endswith("_COLOR"):
                        # Suffix indicates that the value should be shown as a color
                        meta["subtype"] = "COLOR"
                    elif name.endswith("_POS"):
                        meta["subtype"] = "XYZ"
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            pg[obj_attr_name] = ""
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                        meta["object_ref_type"] = "POS"
                    elif name.endswith("_ROT"):
                        meta["subtype"] = "EULER"
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            pg[obj_attr_name] = ""
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                        meta["object_ref_type"] = "ROT"
                    elif name.endswith("_SCL"):
                        meta["subtype"] = "XYZ"
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            pg[obj_attr_name] = ""
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                        meta["object_ref_type"] = "SCL"
                    elif name.endswith("_MAT"):
                        meta["subtype"] = "COLOR"
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            pg[obj_attr_name] = ""
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                        meta["object_ref_type"] = "MAT"
                    schema[attr_name] = meta
            st["config_schema"] = schema
            pg["_config_schema"] = schema
            # Restore cached property values when possible, but only update
            # when the value actually differs from the current one.
            for name, value in cached_values.items():
                if name in schema and pg.get(name) != value:
                    pg[name] = value
        return

    if not pg.color_function or not pg.color_function.path:
        return

    ap = abspath(pg.color_function.path)
    if not ap.lower().endswith(".py"):
        reset_state()
        return
    if ap != st.get("absolute_path", ""):
        reset_state()
        st["absolute_path"] = ap
        pg["_absolute_path"] = ap
    if "module" not in st:
        st["module"] = load_module(ap)
    module = st["module"]
    if "config_schema" not in st:
        schema: dict[str, dict] = {}
        for name in dir(module):
            if not name.isupper():
                continue
            attr_name = name.lower()
            if attr_name in reserved:
                continue
            value = getattr(module, name)
            if isinstance(value, (int, float)):
                if attr_name not in pg:
                    pg[attr_name] = value
                schema[attr_name] = {"default": value}
            elif (
                isinstance(value, (tuple, list))
                and all(isinstance(v, (int, float)) for v in value)
            ):
                list_value = list(value)
                if attr_name not in pg:
                    pg[attr_name] = list_value
                meta = {"default": list_value}
                if name.endswith("_COLOR"):
                    # Suffix indicates that the value should be shown as a color
                    meta["subtype"] = "COLOR"
                elif name.endswith("_POS"):
                    meta["subtype"] = "XYZ"
                    obj_attr_name = f"{attr_name}_object"
                    if obj_attr_name not in pg:
                        pg[obj_attr_name] = ""
                    schema[obj_attr_name] = {
                        "default": "",
                        "type": "OBJECT",
                    }
                    meta["object_ref_attr"] = obj_attr_name
                    meta["object_ref_type"] = "POS"
                elif name.endswith("_ROT"):
                    meta["subtype"] = "EULER"
                    obj_attr_name = f"{attr_name}_object"
                    if obj_attr_name not in pg:
                        pg[obj_attr_name] = ""
                    schema[obj_attr_name] = {
                        "default": "",
                        "type": "OBJECT",
                    }
                    meta["object_ref_attr"] = obj_attr_name
                    meta["object_ref_type"] = "ROT"
                elif name.endswith("_SCL"):
                    meta["subtype"] = "XYZ"
                    obj_attr_name = f"{attr_name}_object"
                    if obj_attr_name not in pg:
                        pg[obj_attr_name] = ""
                    schema[obj_attr_name] = {
                        "default": "",
                        "type": "OBJECT",
                    }
                    meta["object_ref_attr"] = obj_attr_name
                    meta["object_ref_type"] = "SCL"
                elif name.endswith("_MAT"):
                    meta["subtype"] = "COLOR"
                    obj_attr_name = f"{attr_name}_object"
                    if obj_attr_name not in pg:
                        pg[obj_attr_name] = ""
                    schema[obj_attr_name] = {
                        "default": "",
                        "type": "OBJECT",
                    }
                    meta["object_ref_attr"] = obj_attr_name
                    meta["object_ref_type"] = "MAT"
                schema[attr_name] = meta
        st["config_schema"] = schema
        pg["_config_schema"] = schema
        for name, value in cached_values.items():
            if name in schema and pg.get(name) != value:
                pg[name] = value


def ensure_color_function_initialized(pg) -> None:
    if pg.type == "FUNCTION":
        initialize_color_function(pg)


def ensure_all_function_entries_initialized() -> None:
    for scene in bpy.data.scenes:
        le_group = getattr(scene.skybrush, "light_effects", None)
        if not le_group:
            continue
        for entry in getattr(le_group, "entries", []):
            ensure_color_function_initialized(entry)


@persistent
def _ensure_light_effects_initialized(_dummy=None):  # pragma: no cover - Blender UI
    ensure_all_function_entries_initialized()

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
    use_inside_mesh_vertex_colors = BoolProperty(
        name="Sample Position from Mesh Colors",
        description=(
            "When targeting drones inside a mesh, use the mesh vertex colors as"
            " the sampling position for the Color Ramp"
        ),
        default=False,
    )
    color_function_text = PointerProperty(
        name="Color Function Text", type=bpy.types.Text
    )
    @property
    def color_function_ref(self) -> Optional[Callable]:
        if self.type != "FUNCTION":
            return None
        function_name = getattr(getattr(self, "color_function", None), "name", None)
        if not function_name:
            return None
        st = get_state(self)
        module = st.get("module")
        if module is None:
            initialize_color_function(self)
            module = st.get("module")
        if module is None:
            return None
        return getattr(module, function_name, None)

    def draw_color_function_config(self, layout) -> None:
        """Draw UI controls for dynamically discovered config variables."""
        st = get_state(self)
        schema = st.get("config_schema", {})
        if not schema:
            return
        ensure_schema(self, schema)
        draw_dynamic(self, layout, schema)

    def _get_spatial_effect_predicate(self):
        if self.target == "COLLECTION" and getattr(self, "target_collection", None):
            positions = {
                tuple(get_position_of_object(obj))
                for obj in self.target_collection.objects
            }
            if self.invert_target:
                return lambda pos: tuple(pos) not in positions
            return lambda pos: tuple(pos) in positions
        original = getattr(LightEffect, "_original_get_spatial_effect_predicate", None)
        if original is not None:
            return original(self)
        return constant(True)

    def apply_on_colors(
        self,
        colors: Sequence[MutableRGBAColor],
        positions: Sequence[Coordinate3D],
        mapping: Optional[list[int]],
        *,
        frame: int,
        random_seq: RandomSequence,
    ) -> None:
        ensure_color_function_initialized(self)
        def get_output_based_on_output_type(
            output_type: str,
            mapping_mode: str,
            output_function,
        ) -> tuple[Optional[list[Optional[float]]], Optional[float]]:
            
            outputs: Optional[list[Optional[float]]] = None
            common_output: Optional[float] = None
            order: Optional[list[int]] = None

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
                module = get_state(self).get("module")
                if module is None:
                    path = output_function.path
                    module = (
                        load_module(abspath(path))
                        if path and path.lower().endswith(".py")
                        else None
                    )
                if self.output_function.name and module:
                    fn_name = self.output_function.name
                    if getattr(self, "color_function_text", None):
                        ctx = module.__dict__
                        outputs = []
                        for index in range(num_positions):
                            formation_index = (
                                mapping[index] if mapping is not None else None
                            )
                            ctx.update(
                                frame=frame,
                                time_fraction=time_fraction,
                                drone_index=index,
                                formation_index=formation_index,
                                position=positions[index],
                                drone_count=num_positions,
                            )
                            outputs.append(
                                eval(
                                    f"{fn_name}(frame=frame, time_fraction=time_fraction, drone_index=drone_index, formation_index=formation_index, position=position, drone_count=drone_count)",
                                    ctx,
                                )
                            )
                    else:
                        fn = getattr(module, fn_name)
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
            return outputs, common_output
        if not self.enabled or not self.contains_frame(frame):
            return
        time_fraction = (frame - self.frame_start) / max(self.duration - 1, 1)
        num_positions = len(positions)
        color_ramp = self.color_ramp
        color_image = self.color_image
        color_function_ref = self.color_function_ref
        function_name = getattr(getattr(self, "color_function", None), "name", "")
        if (
            self.type == "FUNCTION"
            and not getattr(self, "color_function_text", None)
            and not getattr(getattr(self, "color_function", None), "name", "")
        ):
            return
        st = get_state(self)
        module = st.get("module", None)
        if module is not None:
            schema = st.get("config_schema", {})
            if getattr(self, "color_function_text", None):
                ctx = module.__dict__
                for name, meta in schema.items():
                    if meta.get("type") == "OBJECT":
                        continue
                    value = self.get(name)
                    obj_attr = meta.get("object_ref_attr")
                    if obj_attr:
                        obj_name = self.get(obj_attr)
                        if obj_name:
                            obj = bpy.data.objects.get(obj_name)
                            if obj is not None:
                                obj_val = _get_object_property(
                                    obj, meta.get("object_ref_type", "POS")
                                )
                                if obj_val is not None:
                                    value = obj_val
                    if isinstance(value, Iterable):
                        ctx[name.upper()] = tuple(value)
                    else:
                        ctx[name.upper()] = value
            elif color_function_ref is not None:
                for name, meta in schema.items():
                    if meta.get("type") == "OBJECT":
                        continue
                    value = self.get(name)
                    obj_attr = meta.get("object_ref_attr")
                    if obj_attr:
                        obj_name = self.get(obj_attr)
                        if obj_name:
                            obj = bpy.data.objects.get(obj_name)
                            if obj is not None:
                                obj_val = _get_object_property(
                                    obj, meta.get("object_ref_type", "POS")
                                )
                                if obj_val is not None:
                                    value = obj_val
                    if isinstance(value, Iterable):
                        setattr(module, name.upper(), tuple(value))
                    else:
                        setattr(module, name.upper(), value)
        new_color = [0.0] * 4
        mesh_outputs = None
        if (
            self.type == "COLOR_RAMP"
            and self.target == "INSIDE_MESH"
            and getattr(self, "use_inside_mesh_vertex_colors", False)
            and getattr(self, "mesh", None)
        ):
            mesh_outputs = sample_vertex_color_factors(self.mesh, positions)
        outputs_x, common_output_x = get_output_based_on_output_type(
            self.output, self.output_mapping_mode, self.output_function
        )
        if mesh_outputs is not None:
            if common_output_x is not None:
                outputs_x = [common_output_x] * num_positions
                common_output_x = None
            if outputs_x is None:
                outputs_x = mesh_outputs
            else:
                for i, value in enumerate(mesh_outputs):
                    if value is not None:
                        outputs_x[i] = value
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
            if getattr(self, "color_function_text", None):
                ctx = st.get("module", ModuleType("_dummy")).__dict__
                ctx.update(
                    frame=frame,
                    time_fraction=time_fraction,
                    drone_index=index,
                    formation_index=(mapping[index] if mapping is not None else None),
                    position=position,
                    drone_count=num_positions,
                )
                try:
                    new_color[:] = eval(
                        f"color_function(frame=frame, time_fraction=time_fraction, drone_index=drone_index, formation_index=formation_index, position=position, drone_count=drone_count)",
                        ctx,
                    )
                except Exception as exc:
                    raise RuntimeError("ERROR_COLOR_FUNCTION") from exc
            elif color_function_ref is not None:
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
    LightEffect._original_color_function_ref = getattr(
        LightEffect, "color_function_ref", None
    )
    LightEffect._original_draw_color_function_config = getattr(
        LightEffect, "draw_color_function_config", None
    )
    LightEffect._original_target = getattr(LightEffect, "target", None)
    LightEffect._original_get_spatial_effect_predicate = getattr(
        LightEffect, "_get_spatial_effect_predicate", None
    )
    LightEffect.type = PatchedLightEffect.type
    LightEffect.loop_count = PatchedLightEffect.loop_count
    LightEffect.loop_method = PatchedLightEffect.loop_method
    LightEffect.use_inside_mesh_vertex_colors = (
        PatchedLightEffect.use_inside_mesh_vertex_colors
    )
    LightEffect.color_function_text = PatchedLightEffect.color_function_text
    LightEffect.apply_on_colors = PatchedLightEffect.apply_on_colors
    LightEffect.color_function_ref = PatchedLightEffect.color_function_ref
    LightEffect.draw_color_function_config = (
        PatchedLightEffect.draw_color_function_config
    )
    LightEffect.target = EnumProperty(
        name="Target",
        description=(
            "Specifies whether to apply this light effect to all drones or only"
            " to those drones that are inside the given mesh or are in front of"
            " the plane of the first face of the mesh. See also the 'Invert'"
            " property"
        ),
        items=[
            ("ALL", "All drones", "", 1),
            ("INSIDE_MESH", "Inside the mesh", "", 2),
            ("OUTSIDE_MESH", "Outside mesh", "", 3),
            ("COLLECTION", "Collection", "", 4),
        ],
        default="ALL",
    )
    LightEffect.target_collection = PointerProperty(
        name="Collection", type=bpy.types.Collection
        )
    LightEffect._get_spatial_effect_predicate = (
        PatchedLightEffect._get_spatial_effect_predicate
    )
    LightEffect.__annotations__["type"] = LightEffect.type
    LightEffect.__annotations__["loop_count"] = LightEffect.loop_count
    LightEffect.__annotations__["loop_method"] = LightEffect.loop_method
    LightEffect.__annotations__["use_inside_mesh_vertex_colors"] = (
        LightEffect.use_inside_mesh_vertex_colors
    )
    LightEffect.__annotations__["color_function_text"] = LightEffect.color_function_text
    LightEffect.__annotations__["target"] = LightEffect.target
    LightEffect.__annotations__["target_collection"] = LightEffect.target_collection
    ensure_all_function_entries_initialized()

def unpatch_light_effect_class():
    if LightEffect is None or not hasattr(LightEffect, "_original_type"):  # pragma: no cover
        return
    bpy.utils.unregister_class(LightEffect)
    LightEffect.type = LightEffect._original_type
    LightEffect.__annotations__["type"] = LightEffect._original_type
    for attr in (
        "loop_count",
        "loop_method",
        "use_inside_mesh_vertex_colors",
        "color_function_text",
    ):
        if hasattr(LightEffect, attr):
            delattr(LightEffect, attr)
            LightEffect.__annotations__.pop(attr, None)
    if getattr(LightEffect, "_original_apply_on_colors", None) is not None:
        LightEffect.apply_on_colors = LightEffect._original_apply_on_colors
        LightEffect._original_apply_on_colors = None
    if getattr(LightEffect, "_original_color_function_ref", None) is not None:
        LightEffect.color_function_ref = LightEffect._original_color_function_ref
        LightEffect._original_color_function_ref = None
    if getattr(LightEffect, "_original_draw_color_function_config", None) is not None:
        LightEffect.draw_color_function_config = (
            LightEffect._original_draw_color_function_config
        )
        LightEffect._original_draw_color_function_config = None
    elif hasattr(LightEffect, "draw_color_function_config"):
        delattr(LightEffect, "draw_color_function_config")
    if getattr(LightEffect, "_original_target", None) is not None:
        LightEffect.target = LightEffect._original_target
        LightEffect.__annotations__["target"] = LightEffect._original_target
        LightEffect._original_target = None
    if getattr(LightEffect, "_original_get_spatial_effect_predicate", None) is not None:
        LightEffect._get_spatial_effect_predicate = (
            LightEffect._original_get_spatial_effect_predicate
        )
        LightEffect._original_get_spatial_effect_predicate = None
    elif hasattr(LightEffect, "_get_spatial_effect_predicate"):
        delattr(LightEffect, "_get_spatial_effect_predicate")
    if hasattr(LightEffect, "target_collection"):
        delattr(LightEffect, "target_collection")
        LightEffect.__annotations__.pop("target_collection", None)
    LightEffect._original_type = None
    bpy.utils.register_class(LightEffect)
# UI patching
# ---------------------------------------------------------------------------


class EmbedColorFunctionOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.embed_color_function"
    bl_label = "Embed"
    bl_description = "Embed color function file into a text datablock"

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return (
            entry
            and entry.type == "FUNCTION"
            and bool(entry.color_function.path)
        )

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        path = abspath(entry.color_function.path)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        name = pick_unique_name(basename(path), bpy.data.texts)
        text = bpy.data.texts.new(name)
        text.from_string(content)
        entry.color_function_text = text
        entry.color_function.path = ""
        initialize_color_function(entry)
        return {'FINISHED'}


class UnembedColorFunctionOperator(bpy.types.Operator):  # pragma: no cover
    bl_idname = "skybrush.unembed_color_function"
    bl_label = "Unembed"
    bl_description = "Write embedded color function to external file"

    filepath: StringProperty(subtype="FILE_PATH", default="")

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry and entry.type == "FUNCTION" and entry.color_function_text

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        path = self.filepath or entry.color_function.path
        if not path:
            path = join(bpy.app.tempdir, basename(entry.color_function_text.name))
        path = abspath(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(entry.color_function_text.as_string())
        entry.color_function.path = path
        entry.color_function_text = None
        st = get_state(entry)
        st.clear()
        return {'FINISHED'}


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


class GeneratePathGradientMeshOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.generate_path_gradient_mesh"
    bl_label = "Generate Curve Mesh"
    bl_description = (
        "Create a curve with Geometry Nodes from the selected vertices and"
        " assign it to this light effect"
    )

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        edit_obj = context.edit_object
        return (
            entry is not None
            and entry.type == "COLOR_RAMP"
            and edit_obj is not None
            and edit_obj.type == "MESH"
        )

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        try:
            curve_obj = path_gradient.create_gradient_curve_from_selection()
        except Exception as exc:  # pragma: no cover - Blender UI
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        entry.mesh = curve_obj
        entry.target = "INSIDE_MESH"
        if hasattr(entry, "use_inside_mesh_vertex_colors"):
            entry.use_inside_mesh_vertex_colors = True

        self.report({'INFO'}, f"Generated curve mesh '{curve_obj.name}'")
        return {'FINISHED'}


class BakeColorRampSplitOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.bake_color_ramp_split"
    bl_label = "Bake (Split)"
    bl_description = (
        "Bake looped ColorRamp, splitting into multiple effects if needed"
    )

    max_points: IntProperty(name="Max Ramp Points", default=32, min=2)

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry and entry.type == "COLOR_RAMP" and entry.texture

    def _bake_into_ramp(self, ramp, baked_points):
        elems = ramp.elements
        baked_points.sort(key=lambda x: x[0])
        elems[0].position, elems[0].color = baked_points[0]
        while len(elems) > 1:
            elems.remove(elems[-1])
        for pos, col in baked_points[1:]:
            elem = elems.new(pos)
            elem.color = col

    def execute(self, context):
        scene = context.scene
        le = scene.skybrush.light_effects
        index = le.active_entry_index
        entry = le.active_entry
        if entry is None:
            return {'CANCELLED'}

        ramp = entry.texture.color_ramp
        loops = max(entry.loop_count, 1)
        src = [(e.position, list(e.color)) for e in ramp.elements]

        # If fits into limit, perform normal bake
        if len(src) * loops <= max(self.max_points, 2):
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
            self._bake_into_ramp(ramp, baked)
            entry.loop_count = 0
            entry.loop_method = "FORWARD"
            return {'FINISHED'}

        # Need to split into multiple effects by time segments
        src_count = max(len(src), 1)
        loops_per_chunk = max(self.max_points // src_count, 1)
        num_chunks = (loops + loops_per_chunk - 1) // loops_per_chunk

        base_name = entry.name
        group_id = uuid4().hex
        base_ramp_json = json.dumps(
            [[float(p), [float(c[0]), float(c[1]), float(c[2]), float(c[3])]] for p, c in src]
        )
        original_duration = int(entry.duration)
        original_start = int(entry.frame_start)
        loop_method = entry.loop_method

        # Prepare chunk timings
        # Split duration as evenly as possible
        chunk_starts = [original_start + (original_duration * i) // num_chunks for i in range(num_chunks)]
        chunk_durs = [
            (original_start + (original_duration * (i + 1)) // num_chunks) - chunk_starts[i]
            for i in range(num_chunks)
        ]
        chunk_durs = [max(d, 1) for d in chunk_durs]

        # We'll overwrite the original entry with the first chunk and append others
        remaining_loops = loops
        loop_index_offset = 0

        def build_baked_for_chunk(offset, count):
            baked_local = []
            for j in range(count):
                i = offset + j
                if loop_method == "PINGPONG":
                    forward = i % 2 == 0
                elif loop_method == "REVERSE":
                    forward = False
                else:
                    forward = True
                for pos, col in src:
                    new_pos = j / max(count, 1) + ((pos if forward else 1 - pos) / max(count, 1))
                    baked_local.append((new_pos, col[:]))
            return baked_local

        # Apply first chunk to the selected entry
        first_count = min(loops_per_chunk, remaining_loops)
        baked0 = build_baked_for_chunk(loop_index_offset, first_count)
        self._bake_into_ramp(ramp, baked0)
        entry.loop_count = 0
        entry.loop_method = "FORWARD"
        entry.frame_start = chunk_starts[0]
        entry.duration = chunk_durs[0]
        entry.name = pick_unique_name(f"{base_name} [1/{num_chunks}]", le.entries)
        entry["_split_group_id"] = group_id
        entry["_split_group_order"] = 0
        entry["_split_group_size"] = num_chunks
        entry["_split_group_loops_total"] = loops
        entry["_split_group_loop_method"] = loop_method
        entry["_split_group_base_name"] = base_name
        entry["_split_group_base_ramp"] = base_ramp_json
        entry["_split_group_original_start"] = original_start
        entry["_split_group_original_duration"] = original_duration

        remaining_loops -= first_count
        loop_index_offset += first_count

        # Create subsequent chunks
        for k in range(1, num_chunks):
            count = min(loops_per_chunk, remaining_loops)
            bakedk = build_baked_for_chunk(loop_index_offset, count)
            # Duplicate after current index to preserve order
            le.active_entry_index = index + k - 1
            new_entry = le.duplicate_selected_entry(select=True)
            new_entry.frame_start = chunk_starts[k]
            new_entry.duration = chunk_durs[k]
            new_entry.name = pick_unique_name(f"{base_name} [{k+1}/{num_chunks}]", le.entries)
            # Write ramp
            self._bake_into_ramp(new_entry.texture.color_ramp, bakedk)
            new_entry.loop_count = 0
            new_entry.loop_method = "FORWARD"
            # Tag as part of group
            new_entry["_split_group_id"] = group_id
            new_entry["_split_group_order"] = k
            new_entry["_split_group_size"] = num_chunks
            new_entry["_split_group_loops_total"] = loops
            new_entry["_split_group_loop_method"] = loop_method
            new_entry["_split_group_base_name"] = base_name
            new_entry["_split_group_base_ramp"] = base_ramp_json
            new_entry["_split_group_original_start"] = original_start
            new_entry["_split_group_original_duration"] = original_duration

            remaining_loops -= count
            loop_index_offset += count

        # Reselect first of the group
        le.active_entry_index = index
        return {'FINISHED'}


class MergeSplitLightEffectsOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.merge_split_light_effects"
    bl_label = "Merge Split Ramps"
    bl_description = "Merge split ColorRamp effects back into one looped effect"

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry and entry.get("_split_group_id") is not None

    def execute(self, context):
        scene = context.scene
        le = scene.skybrush.light_effects
        entry = le.active_entry
        if entry is None:
            return {'CANCELLED'}
        gid = entry.get("_split_group_id")
        if not gid:
            return {'CANCELLED'}
        # Collect all entries in the same group
        members = []
        for idx, e in enumerate(le.entries):
            try:
                if e.get("_split_group_id") == gid:
                    members.append((idx, e))
            except Exception:
                pass
        if not members:
            return {'CANCELLED'}
        members.sort(key=lambda t: int(t[1].get("_split_group_order", 0)))

        # Restore base values
        base_name = members[0][1].get("_split_group_base_name") or entry.name
        base_ramp_json = members[0][1].get("_split_group_base_ramp")
        loops_total = int(members[0][1].get("_split_group_loops_total", 1))
        loop_method = members[0][1].get("_split_group_loop_method", "FORWARD")
        original_start = int(members[0][1].get("_split_group_original_start", members[0][1].frame_start))
        original_duration = int(members[0][1].get("_split_group_original_duration", members[0][1].duration))

        # Use the first entry slot to place the merged effect
        first_index = members[0][0]
        le.active_entry_index = first_index
        merged = le.entries[first_index]
        merged.name = pick_unique_name(str(base_name), le.entries)
        merged.frame_start = original_start
        merged.duration = original_duration
        merged.loop_count = loops_total
        merged.loop_method = loop_method

        # Restore base ramp and then bake if desired? Keep as looped to avoid 32-limit again
        try:
            src = json.loads(base_ramp_json) if base_ramp_json else None
        except Exception:
            src = None
        if src:
            # Write base ramp back
            elems = merged.texture.color_ramp.elements
            elems[0].position, elems[0].color = float(src[0][0]), src[0][1]
            while len(elems) > 1:
                elems.remove(elems[-1])
            for pos, col in src[1:]:
                el = elems.new(float(pos))
                el.color = col

        # Remove the rest of the members (from last to first+1)
        for idx, _e in sorted(members[1:], key=lambda t: t[0], reverse=True):
            le.entries.remove(idx)

        # Clear split tags on merged
        for key in list(merged.keys()):
            if key.startswith("_split_group_"):
                try:
                    del merged[key]
                except Exception:
                    pass

        return {'FINISHED'}


class ConvertCollectionToMeshOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.convert_collection_to_mesh"
    bl_label = "Convert Collection to Mesh"
    bl_description = "Create animated mesh from target collection"

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return (
            entry
            and entry.target == "COLLECTION"
            and getattr(entry, "target_collection", None)
            and len(entry.target_collection.objects) > 0
        )

    def execute(self, context):
        scene = context.scene
        entry = scene.skybrush.light_effects.active_entry
        objs = list(entry.target_collection.objects)

        mesh_name = pick_unique_name(
            f"{entry.name}_targets", bpy.data.meshes
        )
        mesh = bpy.data.meshes.new(mesh_name)
        mesh_obj = bpy.data.objects.new(mesh_name, mesh)
        scene.collection.objects.link(mesh_obj)

        arm_name = pick_unique_name(f"{entry.name}_arm", bpy.data.armatures)
        armature = bpy.data.armatures.new(arm_name)
        arm_obj = bpy.data.objects.new(arm_name, armature)
        scene.collection.objects.link(arm_obj)

        vg_map = []
        with use_b_mesh(mesh) as bm:
            for obj in objs:
                loc = Vector(get_position_of_object(obj))
                result = bmesh.ops.create_cube(
                    bm, size=0.75, matrix=Matrix.Translation(loc)
                )
                vg_map.append([v.index for v in result["verts"]])

        mesh_obj.parent = arm_obj
        mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj

        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        for i, obj in enumerate(objs):
            loc = Vector(get_position_of_object(obj))
            bone = armature.edit_bones.new(f"Bone_{i}")
            bone.head = loc
            bone.tail = loc + Vector((0, 0, 0.1))
        bpy.ops.object.mode_set(mode="POSE")
        for i, obj in enumerate(objs):
            pb = arm_obj.pose.bones[f"Bone_{i}"]
            constr = pb.constraints.new(type="COPY_TRANSFORMS")
            constr.target = obj
        bpy.ops.object.mode_set(mode="OBJECT")

        for i, verts in enumerate(vg_map):
            vg = mesh_obj.vertex_groups.new(name=f"Bone_{i}")
            vg.add(verts, 1.0, "REPLACE")

        entry.mesh = mesh_obj
        entry.target = "INSIDE_MESH"
        entry.target_collection = None

        return {'FINISHED'}

def dyn_keys(pg):
    """Return dynamic ID property names for ``pg``."""
    # Ignore RNA slots and internal keys prefixed with ``_``
    return [k for k in pg.keys() if k != "_RNA_UI" and not k.startswith("_")]

def ensure_schema(pg, schema):
    for name, meta in schema.items():
        if name not in pg:
            default = meta.get("default") if hasattr(meta, "get") else meta["default"]
            pg[name] = default
        ui = pg.id_properties_ui(name)
        ui_kwargs = {}
        for k_src, k_dst in [
            ("min", "min"), ("max", "max"),
            ("soft_min", "soft_min"), ("soft_max", "soft_max"),
            ("desc", "description"),
            ("subtype", "subtype"),
        ]:
            if (hasattr(meta, "get") and k_src in meta) or (
                not hasattr(meta, "get") and k_src in meta
            ):
                value = meta.get(k_src) if hasattr(meta, "get") else meta[k_src]
                ui_kwargs[k_dst] = value
        if ui_kwargs:
            ui.update(**ui_kwargs)

def draw_dynamic(pg, layout, schema):
    """Draw ID properties from ``pg`` on ``layout``."""
    for name, meta in schema.items():
        if meta.get("type") == "OBJECT":
            layout.prop_search(pg, f'["{name}"]', bpy.data, "objects", text=name.upper())
        else:
            layout.prop(pg, f'["{name}"]', text=name.upper())

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
                    row = box.row(align=True)
                    if entry.color_function_text:
                        row.operator(
                            UnembedColorFunctionOperator.bl_idname, text="Unembed"
                        )
                    else:
                        row.prop(entry.color_function, "path", text="")
                        row.operator(
                            EmbedColorFunctionOperator.bl_idname, text="Embed"
                        )
                    if not entry.color_function_text:
                        box.prop(entry.color_function, "name", text="")
                    box.prop(entry, "color_function_text", text="")
                    entry.draw_color_function_config(box)
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
                row2 = col.row(align=True)
                if entry.get("_split_group_id"):
                    row2.operator(
                        MergeSplitLightEffectsOperator.bl_idname, text="Merge Splits"
                    )
                else:
                    row2.operator(BakeColorRampSplitOperator.bl_idname, text="Bake (Split)")
                col.separator()
                row_mesh = col.row()
                row_mesh.enabled = entry.target == "INSIDE_MESH"
                row_mesh.prop(
                    entry,
                    "use_inside_mesh_vertex_colors",
                    text="Use Mesh Vertex Colors",
                )
                col.operator(
                    GeneratePathGradientMeshOperator.bl_idname,
                    text="Generate Curve Mesh",
                    icon="OUTLINER_OB_CURVE",
                )
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
            if entry.target == "COLLECTION":
                col.prop(entry, "target_collection")
                col.operator(
                    ConvertCollectionToMeshOperator.bl_idname,
                    text="Convert to Mesh",
                )
            col.prop(entry, "blend_mode")
            col.prop(entry, "influence", slider=True)

            if effect_type_supports_randomization(entry.type):
                col.prop(entry, "randomness", slider=True)

def patch_light_effects_panel():
    LightEffectsPanel._original_draw = LightEffectsPanel.draw
    LightEffectsPanel.draw = PatchedLightEffectsPanel.draw


def unpatch_light_effects_panel():
    LightEffectsPanel.draw = LightEffectsPanel._original_draw
    LightEffectsPanel._original_draw = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register():  # pragma: no cover - executed in Blender
    bpy.utils.register_class(EmbedColorFunctionOperator)
    bpy.utils.register_class(UnembedColorFunctionOperator)
    bpy.utils.register_class(BakeColorRampOperator)
    bpy.utils.register_class(GeneratePathGradientMeshOperator)
    bpy.utils.register_class(ConvertCollectionToMeshOperator)
    bpy.utils.register_class(BakeColorRampSplitOperator)
    bpy.utils.register_class(MergeSplitLightEffectsOperator)
    if _ensure_light_effects_initialized not in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.append(_ensure_light_effects_initialized)


def unregister():  # pragma: no cover - executed in Blender
    if _ensure_light_effects_initialized in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.remove(_ensure_light_effects_initialized)
    bpy.utils.unregister_class(ConvertCollectionToMeshOperator)
    bpy.utils.unregister_class(GeneratePathGradientMeshOperator)
    bpy.utils.unregister_class(BakeColorRampOperator)
    bpy.utils.unregister_class(UnembedColorFunctionOperator)
    bpy.utils.unregister_class(EmbedColorFunctionOperator)
    bpy.utils.unregister_class(BakeColorRampSplitOperator)
    bpy.utils.unregister_class(MergeSplitLightEffectsOperator)

