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
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from os.path import basename, join

from bpy.app.handlers import persistent
from bpy.app import timers

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
from bpy.path import abspath as bpy_abspath
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
from sbutil import delaunay, path_gradient, selection_order

import sbstudio

OUTPUT_VERTEX_COLOR = "MESH_VERTEX_COLOR"
OUTPUT_MESH_UV_U = "MESH_UV_U"
OUTPUT_MESH_UV_V = "MESH_UV_V"


_selection_tracker_active = False


def linear_to_srgb(value: float) -> float:
    """Convert a linear RGB channel value to sRGB."""

    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0

    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4
    if value < 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _mesh_object_poll(_self, obj):
    return obj is None or getattr(obj, "type", "") == "MESH"


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
_delayed_id_property_updates: set[tuple[int, str]] = set()
_pending_dynamic_array_updates: dict[int, tuple] = {}
_dynamic_array_timer_running = False


def _schedule_id_property_update(pg, name: str, value):
    """Schedule ``pg[name] = value`` once writing to ID properties is allowed."""

    key = (pg.as_pointer(), name)
    if key in _delayed_id_property_updates:
        return

    def _apply():
        try:
            current = pg.get(name, None)
            if name in pg.keys() and _as_dict(current) == _as_dict(value):
                _delayed_id_property_updates.discard(key)
                return None
            pg[name] = value
        except (AttributeError, RuntimeError):
            return 0.1
        _delayed_id_property_updates.discard(key)
        return None

    _delayed_id_property_updates.add(key)
    timers.register(_apply, first_interval=0.0)


def _schedule_dynamic_array_update(array, payload):
    """Apply ``payload`` to ``array`` once data blocks become writable."""

    global _dynamic_array_timer_running

    _pending_dynamic_array_updates[array.as_pointer()] = (array,) + payload

    if _dynamic_array_timer_running:
        return

    def _apply():
        global _dynamic_array_timer_running
        if not _pending_dynamic_array_updates:
            _dynamic_array_timer_running = False
            return None

        for pointer, values in list(_pending_dynamic_array_updates.items()):
            array_obj = values[0]
            try:
                _apply_dynamic_array_values(array_obj, *values[1:])
            except (AttributeError, RuntimeError, ReferenceError):
                continue
            else:
                _pending_dynamic_array_updates.pop(pointer, None)

        if _pending_dynamic_array_updates:
            return 0.1

        _dynamic_array_timer_running = False
        return None

    _dynamic_array_timer_running = True
    timers.register(_apply, first_interval=0.0)


def _set_id_property(pg, name: str, value) -> None:
    """Safely assign ``value`` to ``pg[name]`` with context-aware fallbacks."""

    if name in pg.keys():
        current = pg.get(name, None)
        if _as_dict(current) == _as_dict(value):
            return
    try:
        pg[name] = value
    except (AttributeError, RuntimeError):
        _schedule_id_property_update(pg, name, _as_dict(value))


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
        tree = _build_bvh_tree(eval_mesh)
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
            value = (c0[0] * w0 + c1[0] * w1 + c2[0] * w2)
            outputs.append(float(value))
        return outputs
    finally:
        eval_obj.to_mesh_clear()


def sample_uv_factors(
    mesh_obj,
    positions: Sequence[Coordinate3D],
    axis: int,
) -> Optional[list[Optional[float]]]:
    """Return pseudo-UV ``axis`` samples using the object's bounding box."""

    if mesh_obj is None or not positions:
        return None

    if axis not in (0, 1):
        return [None] * len(positions)

    try:
        matrix = mesh_obj.matrix_world.copy()
    except ReferenceError:
        return [None] * len(positions)

    try:
        inv_world = matrix.inverted()
    except Exception:
        inv_world = matrix.inverted_safe()

    corners = getattr(mesh_obj, "bound_box", None) or ()
    local_points = [Vector(corner) for corner in corners]
    if not local_points:
        mesh_data = getattr(mesh_obj, "data", None)
        if mesh_data is not None and getattr(mesh_data, "vertices", None):
            try:
                local_points = [vert.co.copy() for vert in mesh_data.vertices]
            except Exception:
                local_points = []

    if not local_points:
        return [None] * len(positions)

    component_index = 0 if axis == 0 else 2  # U=X, V=Z
    values = [point[component_index] for point in local_points]
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if abs(range_val) < 1e-9:
        return [0.5 for _ in positions]

    inv_range = 1.0 / range_val
    outputs: list[Optional[float]] = []
    for pos in positions:
        co = Vector(pos) if not isinstance(pos, Vector) else pos.copy()
        local = inv_world @ co
        value = (local[component_index] - min_val) * inv_range
        outputs.append(max(0.0, min(1.0, float(value))))

    return outputs
        


def _build_bvh_tree(mesh):  # pragma: no cover - Blender integration
    """Construct a BVH tree for ``mesh`` compatible with multiple Blender versions."""

    from_mesh = getattr(BVHTree, "FromMesh", None)
    if from_mesh is not None:
        return from_mesh(mesh, epsilon=0.0)

    from_bmesh = getattr(BVHTree, "FromBMesh", None)
    if from_bmesh is not None:
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            return from_bmesh(bm)
        finally:
            bm.free()

    return None
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


def _normalize_float_sequence(values, length, fill_value=0.0):
    result = [float(v) for v in values]
    if len(result) < length:
        result.extend([fill_value] * (length - len(result)))
    return result[:length]


def _default_item_for_type(item_type):
    if item_type == "FLOAT":
        return 0.0
    if item_type in {"VECTOR", "EULER"}:
        return [0.0, 0.0, 0.0]
    if item_type == "SCALE":
        return [1.0, 1.0, 1.0]
    if item_type == "COLOR":
        return [1.0, 1.0, 1.0, 1.0]
    if item_type == "COLOR_RGB":
        return [1.0, 1.0, 1.0]
    return 0.0


def _determine_array_meta(name, value):
    if not isinstance(value, (tuple, list)):
        return None
    base_name = name[:-6]
    raw_items = list(value)
    converted: list[float | list[float]] = []
    for item in raw_items:
        if isinstance(item, (int, float)):
            converted.append(float(item))
            continue
        if isinstance(item, (tuple, list)) and all(isinstance(v, (int, float)) for v in item):
            converted.append([float(v) for v in item])
            continue
        return None
    item_type = "FLOAT"
    item_length = 1
    if converted and isinstance(converted[0], list):
        item_length = len(converted[0])
    suffix_type = None
    for suffix in ("_COLOR", "_POS", "_ROT", "_SCL", "_MAT"):
        if base_name.endswith(suffix):
            suffix_type = suffix
            break
    if suffix_type in {"_COLOR", "_MAT"}:
        if item_length >= 4:
            item_type = "COLOR"
            item_length = 4
            converted = [
                _normalize_float_sequence(it, 4, 1.0)  # type: ignore[arg-type]
                for it in converted  # type: ignore[arg-type]
            ]
        else:
            item_type = "COLOR_RGB"
            item_length = 3
            converted = [
                _normalize_float_sequence(it, 3, 1.0)  # type: ignore[arg-type]
                for it in converted  # type: ignore[arg-type]
            ]
    elif suffix_type == "_POS":
        item_type = "VECTOR"
        item_length = 3
        converted = [
            _normalize_float_sequence(it, 3, 0.0)  # type: ignore[arg-type]
            for it in converted  # type: ignore[arg-type]
        ]
    elif suffix_type == "_ROT":
        item_type = "EULER"
        item_length = 3
        converted = [
            _normalize_float_sequence(it, 3, 0.0)  # type: ignore[arg-type]
            for it in converted  # type: ignore[arg-type]
        ]
    elif suffix_type == "_SCL":
        item_type = "SCALE"
        item_length = 3
        converted = [
            _normalize_float_sequence(it, 3, 1.0)  # type: ignore[arg-type]
            for it in converted  # type: ignore[arg-type]
        ]
    else:
        if converted and isinstance(converted[0], list):
            if item_length == 4:
                item_type = "COLOR"
                converted = [
                    _normalize_float_sequence(it, 4, 1.0)  # type: ignore[arg-type]
                    for it in converted  # type: ignore[arg-type]
                ]
            elif item_length == 3:
                item_type = "VECTOR"
                converted = [
                    _normalize_float_sequence(it, 3, 0.0)  # type: ignore[arg-type]
                    for it in converted  # type: ignore[arg-type]
                ]
            else:
                return None
        else:
            item_type = "FLOAT"
            item_length = 1
    default_item = (
        converted[0] if converted else _default_item_for_type(item_type)
    )
    object_ref_type = None
    if suffix_type == "_POS":
        object_ref_type = "POS"
    elif suffix_type == "_ROT":
        object_ref_type = "ROT"
    elif suffix_type == "_SCL":
        object_ref_type = "SCL"
    elif suffix_type == "_MAT":
        object_ref_type = "MAT"
    meta = {
        "default": converted,
        "type": "ARRAY",
        "item_type": item_type,
        "item_length": item_length,
        "item_default": default_item,
    }
    if object_ref_type:
        meta["object_ref_type"] = object_ref_type
    return meta


def _sanitize_array_value(value, meta):
    data = _as_dict(value)
    if not isinstance(data, list):
        data = []
    item_type = meta.get("item_type", "FLOAT")
    item_length = int(meta.get("item_length", 1))
    sanitized: list[float | list[float]] = []
    if item_type == "FLOAT":
        for item in data:
            try:
                sanitized.append(float(item))
            except Exception:
                continue
        return sanitized
    fill = 0.0
    target_length = item_length
    if item_type == "SCALE":
        fill = 1.0
    if item_type == "COLOR":
        fill = 1.0
        target_length = 4
    elif item_type == "COLOR_RGB":
        fill = 1.0
        target_length = 3
    for item in data:
        seq = item if isinstance(item, (list, tuple)) else [item]
        sanitized.append(_normalize_float_sequence(seq, target_length, fill))
    return sanitized


def _convert_color_ramp_points(value):
    if not isinstance(value, (tuple, list)):
        return []
    points = []
    for item in value:
        if isinstance(item, dict):
            pos = float(item.get("position", 0.0))
            color = item.get("color", (1.0, 1.0, 1.0, 1.0))
        elif isinstance(item, (tuple, list)):
            if not item:
                continue
            pos = float(item[0])
            if len(item) == 2 and isinstance(item[1], (tuple, list)):
                color = item[1]
            else:
                color = item[1:5]
        else:
            continue
        color_seq = _normalize_float_sequence(color, 4, 1.0)
        pos = max(0.0, min(1.0, pos))
        points.append({"position": pos, "color": color_seq})
    points.sort(key=lambda item: item["position"])
    if not points:
        points = [
            {"position": 0.0, "color": [0.0, 0.0, 0.0, 1.0]},
            {"position": 1.0, "color": [1.0, 1.0, 1.0, 1.0]},
        ]
    return points


def _default_color_ramp_point(points):
    if not points:
        return {"position": 0.5, "color": [1.0, 1.0, 1.0, 1.0]}
    if len(points) == 1:
        pos = min(points[0]["position"] + 0.1, 1.0)
        return {"position": pos, "color": list(points[0]["color"])}
    first = points[0]
    last = points[-1]
    pos = (first["position"] + last["position"]) * 0.5
    color = [
        (first["color"][i] + last["color"][i]) * 0.5 for i in range(4)
    ]
    return {"position": max(0.0, min(1.0, pos)), "color": color}


def _find_dynamic_array_owner(item):
    owner = getattr(item, "id_data", None)
    if owner is None:
        return None, None
    skybrush = getattr(owner, "skybrush", None)
    light_effects = getattr(getattr(skybrush, "light_effects", None), "entries", [])
    for entry in light_effects:
        arrays = getattr(entry, "dynamic_arrays", None)
        if not arrays:
            continue
        for array in arrays:
            for value in array.values:
                if value.as_pointer() == item.as_pointer():
                    return entry, array
    return None, None


def _find_dynamic_color_ramp_owner(item):
    owner = getattr(item, "id_data", None)
    if owner is None:
        return None, None
    skybrush = getattr(owner, "skybrush", None)
    light_effects = getattr(getattr(skybrush, "light_effects", None), "entries", [])
    for entry in light_effects:
        ramps = getattr(entry, "dynamic_color_ramps", None)
        if not ramps:
            continue
        for ramp in ramps:
            for point in ramp.points:
                if point.as_pointer() == item.as_pointer():
                    return entry, ramp
    return None, None


def _update_dynamic_array_item(self, _context):
    entry, array = _find_dynamic_array_owner(self)
    if entry is not None and array is not None and array.name:
        _set_id_property(entry, array.name, array.to_storage_list())


def _update_dynamic_color_ramp_point(self, _context):
    entry, ramp = _find_dynamic_color_ramp_owner(self)
    if entry is not None and ramp is not None and ramp.name:
        _set_id_property(entry, ramp.name, ramp.to_storage_list())


def _find_sequence_delay_owner(item):
    owner = getattr(item, "id_data", None)
    if owner is None:
        return None
    skybrush = getattr(owner, "skybrush", None)
    light_effects = getattr(getattr(skybrush, "light_effects", None), "entries", [])
    for entry in light_effects:
        delays = getattr(entry, "sequence_delays", None)
        if not delays:
            continue
        for delay_entry in delays:
            if delay_entry.as_pointer() == item.as_pointer():
                return entry
    return None


def _update_sequence_delay_entry(self, _context):
    entry = _find_sequence_delay_owner(self)
    if entry is not None and hasattr(entry, "update_sequence_total_duration"):
        entry.update_sequence_total_duration()


def _update_sequence_duration(self, _context):
    if hasattr(self, "update_sequence_total_duration"):
        self.update_sequence_total_duration()


def _update_sequence_delay(self, _context):
    if hasattr(self, "update_sequence_total_duration"):
        self.update_sequence_total_duration()


def _update_sequence_manual_delay(self, _context):
    if hasattr(self, "ensure_sequence_delay_entries"):
        self.ensure_sequence_delay_entries(update_total=False)
    if hasattr(self, "update_sequence_total_duration"):
        self.update_sequence_total_duration()


def _update_sequence_mask_collection(self, _context):
    if hasattr(self, "ensure_sequence_delay_entries"):
        self.ensure_sequence_delay_entries(update_total=False)
    if hasattr(self, "update_sequence_total_duration"):
        self.update_sequence_total_duration()


def _update_sequence_mode(self, _context):
    if hasattr(self, "ensure_sequence_delay_entries"):
        self.ensure_sequence_delay_entries(update_total=False)
    if getattr(self, "sequence_mode", False) and hasattr(self, "sequence_duration"):
        try:
            current_duration = max(int(getattr(self, "duration", 0)), 1)
        except (TypeError, ValueError):
            current_duration = 1
        existing_duration = getattr(self, "sequence_duration", 0)
        if existing_duration <= 0 or existing_duration == DEFAULT_LIGHT_EFFECT_DURATION:
            self.sequence_duration = current_duration
    if hasattr(self, "update_sequence_total_duration"):
        self.update_sequence_total_duration()


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
        if hasattr(pg, "dynamic_arrays"):
            pg.dynamic_arrays.clear()
        if hasattr(pg, "dynamic_color_ramps"):
            pg.dynamic_color_ramps.clear()

    if getattr(pg, "color_function_text", None):
        text = pg.color_function_text
        source = text.as_string()
        text_hash = hashlib.sha256(source.encode()).hexdigest()
        if text_hash != st.get("text_hash"):
            reset_state()
            st["text_hash"] = text_hash
            _set_id_property(pg, "_text_hash", text_hash)
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
                        _set_id_property(pg, attr_name, value)
                    schema[attr_name] = {"default": value}
                elif isinstance(value, (tuple, list)):
                    if name.endswith("_COLORRAMP"):
                        points = _convert_color_ramp_points(value)
                        current = _convert_color_ramp_points(pg.get(attr_name, points))
                        _set_id_property(pg, attr_name, current)
                        schema[attr_name] = {
                            "default": points,
                            "type": "COLOR_RAMP",
                            "default_point": _default_color_ramp_point(points),
                        }
                    elif name.endswith("_ARRAY"):
                        meta = _determine_array_meta(name, value)
                        if not meta:
                            continue
                        current = _sanitize_array_value(pg.get(attr_name, meta["default"]), meta)
                        _set_id_property(pg, attr_name, current)
                        obj_type = meta.get("object_ref_type")
                        if obj_type:
                            obj_attr_name = f"{attr_name}_object"
                            if obj_attr_name not in pg:
                                _set_id_property(pg, obj_attr_name, "")
                            schema[obj_attr_name] = {
                                "default": "",
                                "type": "OBJECT",
                            }
                            meta["object_ref_attr"] = obj_attr_name
                        schema[attr_name] = meta
                    elif all(isinstance(v, (int, float)) for v in value):
                        list_value = list(value)
                        if attr_name not in pg:
                            _set_id_property(pg, attr_name, list_value)
                        meta = {"default": list_value}
                        if name.endswith("_COLOR"):
                            meta["subtype"] = "COLOR"
                        elif name.endswith("_POS"):
                            meta["subtype"] = "XYZ"
                            obj_attr_name = f"{attr_name}_object"
                            if obj_attr_name not in pg:
                                _set_id_property(pg, obj_attr_name, "")
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
                                _set_id_property(pg, obj_attr_name, "")
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
                                _set_id_property(pg, obj_attr_name, "")
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
                                _set_id_property(pg, obj_attr_name, "")
                            schema[obj_attr_name] = {
                                "default": "",
                                "type": "OBJECT",
                            }
                            meta["object_ref_attr"] = obj_attr_name
                            meta["object_ref_type"] = "MAT"
                        schema[attr_name] = meta
            st["config_schema"] = schema
            _set_id_property(pg, "_config_schema", schema)
            # Restore cached property values when possible, but only update
            # when the value actually differs from the current one.
            for name, value in cached_values.items():
                if name in schema and pg.get(name) != value:
                    _set_id_property(pg, name, value)
        return

    if not pg.color_function or not pg.color_function.path:
        return

    ap = bpy_abspath(pg.color_function.path)
    if not ap.lower().endswith(".py"):
        reset_state()
        return
    if ap != st.get("absolute_path", ""):
        reset_state()
        st["absolute_path"] = ap
        _set_id_property(pg, "_absolute_path", ap)
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
                    _set_id_property(pg, attr_name, value)
                schema[attr_name] = {"default": value}
            elif isinstance(value, (tuple, list)):
                if name.endswith("_COLORRAMP"):
                    points = _convert_color_ramp_points(value)
                    current = _convert_color_ramp_points(pg.get(attr_name, points))
                    _set_id_property(pg, attr_name, current)
                    schema[attr_name] = {
                        "default": points,
                        "type": "COLOR_RAMP",
                        "default_point": _default_color_ramp_point(points),
                    }
                elif name.endswith("_ARRAY"):
                    meta = _determine_array_meta(name, value)
                    if not meta:
                        continue
                    current = _sanitize_array_value(pg.get(attr_name, meta["default"]), meta)
                    _set_id_property(pg, attr_name, current)
                    obj_type = meta.get("object_ref_type")
                    if obj_type:
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            _set_id_property(pg, obj_attr_name, "")
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                    schema[attr_name] = meta
                elif all(isinstance(v, (int, float)) for v in value):
                    list_value = list(value)
                    if attr_name not in pg:
                        _set_id_property(pg, attr_name, list_value)
                    meta = {"default": list_value}
                    if name.endswith("_COLOR"):
                        meta["subtype"] = "COLOR"
                    elif name.endswith("_POS"):
                        meta["subtype"] = "XYZ"
                        obj_attr_name = f"{attr_name}_object"
                        if obj_attr_name not in pg:
                            _set_id_property(pg, obj_attr_name, "")
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
                            _set_id_property(pg, obj_attr_name, "")
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
                            _set_id_property(pg, obj_attr_name, "")
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
                            _set_id_property(pg, obj_attr_name, "")
                        schema[obj_attr_name] = {
                            "default": "",
                            "type": "OBJECT",
                        }
                        meta["object_ref_attr"] = obj_attr_name
                        meta["object_ref_type"] = "MAT"
                    schema[attr_name] = meta
        st["config_schema"] = schema
        _set_id_property(pg, "_config_schema", schema)
        for name, value in cached_values.items():
            if name in schema and pg.get(name) != value:
                _set_id_property(pg, name, value)


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


class SequenceDelayEntry(PropertyGroup):
    delay: IntProperty(name="Delay", default=0, min=0, update=_update_sequence_delay_entry)


class DynamicArrayValue(PropertyGroup):
    value_float: FloatProperty(name="Value", default=0.0, update=_update_dynamic_array_item)
    vector: FloatVectorProperty(
        name="Vector",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        update=_update_dynamic_array_item,
    )
    rotation: FloatVectorProperty(
        name="Rotation",
        size=3,
        subtype="EULER",
        default=(0.0, 0.0, 0.0),
        update=_update_dynamic_array_item,
    )
    scale: FloatVectorProperty(
        name="Scale",
        size=3,
        subtype="XYZ",
        default=(1.0, 1.0, 1.0),
        update=_update_dynamic_array_item,
    )
    color: FloatVectorProperty(
        name="Color",
        size=4,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        update=_update_dynamic_array_item,
    )
    color_rgb: FloatVectorProperty(
        name="Color",
        size=3,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0),
        update=_update_dynamic_array_item,
    )

    def set_value(self, item_type: str, value, item_length: int) -> None:
        if item_type == "FLOAT":
            try:
                self.value_float = float(value if not isinstance(value, (list, tuple)) else value[0])
            except Exception:
                self.value_float = 0.0
            return
        seq = value if isinstance(value, (list, tuple)) else [value]
        if item_type == "VECTOR":
            self.vector = _normalize_float_sequence(seq, 3, 0.0)
        elif item_type == "EULER":
            self.rotation = _normalize_float_sequence(seq, 3, 0.0)
        elif item_type == "SCALE":
            self.scale = _normalize_float_sequence(seq, 3, 1.0)
        elif item_type == "COLOR":
            self.color = _normalize_float_sequence(seq, 4, 1.0)
        elif item_type == "COLOR_RGB":
            self.color_rgb = _normalize_float_sequence(seq, 3, 1.0)
        else:
            self.value_float = float(seq[0]) if seq else 0.0

    def to_storage(self, item_type: str, item_length: int):
        if item_type == "FLOAT":
            return float(self.value_float)
        if item_type == "VECTOR":
            return list(self.vector)[:item_length]
        if item_type == "EULER":
            return list(self.rotation)[:item_length]
        if item_type == "SCALE":
            return list(self.scale)[:item_length]
        if item_type == "COLOR":
            return list(self.color)[:4]
        if item_type == "COLOR_RGB":
            return list(self.color_rgb)[:3]
        return float(self.value_float)

    def to_script(self, item_type: str, item_length: int):
        value = self.to_storage(item_type, item_length)
        if isinstance(value, list):
            return tuple(float(v) for v in value)
        return float(value)


def _apply_dynamic_array_values(array, item_type: str, item_length: int, sanitized_values: Sequence):
    """Assign ``sanitized_values`` to ``array`` with the given metadata."""

    array.item_type = item_type
    array.item_length = item_length
    array.values.clear()
    for item in sanitized_values:
        arr_item = array.values.add()
        arr_item.set_value(item_type, item, item_length)


class DynamicArrayProperty(PropertyGroup):
    name: StringProperty(name="Name", default="")
    item_type: StringProperty(name="Item Type", default="FLOAT")
    item_length: IntProperty(name="Item Length", default=1, min=1, max=4)
    values: CollectionProperty(type=DynamicArrayValue)

    def set_from_python(self, data, meta):
        item_type = meta.get("item_type", "FLOAT")
        item_length = int(meta.get("item_length", 1))
        sanitized = list(_sanitize_array_value(data, meta))

        try:
            _apply_dynamic_array_values(self, item_type, item_length, sanitized)
        except (AttributeError, RuntimeError, ReferenceError):
            _schedule_dynamic_array_update(self, (item_type, item_length, sanitized))

    def append_default(self, meta):
        default_value = meta.get("item_default", _default_item_for_type(self.item_type))
        arr_item = self.values.add()
        arr_item.set_value(self.item_type, default_value, self.item_length)
        return arr_item

    def to_storage_list(self):
        return [item.to_storage(self.item_type, self.item_length) for item in self.values]

    def to_script_value(self):
        return [item.to_script(self.item_type, self.item_length) for item in self.values]


class DynamicColorRampPoint(PropertyGroup):
    position: FloatProperty(
        name="Position", min=0.0, max=1.0, default=0.0, update=_update_dynamic_color_ramp_point
    )
    color: FloatVectorProperty(
        name="Color",
        size=4,
        subtype="COLOR",
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        update=_update_dynamic_color_ramp_point,
    )


class DynamicColorRamp(PropertyGroup):
    name: StringProperty(name="Name", default="")
    points: CollectionProperty(type=DynamicColorRampPoint)

    def set_from_python(self, data):
        points = _convert_color_ramp_points(data)
        self.points.clear()
        for entry in points:
            point = self.points.add()
            point.position = entry["position"]
            point.color = entry["color"]

    def append_point(self, position: float, color):
        data = self.to_storage_list()
        data.append(
            {
                "position": max(0.0, min(1.0, float(position))),
                "color": _normalize_float_sequence(color, 4, 1.0),
            }
        )
        self.set_from_python(data)

    def remove_point(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points.remove(index)

    def to_storage_list(self):
        return [
            {"position": point.position, "color": list(point.color)}
            for point in self.points
        ]

    def to_script_value(self):
        return [
            (float(point.position), tuple(point.color))
            for point in self.points
        ]

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
    color_function_text = PointerProperty(
        name="Color Function Text", type=bpy.types.Text
    )
    convert_srgb = BoolProperty(
        name="Convert sRGB",
        description=(
            "Convert colors sampled from images from linear space to sRGB"
        ),
        default=False,
    )
    sequence_mode = BoolProperty(
        name="Sequence Mode",
        description="Enable sequential playback across multiple mask meshes",
        default=False,
        update=_update_sequence_mode,
    )
    sequence_mask_collection = PointerProperty(
        name="Mask Collection",
        description="Collection that stores meshes participating in the sequence",
        type=bpy.types.Collection,
        update=_update_sequence_mask_collection,
    )
    sequence_duration = IntProperty(
        name="Sequence Duration",
        description="Duration of each individual sequence entry",
        default=DEFAULT_LIGHT_EFFECT_DURATION,
        min=1,
        update=_update_sequence_duration,
    )
    sequence_delay = IntProperty(
        name="Delay",
        description="Delay between the start frames of consecutive sequence entries",
        default=0,
        min=0,
        update=_update_sequence_delay,
    )
    sequence_manual_delay = BoolProperty(
        name="Manual Delay",
        description="Allow specifying individual delays between each pair of sequence entries",
        default=False,
        update=_update_sequence_manual_delay,
    )
    sequence_delays = CollectionProperty(
        name="Sequence Delays",
        description="Per-pair delays between sequence entries when manual delay is enabled",
        type=SequenceDelayEntry,
    )
    dynamic_arrays = CollectionProperty(
        name="Dynamic Arrays",
        description="Storage for dynamic custom function arrays",
        type=DynamicArrayProperty,
    )
    dynamic_color_ramps = CollectionProperty(
        name="Dynamic Color Ramps",
        description="Storage for dynamic color ramp parameters",
        type=DynamicColorRamp,
    )

    def update_sequence_total_duration(self) -> None:
        if not getattr(self, "sequence_mode", False):
            return
        try:
            base_duration = max(int(getattr(self, "sequence_duration", 0)), 0)
        except (TypeError, ValueError):
            base_duration = 0
        if base_duration <= 0:
            base_duration = 1

        meshes = self.get_sequence_meshes()
        mesh_pairs = max(len(meshes) - 1, 0)
        total_delay = 0
        if mesh_pairs > 0:
            if getattr(self, "sequence_manual_delay", False):
                self.ensure_sequence_delay_entries(mesh_pairs, update_total=False)
                for delay_entry in self.sequence_delays[:mesh_pairs]:
                    try:
                        total_delay += max(int(getattr(delay_entry, "delay", 0)), 0)
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    per_delay = max(int(getattr(self, "sequence_delay", 0)), 0)
                except (TypeError, ValueError):
                    per_delay = 0
                total_delay = per_delay * mesh_pairs

        total_duration = max(base_duration + total_delay, 1)

        # Avoid writing to ``self.duration`` here as it may run in contexts where
        # ID property updates are not allowed (e.g. while drawing the UI).
        # The effective duration is still derived on demand via
        # ``calculate_effective_sequence_span``.
        return total_duration

    def ensure_sequence_delay_entries(
        self, count: int | None = None, *, update_total: bool = True
    ) -> None:
        meshes = self.get_sequence_meshes()
        desired = max(len(meshes) - 1, 0) if count is None else max(count, 0)
        current = len(self.sequence_delays)
        if current < desired:
            for _ in range(desired - current):
                item = self.sequence_delays.add()
                item.delay = max(self.sequence_delay, 0)
        elif current > desired:
            for _ in range(current - desired):
                self.sequence_delays.remove(len(self.sequence_delays) - 1)
        if update_total:
            self.update_sequence_total_duration()

    def get_sequence_meshes(self) -> list[bpy.types.Object]:
        collection = getattr(self, "sequence_mask_collection", None)
        if not collection:
            return []
        meshes = [
            obj
            for obj in getattr(collection, "objects", [])
            if getattr(obj, "type", "") == "MESH"
        ]
        meshes.sort(key=lambda obj: obj.name)
        return meshes

    def get_sequence_delays(self) -> list[int]:
        meshes = self.get_sequence_meshes()
        desired = max(len(meshes) - 1, 0)
        self.ensure_sequence_delay_entries(desired)
        if self.sequence_manual_delay:
            return [entry.delay for entry in self.sequence_delays[:desired]]
        return [self.sequence_delay for _ in range(desired)]

    def _get_dynamic_array(self, name: str) -> DynamicArrayProperty | None:
        for array in self.dynamic_arrays:
            if array.name == name:
                return array
        return None

    def ensure_dynamic_array(self, name: str, meta: dict | None = None) -> DynamicArrayProperty:
        meta = meta or get_state(self).get("config_schema", {}).get(name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        array = self._get_dynamic_array(name)
        if array is None:
            array = self.dynamic_arrays.add()
            array.name = name
        array.set_from_python(self.get(name, meta.get("default", [])), meta)
        _set_id_property(self, name, array.to_storage_list())
        return array

    def get_dynamic_array_value(self, name: str, meta: dict | None = None):
        array = self.ensure_dynamic_array(name, meta)
        return array.to_script_value()

    def _get_dynamic_color_ramp(self, name: str) -> DynamicColorRamp | None:
        for ramp in self.dynamic_color_ramps:
            if ramp.name == name:
                return ramp
        return None

    def ensure_dynamic_color_ramp(self, name: str, meta: dict | None = None) -> DynamicColorRamp:
        meta = meta or get_state(self).get("config_schema", {}).get(name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        ramp = self._get_dynamic_color_ramp(name)
        if ramp is None:
            ramp = self.dynamic_color_ramps.add()
            ramp.name = name
        ramp.set_from_python(self.get(name, meta.get("default", [])))
        _set_id_property(self, name, ramp.to_storage_list())
        return ramp

    def get_dynamic_color_ramp_value(self, name: str, meta: dict | None = None):
        ramp = self.ensure_dynamic_color_ramp(name, meta)
        return ramp.to_script_value()

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

        def apply_single_mode(*, clamp_temporal_to_duration: bool = False) -> None:
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
                elif output_type == OUTPUT_VERTEX_COLOR:
                    outputs = None
                    if (
                        self.type == "COLOR_RAMP"
                        and self.target == "INSIDE_MESH"
                        and getattr(self, "mesh", None)
                    ):
                        sampled = sample_vertex_color_factors(self.mesh, positions)
                        if sampled is not None:
                            outputs = sampled
                    if outputs is None:
                        outputs = [None] * num_positions
                elif output_type in {OUTPUT_MESH_UV_U, OUTPUT_MESH_UV_V}:
                    outputs = None
                    uv_source = getattr(self, "uv_mesh", None) or getattr(self, "mesh", None)
                    if uv_source is not None:
                        axis = 0 if output_type == OUTPUT_MESH_UV_U else 1
                        sampled = sample_uv_factors(uv_source, positions, axis)
                        if sampled is not None:
                            outputs = sampled
                    if outputs is None:
                        outputs = [None] * num_positions
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
                            load_module(bpy_abspath(path))
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

            if not self.enabled:
                return

            effective_frame = frame
            if clamp_temporal_to_duration:
                start_frame = self.frame_start
                duration_span = max(self.duration - 1, 0)
                if duration_span <= 0:
                    effective_frame = start_frame
                else:
                    end_frame = start_frame + duration_span
                    if frame < start_frame:
                        effective_frame = start_frame
                    elif frame > end_frame:
                        effective_frame = end_frame
            elif not self.contains_frame(frame):
                return

            time_fraction = (effective_frame - self.frame_start) / max(
                self.duration - 1, 1
            )
            num_positions = len(positions)
            color_ramp = self.color_ramp
            color_image = self.color_image
            color_function_ref = self.color_function_ref
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
                        meta_type = meta.get("type")
                        if meta_type == "ARRAY":
                            value = self.get_dynamic_array_value(name, meta)
                        elif meta_type == "COLOR_RAMP":
                            value = self.get_dynamic_color_ramp_value(name, meta)
                        else:
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
                        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                            ctx[name.upper()] = tuple(value)
                        else:
                            ctx[name.upper()] = value
                elif color_function_ref is not None:
                    for name, meta in schema.items():
                        if meta.get("type") == "OBJECT":
                            continue
                        meta_type = meta.get("type")
                        if meta_type == "ARRAY":
                            value = self.get_dynamic_array_value(name, meta)
                        elif meta_type == "COLOR_RAMP":
                            value = self.get_dynamic_color_ramp_value(name, meta)
                        else:
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
                        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                            setattr(module, name.upper(), tuple(value))
                        else:
                            setattr(module, name.upper(), value)
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
                    min(
                        self._evaluate_influence_at(
                            position, effective_frame, condition
                        ),
                        1.0,
                    ),
                    0.0,
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
                    offset = (y * width + x) * 4
                    new_color[:] = pixels[offset : offset + 4]
                    if getattr(self, "convert_srgb", False):
                        for idx in range(3):
                            new_color[idx] = linear_to_srgb(new_color[idx])
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

        if not self.sequence_mode:
            apply_single_mode()
            return

        meshes = self.get_sequence_meshes()
        if not meshes:
            apply_single_mode()
            return

        delays = self.get_sequence_delays()
        original_start = self.frame_start
        original_duration = self.duration
        original_mesh = getattr(self, "mesh", None)
        has_frame_end = hasattr(self, "frame_end")
        original_end = getattr(self, "frame_end", None) if has_frame_end else None
        chunk_start = original_start
        sequence_duration_value = getattr(self, "sequence_duration", original_duration)
        try:
            sequence_duration = max(int(sequence_duration_value), 1)
        except (TypeError, ValueError):
            try:
                sequence_duration = max(int(original_duration), 1)
            except (TypeError, ValueError):
                sequence_duration = 1
        duration_span = max(sequence_duration - 1, 0)
        for index, mesh in enumerate(meshes):
            if index > 0:
                delay = (
                    delays[index - 1]
                    if index - 1 < len(delays)
                    else self.sequence_delay
                )
                chunk_start += max(delay, 0)
            chunk_end = chunk_start + duration_span
            self.frame_start = chunk_start
            self.duration = sequence_duration
            self.mesh = mesh
            apply_single_mode(clamp_temporal_to_duration=True)
        self.frame_start = original_start
        self.duration = original_duration
        self.mesh = original_mesh
        if has_frame_end and original_end is not None:
            self.frame_end = original_end


def calculate_effective_sequence_span(effect) -> tuple[int | None, int | None]:
    """Return the effective ``(frame_end, duration)`` span for ``effect``.

    The helper keeps the user-facing duration untouched in sequence mode while
    still allowing export code to query the total span that includes sequence
    delays.
    """

    frame_start = getattr(effect, "frame_start", None)
    duration = getattr(effect, "duration", None)
    if frame_start is None or duration is None:
        return getattr(effect, "frame_end", None), duration

    try:
        frame_start_i = int(frame_start)
        duration_i = max(int(duration), 0)
    except (TypeError, ValueError):
        return getattr(effect, "frame_end", None), duration

    duration_span = max(duration_i - 1, 0)
    sequence_duration_value = getattr(effect, "sequence_duration", duration_i)
    try:
        sequence_duration_i = max(int(sequence_duration_value), 0)
    except (TypeError, ValueError):
        sequence_duration_i = duration_i
    sequence_span = max(sequence_duration_i - 1, 0)
    has_frame_end = hasattr(effect, "frame_end")
    original_end = getattr(effect, "frame_end", None) if has_frame_end else None
    total_end = (
        int(original_end)
        if isinstance(original_end, (int, float))
        else frame_start_i + duration_span
    )

    if not getattr(effect, "sequence_mode", False):
        return total_end, duration_i

    get_meshes = getattr(effect, "get_sequence_meshes", None)
    get_delays = getattr(effect, "get_sequence_delays", None)
    if get_meshes is None or get_delays is None:
        return total_end, duration_i

    meshes = list(get_meshes())
    if len(meshes) <= 1:
        return total_end, duration_i

    delays = list(get_delays())
    chunk_start = frame_start_i
    total_end = max(total_end, frame_start_i + sequence_span)
    for index in range(1, len(meshes)):
        delay = delays[index - 1] if index - 1 < len(delays) else getattr(effect, "sequence_delay", 0)
        try:
            delay_i = max(int(delay), 0)
        except (TypeError, ValueError):
            delay_i = 0
        chunk_start += delay_i
        chunk_end = chunk_start + sequence_span
        total_end = max(total_end, chunk_end)

    total_duration = max(total_end - frame_start_i + 1, duration_i, sequence_duration_i)
    return total_end, total_duration

def patch_light_effect_class():
    """Inject loop properties into ``LightEffect`` using monkey patching."""
    if LightEffect is None:  # pragma: no cover - only runs inside Blender
        return
            
    LightEffect.original_type = getattr(LightEffect, "type", None)
    LightEffect._original_apply_on_colors = getattr(LightEffect, "apply_on_colors", None)
    LightEffect._original_color_function_ref = getattr(
        LightEffect, "color_function_ref", None
    )
    LightEffect._original_draw_color_function_config = getattr(
        LightEffect, "draw_color_function_config", None
    )
    LightEffect.original_target = getattr(LightEffect, "target", None)
    LightEffect._original_get_spatial_effect_predicate = getattr(
        LightEffect, "_get_spatial_effect_predicate", None
    )
    LightEffect.type = PatchedLightEffect.type
    LightEffect.loop_count = PatchedLightEffect.loop_count
    LightEffect.loop_method = PatchedLightEffect.loop_method
    LightEffect.color_function_text = PatchedLightEffect.color_function_text
    LightEffect.convert_srgb = PatchedLightEffect.convert_srgb
    LightEffect.sequence_mode = PatchedLightEffect.sequence_mode
    LightEffect.sequence_mask_collection = PatchedLightEffect.sequence_mask_collection
    LightEffect.sequence_duration = PatchedLightEffect.sequence_duration
    LightEffect.sequence_delay = PatchedLightEffect.sequence_delay
    LightEffect.sequence_manual_delay = PatchedLightEffect.sequence_manual_delay
    LightEffect.sequence_delays = PatchedLightEffect.sequence_delays
    LightEffect.ensure_sequence_delay_entries = PatchedLightEffect.ensure_sequence_delay_entries
    LightEffect.get_sequence_meshes = PatchedLightEffect.get_sequence_meshes
    LightEffect.get_sequence_delays = PatchedLightEffect.get_sequence_delays
    LightEffect.update_sequence_total_duration = PatchedLightEffect.update_sequence_total_duration
    LightEffect.apply_on_colors = PatchedLightEffect.apply_on_colors
    LightEffect.color_function_ref = PatchedLightEffect.color_function_ref
    LightEffect.draw_color_function_config = PatchedLightEffect.draw_color_function_config
    LightEffect._get_dynamic_array = PatchedLightEffect._get_dynamic_array
    LightEffect.ensure_dynamic_array = PatchedLightEffect.ensure_dynamic_array
    LightEffect.get_dynamic_array_value = PatchedLightEffect.get_dynamic_array_value
    LightEffect._get_dynamic_color_ramp = PatchedLightEffect._get_dynamic_color_ramp
    LightEffect.ensure_dynamic_color_ramp = PatchedLightEffect.ensure_dynamic_color_ramp
    LightEffect.get_dynamic_color_ramp_value = PatchedLightEffect.get_dynamic_color_ramp_value
    LightEffect.dynamic_arrays = PatchedLightEffect.dynamic_arrays
    LightEffect.dynamic_color_ramps = PatchedLightEffect.dynamic_color_ramps
    LightEffect.original_output = getattr(LightEffect, "output", None)
    LightEffect.output = EnumProperty(
        name="Output X",
        description="Output function that determines the value that is passed through the color ramp or image horizontal (X) axis to obtain the color to assign to a drone",
        items=[
            ("FIRST_COLOR", "First color", "", 1),
            ("LAST_COLOR", "Last color", "", 2),
            ("INDEXED_BY_DRONES", "Indexed by drones", "", 3),
            ("INDEXED_BY_FORMATION", "Indexed by formation", "", 13),
            ("GRADIENT_XYZ", "Gradient (XYZ)", "", 4),
            ("GRADIENT_XZY", "Gradient (XZY)", "", 5),
            ("GRADIENT_YXZ", "Gradient (YXZ)", "", 6),
            ("GRADIENT_YZX", "Gradient (YZX)", "", 7),
            ("GRADIENT_ZXY", "Gradient (ZXY)", "", 8),
            ("GRADIENT_ZYX", "Gradient (ZYX)", "", 9),
            ("TEMPORAL", "Temporal", "", 10),
            ("DISTANCE", "Distance from mesh", "", 11),
            ("CUSTOM", "Custom expression", "", 12),
            (OUTPUT_VERTEX_COLOR, "Mesh vertex color", "", 14),
            (OUTPUT_MESH_UV_U, "Mesh UV (U)", "", 15),
        ],
        default="LAST_COLOR",
    )
    LightEffect.original_output_y = getattr(LightEffect, "output_y", None)
    LightEffect.output_y = EnumProperty(
        name="Output Y",
        description="Output function that determines the value that is passed through the image vertical (Y) axis",
        items=[
            ("FIRST_COLOR", "First color", "", 1),
            ("LAST_COLOR", "Last color", "", 2),
            ("INDEXED_BY_DRONES", "Indexed by drones", "", 3),
            ("INDEXED_BY_FORMATION", "Indexed by formation", "", 13),
            ("GRADIENT_XYZ", "Gradient (XYZ)", "", 4),
            ("GRADIENT_XZY", "Gradient (XZY)", "", 5),
            ("GRADIENT_YXZ", "Gradient (YXZ)", "", 6),
            ("GRADIENT_YZX", "Gradient (YZX)", "", 7),
            ("GRADIENT_ZXY", "Gradient (ZXY)", "", 8),
            ("GRADIENT_ZYX", "Gradient (ZYX)", "", 9),
            ("TEMPORAL", "Temporal", "", 10),
            ("DISTANCE", "Distance from mesh", "", 11),
            ("CUSTOM", "Custom expression", "", 12),
            (OUTPUT_VERTEX_COLOR, "Mesh vertex color", "", 14),
            (OUTPUT_MESH_UV_V, "Mesh UV (V)", "", 15),
        ],
        default="LAST_COLOR",
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
    LightEffect.original_uv_mesh = getattr(LightEffect, "uv_mesh", None)
    LightEffect.uv_mesh = PointerProperty(
        name="UV Mesh",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
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
    LightEffect.__annotations__["color_function_text"] = LightEffect.color_function_text
    LightEffect.__annotations__["convert_srgb"] = LightEffect.convert_srgb
    LightEffect.__annotations__["sequence_mode"] = LightEffect.sequence_mode
    LightEffect.__annotations__["sequence_mask_collection"] = (
        LightEffect.sequence_mask_collection
    )
    LightEffect.__annotations__["sequence_duration"] = LightEffect.sequence_duration
    LightEffect.__annotations__["sequence_delay"] = LightEffect.sequence_delay
    LightEffect.__annotations__["sequence_manual_delay"] = (
        LightEffect.sequence_manual_delay
    )
    LightEffect.__annotations__["sequence_delays"] = LightEffect.sequence_delays
    LightEffect.__annotations__["output"] = LightEffect.output
    LightEffect.__annotations__["output_y"] = LightEffect.output_y
    LightEffect.__annotations__["target"] = LightEffect.target
    LightEffect.__annotations__["uv_mesh"] = LightEffect.uv_mesh
    LightEffect.__annotations__["target_collection"] = LightEffect.target_collection
    ensure_all_function_entries_initialized()

def unpatch_light_effect_class():
    if LightEffect is None or not hasattr(LightEffect, "original_type"):  # pragma: no cover
        return
    bpy.utils.unregister_class(LightEffect)
    LightEffect.type = LightEffect.original_type
    LightEffect.__annotations__["type"] = LightEffect.original_type
    for attr in (
        "loop_count",
        "loop_method",
        "color_function_text",
        "convert_srgb",
        "sequence_mode",
        "sequence_mask_collection",
        "sequence_duration",
        "sequence_delay",
        "sequence_manual_delay",
        "sequence_delays",
        "ensure_sequence_delay_entries",
        "get_sequence_meshes",
        "get_sequence_delays",
        "update_sequence_total_duration",
    ):
        if hasattr(LightEffect, attr):
            delattr(LightEffect, attr)
            LightEffect.__annotations__.pop(attr, None)
    if getattr(LightEffect, "original_output", None) is not None:
        LightEffect.output = LightEffect.original_output
        LightEffect.__annotations__["output"] = LightEffect.original_output
        LightEffect.original_output = None
    if getattr(LightEffect, "original_output_y", None) is not None:
        LightEffect.output_y = LightEffect.original_output_y
        LightEffect.__annotations__["output_y"] = LightEffect.original_output_y
        LightEffect.original_output_y = None
    elif hasattr(LightEffect, "output_y"):
        delattr(LightEffect, "output_y")
        LightEffect.__annotations__.pop("output_y", None)
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
    if getattr(LightEffect, "original_target", None) is not None:
        LightEffect.target = LightEffect.original_target
        LightEffect.__annotations__["target"] = LightEffect.original_target
        LightEffect.original_target = None
    if getattr(LightEffect, "original_uv_mesh", None) is not None:
        LightEffect.uv_mesh = LightEffect.original_uv_mesh
        LightEffect.__annotations__["uv_mesh"] = LightEffect.original_uv_mesh
        LightEffect.original_uv_mesh = None
    elif hasattr(LightEffect, "uv_mesh"):
        delattr(LightEffect, "uv_mesh")
        LightEffect.__annotations__.pop("uv_mesh", None)
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
    LightEffect.original_type = None
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
        path = bpy_abspath(entry.color_function.path)
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
        path = bpy_abspath(path)
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
        "Create a curve with Geometry Nodes from the selection."
        " When used on multiple objects, run once to enable selection order"
        " tracking, select objects in order, then run again to build the curve."
    )

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry is not None and entry.type == "COLOR_RAMP"

    def execute(self, context):
        global _selection_tracker_active

        entry = context.scene.skybrush.light_effects.active_entry
        try:
            edit_obj = context.edit_object
            if edit_obj is not None and edit_obj.type == "MESH":
                if _selection_tracker_active:
                    selection_order.unregister_selection_order_tracker()
                    _selection_tracker_active = False
                curve_obj = path_gradient.create_gradient_curve_from_selection()
            else:
                selected = [
                    obj
                    for obj in getattr(context, "selected_objects", [])
                    if obj is not None
                ]
                if len(selected) >= 2:
                    if not _selection_tracker_active:
                        selection_order.register_selection_order_tracker()
                        _selection_tracker_active = True
                        self.report(
                            {'INFO'},
                            "Selection order tracking enabled. Select objects in the"
                            " desired order and run the operator again to create the"
                            " curve.",
                        )
                        return {'FINISHED'}

                    try:
                        ordered = selection_order.get_ordered_selected_objects()
                        if len(ordered) < 2:
                            raise ValueError(
                                "Select at least two objects while order tracking is"
                                " active."
                            )
                        curve_obj = path_gradient.create_gradient_curve_from_objects(
                            ordered
                        )
                    finally:
                        selection_order.unregister_selection_order_tracker()
                        _selection_tracker_active = False
                else:
                    active_obj = context.active_object
                    if _selection_tracker_active:
                        selection_order.unregister_selection_order_tracker()
                        _selection_tracker_active = False
                    if active_obj is None or active_obj.type != "CURVE":
                        raise ValueError(
                            "Select a mesh in edit mode, multiple objects, or a curve object."
                        )
                    curve_obj = path_gradient.ensure_gradient_geometry(active_obj)
        except Exception as exc:  # pragma: no cover - Blender UI
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        entry.mesh = curve_obj
        entry.target = "INSIDE_MESH"
        if hasattr(entry, "output"):
            try:
                entry.output = OUTPUT_VERTEX_COLOR
            except Exception:
                pass

        self.report({'INFO'}, f"Generated curve mesh '{curve_obj.name}'")
        return {'FINISHED'}


class CreateBoundingBoxMeshOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.create_bb_mesh"
    bl_label = "Create BB Mesh"
    bl_description = "Create a bounding box mesh covering the selected objects"

    assign_mode: EnumProperty(
        name="Assignment Mode",
        items=[
            ("AUTO", "Auto", "Automatically assign based on the active outputs", 0),
            ("MESH", "Target Mesh", "Assign the created mesh to the target mesh property", 1),
            ("UV", "UV Mesh", "Assign the created mesh to the UV mesh property", 2),
        ],
        default="AUTO",
        options={'HIDDEN'},
    )
    divisions: IntProperty(
        name="Subdivisions",
        description="Legacy subdivision count stored for compatibility",
        default=3,
        min=1,
    )
    thickness: FloatProperty(
        name="Thickness",
        description="Total thickness applied by the solidify modifier",
        default=0.2,
        min=0.0,
        options={'HIDDEN'},
    )
    centered: BoolProperty(
        name="Centered Thickness",
        description="When enabled, solidify expands equally above and below the plane",
        default=True,
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            return False
        skybrush = getattr(scene, "skybrush", None)
        if skybrush is None:
            return False
        light_effects = getattr(skybrush, "light_effects", None)
        if not light_effects:
            return False
        return getattr(light_effects, "active_entry", None) is not None

    def invoke(self, context, event):  # pragma: no cover - Blender UI
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        if entry is not None:
            stored = None
            for key in ("sample_uv_divisions", "uv_sample_divisions", "_uv_sample_divisions"):
                value = entry.get(key) if hasattr(entry, "get") else None
                if isinstance(value, (int, float)):
                    stored = int(value)
                    break
            if stored is not None and stored > 0:
                self.divisions = stored
        return self.execute(context)

    def _get_reference_objects(self, context):
        selected = [obj for obj in getattr(context, "selected_objects", []) if obj]
        if selected:
            return selected
        active_obj = getattr(context, "active_object", None)
        return [active_obj] if active_obj is not None else []

    def _compute_bounds(self, objects):
        if not objects:
            return Vector((1.0, 1.0, 1.0)), Matrix.Identity(4)

        if len(objects) == 1:
            obj = objects[0]
            dims = Vector(getattr(obj, "dimensions", (1.0, 1.0, 1.0)))
            dims = Vector((max(float(d), 1e-4) for d in dims))
            bbox = getattr(obj, "bound_box", None)
            center_world = obj.matrix_world.translation
            if bbox:
                try:
                    corners = [Vector(corner) for corner in bbox]
                    if corners:
                        center_local = sum(corners, Vector()) / float(len(corners))
                        center_world = obj.matrix_world @ center_local
                except Exception:
                    pass
            rotation = obj.matrix_world.to_3x3()
            try:
                rotation = rotation.normalized()
            except Exception:
                rotation = Matrix.Identity(3)
            transform = Matrix.Translation(center_world) @ rotation.to_4x4()
            return dims, transform

        min_corner = Vector((float("inf"), float("inf"), float("inf")))
        max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
        have_corner = False

        for obj in objects:
            bbox = getattr(obj, "bound_box", None)
            corners = []
            if bbox:
                try:
                    corners = [obj.matrix_world @ Vector(corner) for corner in bbox]
                except Exception:
                    corners = []
            if not corners:
                dims = Vector(getattr(obj, "dimensions", (1.0, 1.0, 1.0)))
                dims = Vector((max(float(d), 1e-4) for d in dims))
                hx, hy, hz = (dims.x * 0.5, dims.y * 0.5, dims.z * 0.5)
                local_corners = [
                    Vector((sx * hx, sy * hy, sz * hz))
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                ]
                corners = [obj.matrix_world @ corner for corner in local_corners]

            for corner in corners:
                have_corner = True
                for i in range(3):
                    min_corner[i] = min(min_corner[i], corner[i])
                    max_corner[i] = max(max_corner[i], corner[i])

        if not have_corner:
            obj = objects[0]
            dims = Vector(getattr(obj, "dimensions", (1.0, 1.0, 1.0)))
            dims = Vector((max(float(d), 1e-4) for d in dims))
            center_world = obj.matrix_world.translation
            rotation = obj.matrix_world.to_3x3()
            try:
                rotation = rotation.normalized()
            except Exception:
                rotation = Matrix.Identity(3)
            transform = Matrix.Translation(center_world) @ rotation.to_4x4()
            return dims, transform

        center_world = (min_corner + max_corner) * 0.5
        dims = max_corner - min_corner
        dims = Vector((max(float(d), 1e-4) for d in dims))
        transform = Matrix.Translation(center_world)
        return dims, transform

    def _build_delaunay_object(self, entry, ref_objs):
        positions = delaunay.get_evaluated_world_positions(ref_objs)
        if len(positions) < 3:
            raise ValueError("Select at least three objects for Delaunay triangulation.")

        mesh = delaunay.build_planar_mesh_from_points(positions)
        coords = [Vector(v.co) for v in mesh.vertices]
        center = Vector((0.0, 0.0, 0.0))
        if coords:
            center = sum(coords, Vector((0.0, 0.0, 0.0))) / float(len(coords))
            for vert in mesh.vertices:
                vert.co -= center
        mesh.update()

        data_name = pick_unique_name(f"{entry.name}_Plane", bpy.data.meshes)
        mesh.name = data_name
        obj_name = pick_unique_name(f"{entry.name}_bb_mesh", bpy.data.objects)
        mesh_obj = bpy.data.objects.new(obj_name, mesh)
        mesh_obj.matrix_world = Matrix.Translation(center)

        if hasattr(mesh_obj, "display_type"):
            mesh_obj.display_type = 'BOUNDS'
        elif hasattr(mesh_obj, "display"):
            mesh_obj.display = 'BOUNDS'
        if hasattr(mesh_obj, "hide_render"):
            mesh_obj.hide_render = True

        solid = mesh_obj.modifiers.new("SolidifyY", type='SOLIDIFY')
        solid.thickness = max(float(self.thickness), 0.0)
        solid.offset = 0.0 if self.centered else 1.0
        solid.use_rim = True
        solid.use_even_offset = True
        for attr in ("use_quality_normals", "nonmanifold_boundary_mode"):
            if hasattr(solid, attr):
                try:
                    current = getattr(solid, attr)
                    setattr(solid, attr, True if isinstance(current, bool) else current)
                except Exception:
                    pass

        return mesh_obj

    def _build_cube_mesh(self, dims):
        hx, hy, hz = (dims.x * 0.5, dims.y * 0.5, dims.z * 0.5)
        verts = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]
        bm = bmesh.new()
        try:
            bm_verts = [bm.verts.new(Vector(co)) for co in verts]
            bm.verts.ensure_lookup_table()
            for face in faces:
                try:
                    bm.faces.new([bm_verts[i] for i in face])
                except ValueError:
                    pass
            bm.faces.ensure_lookup_table()

            uv_layer = bm.loops.layers.uv.new("UVMap")
            areas = {
                0: dims.y * dims.z,
                1: dims.x * dims.z,
                2: dims.x * dims.y,
            }
            axis = max(areas, key=areas.get)
            axis_to_uv = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
            u_axis, v_axis = axis_to_uv[axis]
            mins = [-hx, -hy, -hz]
            maxs = [hx, hy, hz]
            ranges = [maxs[i] - mins[i] if maxs[i] - mins[i] > 1e-6 else 1.0 for i in range(3)]
            for face in bm.faces:
                for loop in face.loops:
                    co = loop.vert.co
                    u = (co[u_axis] - mins[u_axis]) / ranges[u_axis]
                    v = (co[v_axis] - mins[v_axis]) / ranges[v_axis]
                    loop[uv_layer].uv = (u, v)

            mesh_name = pick_unique_name("BoundingBoxMesh", bpy.data.meshes)
            mesh = bpy.data.meshes.new(mesh_name)
            bm.to_mesh(mesh)
            mesh.update()
        finally:
            bm.free()
        return mesh

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        ref_objs = self._get_reference_objects(context)
        if not ref_objs:
            self.report({'WARNING'}, "Select one or more objects to define the mesh bounds.")
            return {'CANCELLED'}

        mode = getattr(self, "assign_mode", "AUTO")
        assign_to_uv = False
        assign_to_mesh = False
        if mode == "UV":
            assign_to_uv = True
        elif mode == "MESH":
            assign_to_mesh = True
        else:
            if hasattr(entry, "uv_mesh"):
                try:
                    assign_to_uv = (
                        entry.output in {OUTPUT_MESH_UV_U, OUTPUT_MESH_UV_V}
                        or entry.output_y in {OUTPUT_MESH_UV_U, OUTPUT_MESH_UV_V}
                    )
                except Exception:
                    assign_to_uv = False

        mesh_obj = None
        delaunay_error = None
        delaunay_used = False
        if not assign_to_uv:
            try:
                mesh_obj = self._build_delaunay_object(entry, ref_objs)
                delaunay_used = True
            except Exception as exc:
                delaunay_error = exc

        if mesh_obj is None:
            dims, transform = self._compute_bounds(ref_objs)
            mesh = self._build_cube_mesh(dims)

            obj_name = pick_unique_name(f"{entry.name}_bb_mesh", bpy.data.objects)
            mesh_obj = bpy.data.objects.new(obj_name, mesh)
            mesh_obj.matrix_world = transform
            if hasattr(mesh_obj, "display_type"):
                mesh_obj.display_type = 'BOUNDS'
            elif hasattr(mesh_obj, "display"):
                mesh_obj.display = 'BOUNDS'
            if hasattr(mesh_obj, "hide_render"):
                mesh_obj.hide_render = True
            if delaunay_error is not None:
                self.report(
                    {'WARNING'},
                    f"Falling back to bounding box mesh: {delaunay_error}",
                )

        scene = context.scene
        scene.collection.objects.link(mesh_obj)

        for modifier in list(getattr(mesh_obj, "modifiers", [])):
            if modifier.type == 'NODES' and modifier.name == "Sample UV Subdivide":
                mesh_obj.modifiers.remove(modifier)

        if hasattr(mesh_obj, "display_type"):
            mesh_obj.display_type = 'BOUNDS'
        elif hasattr(mesh_obj, "display"):
            mesh_obj.display = 'BOUNDS'
        if hasattr(mesh_obj, "hide_render"):
            mesh_obj.hide_render = True
        if assign_to_uv:
            try:
                entry.uv_mesh = mesh_obj
            except Exception:
                pass
        elif assign_to_mesh and hasattr(entry, "mesh"):
            entry.mesh = mesh_obj
        if hasattr(entry, "__setitem__"):
            try:
                entry["sample_uv_divisions"] = int(self.divisions)
            except Exception:
                pass

        if delaunay_used:
            message = f"Created Delaunay mesh '{mesh_obj.name}'"
        else:
            message = f"Created bounding box mesh '{mesh_obj.name}'"
        self.report({'INFO'}, message)
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


class CreateTargetCollectionFromSelectionOperator(bpy.types.Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.create_target_collection"
    bl_label = "Create Target Collection"
    bl_description = "Create a collection from the selected objects and assign it"

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        if entry is None or entry.target != "COLLECTION":
            return False
        selected = getattr(context, "selected_objects", [])
        return any(obj is not None for obj in selected)

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        selected = [
            obj for obj in getattr(context, "selected_objects", []) if obj is not None
        ]
        if not selected:
            self.report({'WARNING'}, "Select one or more objects to add to the new collection.")
            return {'CANCELLED'}

        base_name = f"{entry.name}_targets"
        collection_name = pick_unique_name(base_name, bpy.data.collections)
        new_collection = bpy.data.collections.new(collection_name)
        context.scene.collection.children.link(new_collection)

        for obj in selected:
            if obj.name in new_collection.objects:
                continue
            try:
                new_collection.objects.link(obj)
            except RuntimeError:
                pass

        entry.target_collection = new_collection
        self.report({'INFO'}, f"Created collection '{new_collection.name}'")
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
        if meta.get("type") == "ARRAY":
            if name not in pg:
                _set_id_property(pg, name, list(meta.get("default", [])))
            else:
                _set_id_property(pg, name, _sanitize_array_value(pg.get(name), meta))
            pg.ensure_dynamic_array(name, meta)
            continue
        if meta.get("type") == "COLOR_RAMP":
            if name not in pg:
                _set_id_property(
                    pg, name, _convert_color_ramp_points(meta.get("default", []))
                )
            else:
                _set_id_property(pg, name, _convert_color_ramp_points(pg.get(name)))
            pg.ensure_dynamic_color_ramp(name, meta)
            continue
        if name not in pg:
            default = meta.get("default") if hasattr(meta, "get") else meta["default"]
            _set_id_property(pg, name, default)
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


def draw_dynamic_array(pg, layout, name: str, meta: dict) -> None:
    array = pg.ensure_dynamic_array(name, meta)
    box = layout.box()
    header = box.row(align=True)
    header.label(text=name.upper())
    add_op = header.operator(
        "skybrush.dynamic_array_item_add", text="", icon="ADD"
    )
    add_op.property_name = name
    if not array.values:
        box.label(text="No values", icon="INFO")
        return
    for idx, item in enumerate(array.values):
        row = box.row(align=True)
        label = f"{idx}"
        if array.item_type == "FLOAT":
            row.prop(item, "value_float", text=label)
        elif array.item_type == "VECTOR":
            row.prop(item, "vector", text=label)
        elif array.item_type == "EULER":
            row.prop(item, "rotation", text=label)
        elif array.item_type == "SCALE":
            row.prop(item, "scale", text=label)
        elif array.item_type == "COLOR":
            row.prop(item, "color", text="")
        elif array.item_type == "COLOR_RGB":
            row.prop(item, "color_rgb", text="")
        else:
            row.prop(item, "value_float", text=label)
        remove = row.operator(
            "skybrush.dynamic_array_item_remove", text="", icon="X"
        )
        remove.property_name = name
        remove.index = idx


def draw_dynamic_color_ramp(pg, layout, name: str, meta: dict) -> None:
    ramp = pg.ensure_dynamic_color_ramp(name, meta)
    box = layout.box()
    header = box.row(align=True)
    header.label(text=name.upper())
    add_op = header.operator(
        "skybrush.dynamic_color_ramp_point_add", text="", icon="ADD"
    )
    add_op.property_name = name
    if not ramp.points:
        box.label(text="No points", icon="INFO")
        return
    for idx, point in enumerate(ramp.points):
        row = box.row(align=True)
        row.prop(point, "position", text=f"{idx}")
        row.prop(point, "color", text="")
        remove = row.operator(
            "skybrush.dynamic_color_ramp_point_remove", text="", icon="X"
        )
        remove.property_name = name
        remove.index = idx


def draw_dynamic(pg, layout, schema):
    """Draw ID properties from ``pg`` on ``layout``."""
    for name, meta in schema.items():
        if meta.get("type") == "ARRAY":
            draw_dynamic_array(pg, layout, name, meta)
        elif meta.get("type") == "COLOR_RAMP":
            draw_dynamic_color_ramp(pg, layout, name, meta)
        elif meta.get("type") == "OBJECT":
            layout.prop_search(pg, f'["{name}"]', bpy.data, "objects", text=name.upper())
        else:
            layout.prop(pg, f'["{name}"]', text=name.upper())


class AddDynamicArrayItemOperator(Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.dynamic_array_item_add"
    bl_label = "Add Array Item"

    property_name: StringProperty(name="Property Name", default="")

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry is not None and entry.type == "FUNCTION"

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        schema = get_state(entry).get("config_schema", {})
        meta = schema.get(self.property_name) or _as_dict(
            entry.get("_config_schema", {})
        ).get(self.property_name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        array = entry.ensure_dynamic_array(self.property_name, meta)
        array.append_default(meta)
        entry[self.property_name] = array.to_storage_list()
        return {'FINISHED'}


class RemoveDynamicArrayItemOperator(Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.dynamic_array_item_remove"
    bl_label = "Remove Array Item"

    property_name: StringProperty(name="Property Name", default="")
    index: IntProperty(name="Index", default=0, min=0)

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry is not None and entry.type == "FUNCTION"

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        schema = get_state(entry).get("config_schema", {})
        meta = schema.get(self.property_name) or _as_dict(
            entry.get("_config_schema", {})
        ).get(self.property_name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        array = entry.ensure_dynamic_array(self.property_name, meta)
        if 0 <= self.index < len(array.values):
            array.values.remove(self.index)
            entry[self.property_name] = array.to_storage_list()
            return {'FINISHED'}
        return {'CANCELLED'}


class AddDynamicColorRampPointOperator(Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.dynamic_color_ramp_point_add"
    bl_label = "Add Color Ramp Point"

    property_name: StringProperty(name="Property Name", default="")

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry is not None and entry.type == "FUNCTION"

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        schema = get_state(entry).get("config_schema", {})
        meta = schema.get(self.property_name) or _as_dict(
            entry.get("_config_schema", {})
        ).get(self.property_name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        ramp = entry.ensure_dynamic_color_ramp(self.property_name, meta)
        default_point = meta.get(
            "default_point", {"position": 0.5, "color": [1.0, 1.0, 1.0, 1.0]}
        )
        ramp.append_point(default_point.get("position", 0.5), default_point.get("color", [1.0, 1.0, 1.0, 1.0]))
        entry[self.property_name] = ramp.to_storage_list()
        return {'FINISHED'}


class RemoveDynamicColorRampPointOperator(Operator):  # pragma: no cover - Blender UI
    bl_idname = "skybrush.dynamic_color_ramp_point_remove"
    bl_label = "Remove Color Ramp Point"

    property_name: StringProperty(name="Property Name", default="")
    index: IntProperty(name="Index", default=0, min=0)

    @classmethod
    def poll(cls, context):
        entry = getattr(context.scene.skybrush.light_effects, "active_entry", None)
        return entry is not None and entry.type == "FUNCTION"

    def execute(self, context):
        entry = context.scene.skybrush.light_effects.active_entry
        schema = get_state(entry).get("config_schema", {})
        meta = schema.get(self.property_name) or _as_dict(
            entry.get("_config_schema", {})
        ).get(self.property_name, {})
        if hasattr(meta, "items") and not isinstance(meta, dict):
            meta = _as_dict(meta)
        ramp = entry.ensure_dynamic_color_ramp(self.property_name, meta)
        if 0 <= self.index < len(ramp.points):
            ramp.remove_point(self.index)
            entry[self.property_name] = ramp.to_storage_list()
            return {'FINISHED'}
        return {'CANCELLED'}


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
            if not getattr(entry, "sequence_mode", False):
                col.prop(entry, "frame_end")
            col.separator()
            col.prop(entry, "fade_in_duration")
            col.prop(entry, "fade_out_duration")
            col.separator()
            col.prop(entry, "sequence_mode")
            if getattr(entry, "sequence_mode", False):
                col.prop(entry, "sequence_mask_collection", text="Mask Collection")
                col.prop(entry, "sequence_duration")
                if not entry.sequence_manual_delay:
                    col.prop(entry, "sequence_delay")
                col.prop(entry, "sequence_manual_delay")
                meshes = entry.get_sequence_meshes()
                entry.ensure_sequence_delay_entries(len(meshes) - 1)
                if entry.sequence_manual_delay and len(meshes) > 1:
                    for idx in range(len(meshes) - 1):
                        delay_entry = entry.sequence_delays[idx]
                        text = f"{meshes[idx].name} → {meshes[idx + 1].name}"
                        col.prop(delay_entry, "delay", text=text)
            else:
                row_mesh = col.row(align=True)
                row_mesh.prop(entry, "mesh")
                op_mesh = row_mesh.operator(
                    CreateBoundingBoxMeshOperator.bl_idname,
                    text="",
                    icon="MESH_CUBE",
                )
                op_mesh.assign_mode = "MESH"
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
                if (
                    hasattr(entry, "uv_mesh")
                    and entry.output in {OUTPUT_MESH_UV_U, OUTPUT_MESH_UV_V}
                ):
                    row_uv = col.row(align=True)
                    row_uv.prop(entry, "uv_mesh")
                    op_uv = row_uv.operator(
                        CreateBoundingBoxMeshOperator.bl_idname,
                        text="",
                        icon="MESH_CUBE",
                    )
                    op_uv.assign_mode = "UV"
            if output_type_supports_mapping_mode(entry.output):
                col.prop(entry, "output_mapping_mode")
            if entry.type == "IMAGE":
                col.prop(entry, "output_y")
                if entry.output_y == "CUSTOM":
                    col.prop(entry.output_function_y, "path", text="Fn file")
                    col.prop(entry.output_function_y, "name", text="Fn name")
                if (
                    hasattr(entry, "uv_mesh")
                    and entry.output_y in {OUTPUT_MESH_UV_U, OUTPUT_MESH_UV_V}
                ):
                    row_uv_y = col.row(align=True)
                    row_uv_y.prop(entry, "uv_mesh")
                    op_uv_y = row_uv_y.operator(
                        CreateBoundingBoxMeshOperator.bl_idname,
                        text="",
                        icon="MESH_CUBE",
                    )
                    op_uv_y.assign_mode = "UV"
                col.prop(entry, "convert_srgb")
            if output_type_supports_mapping_mode(entry.output_y):
                col.prop(entry, "output_mapping_mode_y")
            col.prop(entry, "target")
            col.prop(entry, "invert_target")
            if entry.target == "COLLECTION":
                row_target = col.row(align=True)
                row_target.prop(entry, "target_collection")
                row_target.operator(
                    CreateTargetCollectionFromSelectionOperator.bl_idname,
                    text="",
                    icon="OUTLINER_COLLECTION",
                )
                col.operator(
                    ConvertCollectionToMeshOperator.bl_idname,
                    text="Convert to Mesh",
                )
            col.prop(entry, "blend_mode")
            col.prop(entry, "influence", slider=True)

            if effect_type_supports_randomization(entry.type):
                col.prop(entry, "randomness", slider=True)

def patched_object_has_mesh_data(self, obj) -> bool:
    return True

def patch_light_effects_panel():
    LightEffectsPanel._original_draw = LightEffectsPanel.draw
    LightEffectsPanel.draw = PatchedLightEffectsPanel.draw
    sbstudio.plugin.model.light_effects._original_object_has_mesh_data = sbstudio.plugin.model.light_effects.object_has_mesh_data
    sbstudio.plugin.model.light_effects.object_has_mesh_data = patched_object_has_mesh_data


def unpatch_light_effects_panel():
    LightEffectsPanel.draw = LightEffectsPanel._original_draw
    LightEffectsPanel._original_draw = None
    sbstudio.plugin.model.light_effects.object_has_mesh_data = sbstudio.plugin.model.light_effects._original_object_has_mesh_data
    sbstudio.plugin.model.light_effects._original_object_has_mesh_data = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register():  # pragma: no cover - executed in Blender
    global _selection_tracker_active
    _selection_tracker_active = False

    bpy.utils.register_class(SequenceDelayEntry)
    bpy.utils.register_class(DynamicArrayValue)
    bpy.utils.register_class(DynamicArrayProperty)
    bpy.utils.register_class(DynamicColorRampPoint)
    bpy.utils.register_class(DynamicColorRamp)
    bpy.utils.register_class(AddDynamicArrayItemOperator)
    bpy.utils.register_class(RemoveDynamicArrayItemOperator)
    bpy.utils.register_class(AddDynamicColorRampPointOperator)
    bpy.utils.register_class(RemoveDynamicColorRampPointOperator)
    bpy.utils.register_class(EmbedColorFunctionOperator)
    bpy.utils.register_class(UnembedColorFunctionOperator)
    bpy.utils.register_class(BakeColorRampOperator)
    bpy.utils.register_class(GeneratePathGradientMeshOperator)
    bpy.utils.register_class(CreateBoundingBoxMeshOperator)
    bpy.utils.register_class(CreateTargetCollectionFromSelectionOperator)
    bpy.utils.register_class(ConvertCollectionToMeshOperator)
    bpy.utils.register_class(BakeColorRampSplitOperator)
    bpy.utils.register_class(MergeSplitLightEffectsOperator)
    if _ensure_light_effects_initialized not in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.append(_ensure_light_effects_initialized)


def unregister():  # pragma: no cover - executed in Blender
    global _selection_tracker_active
    if _selection_tracker_active:
        selection_order.unregister_selection_order_tracker()
        _selection_tracker_active = False

    if _ensure_light_effects_initialized in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.remove(_ensure_light_effects_initialized)
    bpy.utils.unregister_class(ConvertCollectionToMeshOperator)
    bpy.utils.unregister_class(CreateTargetCollectionFromSelectionOperator)
    bpy.utils.unregister_class(CreateBoundingBoxMeshOperator)
    bpy.utils.unregister_class(GeneratePathGradientMeshOperator)
    bpy.utils.unregister_class(BakeColorRampOperator)
    bpy.utils.unregister_class(UnembedColorFunctionOperator)
    bpy.utils.unregister_class(EmbedColorFunctionOperator)
    bpy.utils.unregister_class(BakeColorRampSplitOperator)
    bpy.utils.unregister_class(MergeSplitLightEffectsOperator)
    bpy.utils.unregister_class(RemoveDynamicColorRampPointOperator)
    bpy.utils.unregister_class(AddDynamicColorRampPointOperator)
    bpy.utils.unregister_class(RemoveDynamicArrayItemOperator)
    bpy.utils.unregister_class(AddDynamicArrayItemOperator)
    bpy.utils.unregister_class(DynamicColorRamp)
    bpy.utils.unregister_class(DynamicColorRampPoint)
    bpy.utils.unregister_class(DynamicArrayProperty)
    bpy.utils.unregister_class(DynamicArrayValue)
    bpy.utils.unregister_class(SequenceDelayEntry)

