"""Patches for formation creation behavior.

This module ensures that selected meshes gain a ``Drones`` vertex group
covering all vertices before formations are generated.
"""

import bpy

from sbstudio.plugin.operators.create_formation import CreateFormationOperator

__all__ = (
    "patch_create_formation_operator",
    "unpatch_create_formation_operator",
)


def _remove_gn_sphere_proximity_modifier(obj):
    modifiers = getattr(obj, "modifiers", None)
    if modifiers is None:
        return

    for modifier in list(modifiers):
        if modifier.name == "GN_SphereProximity":
            modifiers.remove(modifier)


def _remove_skybrush_vertex_groups(obj):
    to_remove = [vg for vg in obj.vertex_groups if "Skybrush[" in vg.name]
    for vertex_group in to_remove:
        obj.vertex_groups.remove(vertex_group)


def _assign_drones_vertex_group(obj, group_name="Drones"):
    """Ensure ``group_name`` exists, locked and covering all vertices on ``obj``."""

    if obj.type != "MESH" or obj.data is None:
        return

    _remove_skybrush_vertex_groups(obj)

    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        vertex_group = obj.vertex_groups.new(name=group_name)

    indices = [vertex.index for vertex in obj.data.vertices]
    if indices:
        vertex_group.add(indices, 1.0, "REPLACE")

    if hasattr(vertex_group, "lock_weight"):
        vertex_group.lock_weight = True

    skybrush = getattr(obj, "skybrush", None)
    if skybrush is not None:
        skybrush.formation_vertex_group = group_name


def _is_in_formations_collection(obj, root_collection_name="Formations"):
    formations = bpy.data.collections.get(root_collection_name)
    if formations is None:
        return False

    def _collect_descendants(collection):
        yield collection
        for child in collection.children:
            yield from _collect_descendants(child)

    formations_collections = set(_collect_descendants(formations))

    try:
        return any(collection in formations_collections for collection in obj.users_collection)
    except AttributeError:
        return False


def _prepare_selected_meshes(context):
    for obj in list(context.selected_objects):
        if _is_in_formations_collection(obj):
            obj.select_set(False)
            continue
        _remove_gn_sphere_proximity_modifier(obj)
        _assign_drones_vertex_group(obj)


def _patched_execute_on_formation(self, formation, context):
    _prepare_selected_meshes(context)
    return self._original_execute_on_formation(formation, context)


def patch_create_formation_operator():
    if getattr(CreateFormationOperator, "_original_execute_on_formation", None):
        return
    CreateFormationOperator._original_execute_on_formation = (
        CreateFormationOperator.execute_on_formation
    )
    CreateFormationOperator.execute_on_formation = _patched_execute_on_formation


def unpatch_create_formation_operator():
    original = getattr(CreateFormationOperator, "_original_execute_on_formation", None)
    if original:
        CreateFormationOperator.execute_on_formation = original
        CreateFormationOperator._original_execute_on_formation = None
