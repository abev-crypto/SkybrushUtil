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


def _assign_drones_vertex_group(obj, group_name="Drones"):
    """Ensure ``group_name`` exists and covers all vertices on ``obj``."""

    if obj.type != "MESH" or obj.data is None:
        return

    vertex_group = obj.vertex_groups.get(group_name)
    if vertex_group is None:
        vertex_group = obj.vertex_groups.new(name=group_name)

    indices = [vertex.index for vertex in obj.data.vertices]
    if indices:
        vertex_group.add(indices, 1.0, "REPLACE")

    skybrush = getattr(obj, "skybrush", None)
    if skybrush is not None:
        skybrush.formation_vertex_group = group_name


def _prepare_selected_meshes(context):
    for obj in context.selected_objects:
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
