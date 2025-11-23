"""Patches for storyboard transition recalculation.

This module monkey-patches the Skybrush Studio transition recalculation logic
and adds support for storing a recognized-point mapping on storyboard entries.
"""

from __future__ import annotations

from functools import partial
import json
from typing import Callable, List, Mapping, Optional

try:  # pragma: no cover - Blender dependency
    import bpy
    from bpy.props import StringProperty
    from bpy.types import Context
except Exception:  # pragma: no cover - allows import without Blender
    bpy = None
    StringProperty = None
    Context = None

try:  # pragma: no cover - Skybrush Studio dependency
    from sbstudio.plugin.model.storyboard import StoryboardEntry
    from sbstudio.plugin.operators import recalculate_transitions as _rct
except Exception:  # pragma: no cover - optional dependency
    StoryboardEntry = None
    _rct = None

__all__ = (
    "patch_recalculate_transitions",
    "unpatch_recalculate_transitions",
)


def _handle_recognized_point_mapping_change(
    self: StoryboardEntry, context: Optional[Context] = None
):  # pragma: no cover - Blender callback
    self._invalidate_decoded_recognized_point_mapping()


def _install_recognized_point_mapping_property():
    if StoryboardEntry is None or StringProperty is None:
        return

    if getattr(StoryboardEntry, "recognized_point_mapping", None) is not None:
        return

    StoryboardEntry.recognized_point_mapping = StringProperty(
        name="Mapping by recognized points",
        description=(
            "Mapping where the i-th element is the index of the drone that "
            "was matched to formation point i as recognized in the formation"
            ". Unmatched points are stored as null."
        ),
        default="",
        options={"HIDDEN"},
        update=_handle_recognized_point_mapping_change,
    )

    StoryboardEntry._decoded_recognized_point_mapping = None
    StoryboardEntry.__annotations__ = getattr(StoryboardEntry, "__annotations__", {})
    StoryboardEntry.__annotations__["recognized_point_mapping"] = str
    StoryboardEntry.__annotations__["_decoded_recognized_point_mapping"] = Optional[Mapping]

    def get_recognized_point_mapping(self: StoryboardEntry) -> Optional[Mapping]:
        """Returns the mapping of formation points to drone indices for this entry.
        The i-th element of the returned list contains the index of the drone
        that was mapped to the i-th recognized formation point, or ``None`` if
        the point is not used. Returns ``None`` if there is no mapping yet.
        """

        if self._decoded_recognized_point_mapping is None:
            encoded_mapping = self.recognized_point_mapping.strip()
            if (
                not encoded_mapping
                or len(encoded_mapping) < 2
                or encoded_mapping[0] != "["
                or encoded_mapping[-1] != "]"
            ):
                return None
            else:
                self._decoded_recognized_point_mapping = json.loads(encoded_mapping)

        return self._decoded_recognized_point_mapping

    def update_recognized_point_mapping(
        self: StoryboardEntry, mapping: Optional[Mapping]
    ) -> None:
        """Updates the mapping of formation points to drones for the storyboard entry.
        Arguments:
            mapping: list where the i-th item contains the index of the drone
                that was mapped to the i-th formation point, or ``None`` if the
                point is unused. You can also pass ``None`` to clear the mapping.
        """

        if mapping is None:
            self.recognized_point_mapping = ""
        else:
            self.recognized_point_mapping = json.dumps(mapping)
        assert self._decoded_recognized_point_mapping is None

    def _invalidate_decoded_recognized_point_mapping(self: StoryboardEntry) -> None:
        self._decoded_recognized_point_mapping = None

    StoryboardEntry.get_recognized_point_mapping = get_recognized_point_mapping
    StoryboardEntry.update_recognized_point_mapping = update_recognized_point_mapping
    StoryboardEntry._invalidate_decoded_recognized_point_mapping = (
        _invalidate_decoded_recognized_point_mapping
    )


def _patched_update_transition_for_storyboard_entry(
    entry: _rct.StoryboardEntry,
    entry_index: int,
    drones,
    *,
    get_positions_of,
    previous_entry: Optional[_rct.StoryboardEntry],
    previous_mapping: Optional[Mapping],
    start_of_scene: int,
    start_of_next: Optional[int],
) -> Optional[Mapping]:  # pragma: no cover - depends on sbstudio/blender
    """Patched variant that records recognized point mappings."""

    SkybrushStudioError = _rct.SkybrushStudioError
    get_markers_and_related_objects_from_formation = (
        _rct.get_markers_and_related_objects_from_formation
    )
    calculate_mapping_for_transition_into_storyboard_entry = (
        _rct.calculate_mapping_for_transition_into_storyboard_entry
    )
    update_transition_constraint_properties = _rct.update_transition_constraint_properties
    calculate_departure_index_of_drone = _rct.calculate_departure_index_of_drone
    _LazyFormationTargetList = _rct._LazyFormationTargetList
    InfluenceCurveDescriptor = _rct.InfluenceCurveDescriptor

    if entry.is_locked:
        return None

    formation = entry.formation
    if formation is None:
        return None

    markers_and_objects = get_markers_and_related_objects_from_formation(formation)
    num_markers = len(markers_and_objects)
    end_of_previous = previous_entry.frame_end if previous_entry else start_of_scene

    if previous_entry:
        start_points = get_positions_of(drones, frame=end_of_previous)
    else:
        start_points = get_positions_of(
            (marker for marker, _ in markers_and_objects), frame=end_of_previous
        )
        if len(drones) != len(start_points):
            raise SkybrushStudioError(
                f"First formation has {len(start_points)} markers but the scene "
                f'contains {len(drones)} drones. Check the "Drones" collection '
                f"and the first formation for consistency."
            )

    mapping = calculate_mapping_for_transition_into_storyboard_entry(
        entry,
        start_points,
        num_targets=num_markers,
    )

    entry.update_mapping(mapping)

    recognized_point_mapping: Mapping = [None] * num_markers
    for drone_index, target_index in enumerate(mapping):
        if target_index is not None and 0 <= target_index < num_markers:
            recognized_point_mapping[target_index] = drone_index

    entry.update_recognized_point_mapping(recognized_point_mapping)

    num_drones_transitioning = sum(
        1 for target_index in mapping if target_index is not None
    )

    objects_in_formation = _LazyFormationTargetList(entry)
    objects_in_previous_formation = _LazyFormationTargetList(previous_entry)

    schedule_override_map = entry.get_enabled_schedule_override_map()

    todo: List[Callable[[], None]] = []
    for drone_index, drone in enumerate(drones):
        target_index = mapping[drone_index]
        if target_index is None:
            marker, obj = None, None
        else:
            marker, obj = markers_and_objects[target_index]

        constraint = update_transition_constraint_properties(drone, entry, marker, obj)

        if constraint is not None:
            windup_start_frame = end_of_previous
            start_frame = entry.frame_start
            departure_delay = 0
            arrival_delay = 0
            departure_index: Optional[int] = None

            if entry.is_staggered:
                departure_index = calculate_departure_index_of_drone(
                    drone,
                    drone_index,
                    previous_entry,
                    entry_index - 1,
                    previous_mapping,
                    objects_in_previous_formation,
                )
                arrival_index = objects_in_formation.find(marker)

                departure_delay = entry.pre_delay_per_drone_in_frames * departure_index
                arrival_delay = -entry.post_delay_per_drone_in_frames * (
                    num_drones_transitioning - arrival_index - 1
                )

            if schedule_override_map:
                if departure_index is None:
                    departure_index = calculate_departure_index_of_drone(
                        drone,
                        drone_index,
                        previous_entry,
                        entry_index - 1,
                        previous_mapping,
                        objects_in_previous_formation,
                    )

                override = schedule_override_map.get(departure_index)
                if override:
                    departure_delay = override.pre_delay
                    arrival_delay = -override.post_delay

            windup_start_frame += departure_delay
            start_frame += arrival_delay

            if previous_entry is None:
                start_frame = windup_start_frame = start_of_scene
            else:
                if windup_start_frame >= start_frame:
                    raise SkybrushStudioError(
                        f"Not enough time to plan staggered transition to "
                        f"formation {entry.name!r} at drone index {drone_index + 1} "
                        f"(1-based). Try decreasing departure or arrival delay "
                        f"or allow more time for the transition."
                    )

            descriptor = _rct.InfluenceCurveDescriptor(
                scene_start_frame=start_of_scene,
                windup_start_frame=windup_start_frame,
                start_frame=start_frame,
                end_frame=start_of_next,
            )

            todo.append(
                partial(
                    _rct.update_transition_constraint_influence,
                    drone,
                    constraint,
                    descriptor,
                )
            )

    for func in todo:
        func()

    return mapping


def patch_recalculate_transitions():
    """Apply patches for transition recalculation if sbstudio is available."""

    if _rct is None:
        return

    _install_recognized_point_mapping_property()

    if getattr(_rct, "_original_update_transition_for_storyboard_entry", None):
        return

    _rct._original_update_transition_for_storyboard_entry = (
        _rct.update_transition_for_storyboard_entry
    )
    _rct.update_transition_for_storyboard_entry = (
        _patched_update_transition_for_storyboard_entry
    )


def unpatch_recalculate_transitions():
    """Revert patches applied by :func:`patch_recalculate_transitions`."""

    if _rct is None:
        return

    original = getattr(_rct, "_original_update_transition_for_storyboard_entry", None)
    if original:
        _rct.update_transition_for_storyboard_entry = original
        _rct._original_update_transition_for_storyboard_entry = None
