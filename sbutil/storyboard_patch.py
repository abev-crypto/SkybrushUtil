"""Monkey patching helpers for storyboard entry removal."""

from __future__ import annotations

import bpy
from collections.abc import Iterable
from typing import Optional

__all__ = (
    "patch_storyboard_entry_removal",
    "unpatch_storyboard_entry_removal",
)

try:  # pragma: no cover - Blender-only dependency
    from sbstudio.plugin.operators import remove_storyboard_entry as _remove_storyboard_entry
except Exception:  # pragma: no cover - when sbstudio is unavailable
    _remove_storyboard_entry = None


def _collect_descendants(collection: bpy.types.Collection) -> Iterable[bpy.types.Collection]:
    yield collection
    for child in collection.children:
        yield from _collect_descendants(child)


def _unlink_or_remove_object(obj: bpy.types.Object, owned_collections: set[bpy.types.Collection]):
    try:
        user_collections = set(obj.users_collection)
    except Exception:
        user_collections = set()

    removable_collections = user_collections & owned_collections
    for collection in removable_collections:
        try:
            collection.objects.unlink(obj)
        except Exception:
            pass

    # Delete the object only when it is not referenced outside the owned hierarchy
    if not (user_collections - owned_collections):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _delete_collection_hierarchy(root: Optional[bpy.types.Collection]):
    if root is None:
        return

    owned_collections = set(_collect_descendants(root))

    for collection in list(owned_collections):
        for obj in list(collection.objects):
            _unlink_or_remove_object(obj, owned_collections)

    def _remove_collection_recursive(collection: bpy.types.Collection):
        for child in list(collection.children):
            _remove_collection_recursive(child)
        try:
            bpy.data.collections.remove(collection, do_unlink=True)
        except Exception:
            pass

    _remove_collection_recursive(root)


def _patched_remove_constraints_for_storyboard_entry(entry):
    original = getattr(_remove_storyboard_entry, "_original_remove_constraints_for_storyboard_entry", None)
    if original is None:
        return

    original(entry)

    collection = None
    for attr in ("formation_collection", "collection", "formation"):
        collection = getattr(entry, attr, None)
        if isinstance(collection, bpy.types.Collection):
            break
    else:
        collection = None

    _delete_collection_hierarchy(collection)


def patch_storyboard_entry_removal():
    if _remove_storyboard_entry is None:
        return

    if getattr(_remove_storyboard_entry, "_original_remove_constraints_for_storyboard_entry", None):
        return

    original = getattr(_remove_storyboard_entry, "remove_constraints_for_storyboard_entry", None)
    if original is None:
        return

    _remove_storyboard_entry._original_remove_constraints_for_storyboard_entry = original
    _remove_storyboard_entry.remove_constraints_for_storyboard_entry = _patched_remove_constraints_for_storyboard_entry


def unpatch_storyboard_entry_removal():
    if _remove_storyboard_entry is None:
        return

    original = getattr(_remove_storyboard_entry, "_original_remove_constraints_for_storyboard_entry", None)
    if original is not None:
        _remove_storyboard_entry.remove_constraints_for_storyboard_entry = original
        _remove_storyboard_entry._original_remove_constraints_for_storyboard_entry = None
