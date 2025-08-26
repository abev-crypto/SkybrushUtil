import bpy
import numpy as np
from scipy.spatial import cKDTree


def find_nearest_object(location, coords, tree, objects):
    """Return the object from ``objects`` nearest to ``location`` using a KD-tree."""
    if not objects:
        return None, None

    _, index = tree.query(np.asarray(location))
    return objects[index], index


def _insert_color_keyframes(base_socket, keyframes_by_channel, frame_offset=0, normalize_255=False):
    """Insert color keyframes on ``base_socket`` for all channels at once.

    ``keyframes_by_channel`` should map channel indices to sequences of
    ``(frame, value)`` pairs.  All frames across channels are combined and the
    values from channels without a key on a given frame are kept from their
    previous value.
    """

    channel_frames = {
        ch: {f: v for f, v in frames}
        for ch, frames in keyframes_by_channel.items()
    }
    all_frames = sorted({f for frames in channel_frames.values() for f in frames})
    current = list(base_socket.default_value)

    for frame in all_frames:
        for ch, frames in channel_frames.items():
            if frame in frames:
                val = frames[frame]
                val = val / 255.0 if normalize_255 else val
                current[ch] = val
        for idx, val in enumerate(current):
            base_socket.default_value[idx] = val
        base_socket.keyframe_insert("default_value", frame=frame + frame_offset)

def apply_color_keys_to_nearest(location, keyframes_by_channel, available_objects, frame_offset=0, normalize_255=False):
    """Find the nearest object to ``location`` and apply color keyframes.

    ``keyframes_by_channel`` should map channel indices (0=R,1=G,2=B,3=A)
    to sequences of ``(frame, value)`` pairs.
    """
    if not available_objects:
        return available_objects

    coords = np.array([obj.matrix_world.translation[:] for obj in available_objects])
    tree = cKDTree(coords)
    nearest_obj, index = find_nearest_object(location, coords, tree, available_objects)
    if not nearest_obj:
        return available_objects
    mat = nearest_obj.active_material
    if not mat or not mat.node_tree:
        return available_objects
    base_socket = None
    for node in mat.node_tree.nodes:
        if node.inputs and node.inputs[0].type == 'RGBA':
            base_socket = node.inputs[0]
            break
    if base_socket is None:
        return available_objects

    _insert_color_keyframes(
        base_socket,
        keyframes_by_channel,
        frame_offset=frame_offset,
        normalize_255=normalize_255,
    )

    del available_objects[index]
    return available_objects


def apply_color_keys_from_key_data(
    key_entries,
    start_frame=0,
    collection_name="Drones",
):
    """Apply color keyframes for each entry using nearest drones.

    ``key_entries`` should be a list of dictionaries as produced by
    ``tracks_to_keydata`` in :mod:`CSV2Vertex`, containing the initial location
    of a track and its color keyframes. Keyframes are applied relative to
    ``start_frame`` and matched to the nearest objects found in the Blender
    collection named by ``collection_name``.
    """
    if not key_entries:
        return

    drones_col = bpy.data.collections.get(collection_name)
    if drones_col:
        available = list(drones_col.objects)
        for entry in key_entries:
            available = apply_color_keys_to_nearest(
                entry["location"],
                entry["keys"],
                available,
                frame_offset=start_frame,
                normalize_255=True,
            )

