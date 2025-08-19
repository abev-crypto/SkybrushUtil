import bpy
from mathutils import Vector


def find_nearest_object(location, objects):
    """Return the object from ``objects`` nearest to ``location``."""
    nearest_obj = None
    min_dist = float('inf')
    for obj in objects:
        dist = (Vector(location) - obj.matrix_world.translation).length
        if dist < min_dist:
            min_dist = dist
            nearest_obj = obj
    return nearest_obj


def apply_color_keys_to_nearest(location, keyframes_by_channel, available_objects, frame_offset=0, normalize_255=False):
    """Find the nearest object to ``location`` and apply color keyframes.

    ``keyframes_by_channel`` should map channel indices (0=R,1=G,2=B,3=A)
    to sequences of ``(frame, value)`` pairs.
    """
    nearest_obj = find_nearest_object(location, available_objects)
    if not nearest_obj:
        return
    mat = nearest_obj.active_material
    if not mat or not mat.node_tree:
        return
    base_socket = None
    for node in mat.node_tree.nodes:
        if node.inputs and node.inputs[0].type == 'RGBA':
            base_socket = node.inputs[0]
            break
    if base_socket is None:
        return
    for channel, frames in keyframes_by_channel.items():
        for frame, value in frames:
            val = value / 255.0 if normalize_255 and value > 1.0 else value
            base_socket.default_value[channel] = val
            base_socket.keyframe_insert(
                "default_value", frame=frame + frame_offset, index=channel
            )
    available_objects.remove(nearest_obj)


def apply_color_keys_from_key_data(key_entries, start_frame=0, collection_name="Drones"):
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
    if not drones_col:
        return
    available = list(drones_col.objects)
    for entry in key_entries:
        apply_color_keys_to_nearest(
            entry["location"],
            entry["keys"],
            available,
            frame_offset=start_frame,
            normalize_255=True,
        )
