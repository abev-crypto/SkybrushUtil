import bpy
from mathutils import Vector

# Optional import for debug image generation
try:  # pragma: no cover - Pillow may not be available
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


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
        print(current)

def apply_color_keys_to_nearest(location, keyframes_by_channel, available_objects, frame_offset=0, normalize_255=False):
    """Find the nearest object to ``location`` and apply color keyframes.

    ``keyframes_by_channel`` should map channel indices (0=R,1=G,2=B,3=A)
    to sequences of ``(frame, value)`` pairs.
    """
    nearest_obj = find_nearest_object(location, available_objects)
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

    available_objects.remove(nearest_obj)
    return available_objects


def apply_color_keys_from_key_data(
    key_entries,
    start_frame=0,
    collection_name="Drones",
    debug_image_path=None,
    debug_object=None,
):
    """Apply color keyframes for each entry using nearest drones.

    ``key_entries`` should be a list of dictionaries as produced by
    ``tracks_to_keydata`` in :mod:`CSV2Vertex`, containing the initial location
    of a track and its color keyframes. Keyframes are applied relative to
    ``start_frame`` and matched to the nearest objects found in the Blender
    collection named by ``collection_name``.

    When ``debug_image_path`` is provided, an image visualising the RGB values
    of all entries across time is written to that path.  When ``debug_object``
    is provided, the color keys from the first entry are also applied to that
    object's material for quick preview.
    """
    if not key_entries:
        return

    # Apply to nearest drones in the specified collection
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

    # Optionally apply keys to a provided debug object
    if debug_object and key_entries:
        mat = debug_object.active_material
        if mat is None:
            mat = bpy.data.materials.new(name="DebugColor")
            mat.use_nodes = True
            debug_object.data.materials.append(mat)
        base_socket = None
        for node in mat.node_tree.nodes:
            if node.inputs and node.inputs[0].type == "RGBA":
                base_socket = node.inputs[0]
                break
        if base_socket:
            _insert_color_keyframes(
                base_socket,
                key_entries[0]["keys"],
                frame_offset=start_frame,
                normalize_255=True,
            )

    # Optionally export RGB data as an image for debugging
    if debug_image_path and Image and key_entries:
        # Determine maximum frame across all entries
        max_frame = 0
        for entry in key_entries:
            for frames in entry["keys"].values():
                if frames:
                    max_frame = max(max_frame, int(max(f for f, _ in frames)))
        width = max_frame + 1
        height = len(key_entries)
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y, entry in enumerate(key_entries):
            # Prepare a color for each frame; default black
            row = [[0, 0, 0] for _ in range(width)]
            for ch in range(3):
                frames = entry["keys"].get(ch, [])
                if not frames:
                    continue
                frames = sorted(frames)
                for i, (f0, v0) in enumerate(frames):
                    f1, v1 = frames[i + 1] if i + 1 < len(frames) else (max_frame, v0)
                    f0_i, f1_i = int(round(f0)), int(round(f1))
                    for f in range(f0_i, f1_i + 1):
                        t = 0 if f1_i == f0_i else (f - f0) / (f1 - f0)
                        val = v0 + (v1 - v0) * t
                        row[f][ch] = int(max(0, min(255, val)))
            for x in range(width):
                pixels[x, y] = tuple(row[x])
        try:
            img.save(debug_image_path)
        except Exception:
            pass
