import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup, UIList
import csv, json, os, re, math, shutil, zipfile
from mathutils import Vector
import numpy as np

from sbutil import csv_vat_gn
from sbutil.light_effects import OUTPUT_VERTEX_COLOR, _normalize_float_sequence

from sbutil.color_key_utils import apply_color_keys_from_key_data
from sbutil.copyloc_utils import shape_copyloc_influence_curve

# ---------- Utilities ----------

PREFIX_MAP_FILENAME = "prefix_map.json"
DEFAULT_FOLDER_DURATION = 480


def _storyboard_name(base_name: str, meta: dict | None = None) -> str:
    """Return a storyboard entry name composed of ID and ``base_name`` if present."""

    meta = meta or {}
    try:
        meta_id = meta.get("id")
    except Exception:
        meta_id = None

    if meta_id is not None:
        return f"{meta_id}_{base_name}"
    return base_name

def detect_delimiter(sample_path):
    # Try to sniff; fall back to tab if header matches the sample
    try:
        with open(sample_path, "r", newline="") as f:
            head = f.read(2048)
        dialect = csv.Sniffer().sniff(head, delimiters=",\t; ")
        return dialect.delimiter
    except Exception:
        # Heuristic: if header contains tabs or 'Time [msec]\tx [m]' pattern, use '\t'
        if "\t" in head or re.search(r"Time\s*\[msec\]\s*\tx\s*\[m\]", head):
            return "\t"
        return ","  # default

def load_csv(path, delimiter="auto"):
    if delimiter == "auto":
        delimiter = detect_delimiter(path)
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize expected headers
        colmap = {k.strip(): k for k in reader.fieldnames}
        req = ["Time [msec]", "x [m]", "y [m]", "z [m]", "Red", "Green", "Blue"]
        for r in reader:
            row = {
                "t_ms": float(r[colmap["Time [msec]"]]),
                "x": float(r[colmap["x [m]"]]),
                "y": float(r[colmap["y [m]"]]),
                "z": float(r[colmap["z [m]"]]),
                "r": float(r[colmap["Red"]]),
                "g": float(r[colmap["Green"]]),
                "b": float(r[colmap["Blue"]]),
            }
            rows.append(row)
    # Ensure sorted by time
    rows.sort(key=lambda d: d["t_ms"])
    return rows

def ms_to_frame(ms, fps):
    return (ms / 1000.0) * fps


def parse_gap_value(value, fps):
    """Parse ``value`` into a frame count using ``fps``.

    The suffix ``s`` means seconds, ``f`` means frames (or when there is no
    suffix). Non-matching values resolve to zero to keep the existing schedule.
    """

    if not value:
        return 0
    m = re.match(r"(?i)^(?P<amount>\d+(?:\.\d+)?)(?P<unit>[sf]?)$", value.strip())
    if not m:
        return 0
    amount = float(m.group("amount"))
    unit = m.group("unit").lower()
    if unit == "s":
        return int(round(amount * fps))
    return int(round(amount))


def split_name_and_gap(folder_name, fps):
    """Return ``(base_name, gap_frames)`` parsed from ``folder_name``."""

    m = re.match(r"^(.*)_(\d+(?:\.\d+)?[sf]?)$", folder_name)
    if not m:
        return folder_name, 0
    base_name = m.group(1)
    gap_frames = parse_gap_value(m.group(2), fps)
    return base_name, gap_frames


def load_import_metadata(directory, report):
    """Return an import plan read from :data:`PREFIX_MAP_FILENAME`.

    The file is expected to contain a mapping from folder names to an object
    with the keys ``id`` (required), ``duration`` (optional), ``midlayer``
    (optional), ``middur`` (optional mid-pose duration), ``midpose`` (optional
    flag to disable mid-poses), ``fhandle`` (optional CopyLoc handle frames for
    the formation), and ``mhandle`` (optional CopyLoc handle frames for the
    mid-pose). ``startframe`` can be added either at the top level or per-entry
    to control the starting frame offset for storyboard placement.
    ``duration`` represents the transition duration leading into the formation.
    It defaults to :data:`DEFAULT_FOLDER_DURATION`, ``midlayer`` defaults to
    ``1``, ``middur`` defaults to ``1`` and ``midpose`` defaults to ``True``
    when missing, but these defaults can be overridden by top-level keys in the
    JSON.
    """

    mapping_path = os.path.join(directory, PREFIX_MAP_FILENAME)
    if not os.path.isfile(mapping_path):
        return {}, {"start_frame": None}

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        report({"WARNING"}, f"Could not read {PREFIX_MAP_FILENAME}: {exc}")
        return {}, {"start_frame": None}

    if not isinstance(data, dict):
        report(
            {"WARNING"},
            f"{PREFIX_MAP_FILENAME} must contain an object that maps keys to metadata",
        )
        return {}, {"start_frame": None}

    # Global defaults that can be overridden by top-level keys
    default_start_frame = None
    if "startframe" in data and not isinstance(data["startframe"], dict):
        try:
            default_start_frame = int(data["startframe"])
        except Exception:
            report({"WARNING"}, "Invalid top-level startframe in prefix_map.json, using default")

    default_duration = DEFAULT_FOLDER_DURATION
    if "duration" in data and not isinstance(data["duration"], dict):
        try:
            default_duration = int(data["duration"])
        except Exception:
            report({"WARNING"}, "Invalid top-level duration in prefix_map.json, using default")

    default_midlayer = 1
    if "midlayer" in data and not isinstance(data["midlayer"], dict):
        try:
            default_midlayer = max(1, int(data["midlayer"]))
        except Exception:
            report({"WARNING"}, "Invalid top-level midlayer in prefix_map.json, using default")
    default_middur = 1
    if "middur" in data and not isinstance(data["middur"], dict):
        try:
            default_middur = max(1, int(data["middur"]))
        except Exception:
            report({"WARNING"}, "Invalid top-level middur in prefix_map.json, using default")
    default_midpose = True
    if "midpose" in data and not isinstance(data["midpose"], dict):
        default_midpose = bool(data["midpose"])
    default_ydepth = None
    if "ydepth" in data and not isinstance(data["ydepth"], dict):
        try:
            val = float(data["ydepth"])
            default_ydepth = max(0.0, val)
        except Exception:
            report({"WARNING"}, "Invalid top-level ydepth in prefix_map.json, using default")
    default_fhandle = 5.0
    if "fhandle" in data and not isinstance(data["fhandle"], dict):
        try:
            default_fhandle = float(data["fhandle"])
        except Exception:
            report({"WARNING"}, "Invalid top-level fhandle in prefix_map.json, using default")
    default_mhandle = 5.0
    if "mhandle" in data and not isinstance(data["mhandle"], dict):
        try:
            default_mhandle = float(data["mhandle"])
        except Exception:
            report({"WARNING"}, "Invalid top-level mhandle in prefix_map.json, using default")
    default_traled = False
    if "traled" in data and not isinstance(data["traled"], dict):
        default_traled = bool(data["traled"])
    default_tracolor = None
    if "tracolor" in data and not isinstance(data["tracolor"], dict):
        default_tracolor = str(data["tracolor"])
    default_ledsubdur = 0
    if "ledsubdur" in data and not isinstance(data["ledsubdur"], dict):
        default_ledsubdur = str(data["ledsubdur"])
    default_ledfifo = 0
    if "ledfifo" in data and not isinstance(data["ledfifo"], dict):
        default_ledfifo = str(data["ledfifo"])
    default_ledloop = 0
    if "ledloop" in data and not isinstance(data["ledloop"], dict):
        default_ledloop = str(data["ledloop"])
    default_ledmode = "LAST_COLOR"
    if "ledmode" in data and not isinstance(data["ledmode"], dict):
        default_ledmode = str(data["ledmode"])
    default_ledrandom = 0
    if "ledrandom" in data and not isinstance(data["ledrandom"], dict):
        default_ledrandom = float(data["ledrandom"])
    metadata = {}   
    for key, value in data.items():
        if key in {"duration", "midlayer", "middur", "midpose", "fhandle", "mhandle", "ydepth", "startframe", 
                   "traled", "tracolor", "ledsubdur", "ledfifo", "ledloop", "ledmode", "ledrandom"}:
            continue

        if not isinstance(value, dict):
            report({"WARNING"}, f"Ignoring malformed entry for '{key}' in {PREFIX_MAP_FILENAME}")
            continue

        try:
            entry_id = int(value.get("id"))
        except Exception:
            report({"WARNING"}, f"Missing or invalid id for '{key}' in {PREFIX_MAP_FILENAME}")
            continue

        duration = value.get("duration", default_duration)
        midlayer = value.get("midlayer", default_midlayer)
        middur = value.get("middur", default_middur)
        midpose = value.get("midpose", default_midpose)
        fhandle = value.get("fhandle", default_fhandle)
        mhandle = value.get("mhandle", default_mhandle)
        ydepth = value.get("ydepth", default_ydepth)
        traled = value.get("traled", default_traled)
        tracolor = value.get("tracolor", default_tracolor)
        ledsubdur = value.get("ledsubdur", default_ledsubdur)
        ledfifo = value.get("ledfifo", default_ledfifo)
        ledloop = value.get("ledloop", default_ledloop)
        ledmode = value.get("ledmode", default_ledmode)
        ledrandom = value.get("ledrandom", default_ledrandom)
        start_frame = value.get("startframe", default_start_frame)

        try:
            duration = int(duration)
        except Exception:
            report({"WARNING"}, f"Invalid duration for '{key}', using {default_duration}")
            duration = default_duration

        try:
            midlayer = max(1, int(midlayer))
        except Exception:
            midlayer = default_midlayer
        try:
            middur = max(1, int(middur))
        except Exception:
            middur = default_middur
        midpose = bool(midpose)
        try:
            fhandle = float(fhandle)
        except Exception:
            fhandle = default_fhandle
        try:
            mhandle = float(mhandle)
        except Exception:
            mhandle = default_mhandle

        try:
            ydepth = max(0.0, float(ydepth)) if ydepth is not None else None
        except Exception:
            ydepth = default_ydepth
        traled = bool(traled)
        tracolor = str(tracolor) if tracolor is not None else None
        try:
            start_frame = int(start_frame) if start_frame is not None else None
        except Exception:
            start_frame = default_start_frame

        metadata[str(key)] = {
            "id": entry_id,
            "transition_duration": duration,
            "duration": duration,  # kept for backward compatibility
            "midlayer": midlayer,
            "middur": middur,
            "midpose": midpose,
            "fhandle": fhandle,
            "mhandle": mhandle,
            "ydepth": ydepth,
            "traled": traled,
            "tracolor": tracolor,
            "ledsubdur": ledsubdur,
            "ledfifo": ledfifo,
            "ledloop": ledloop,
            "ledmode": ledmode,
            "ledrandom": ledrandom,
            "start_frame": start_frame,
        }

    return metadata, {"start_frame": default_start_frame}


def _metadata_transition_duration(meta, default=None):
    """Return transition duration value from ``meta`` if present."""

    meta = meta or {}
    try:
        duration = meta.get("transition_duration", meta.get("duration", default))
    except Exception:
        duration = default

    if duration is None:
        return default

    return int(duration)


def _metadata_handle(meta, key, default=None):
    """Return CopyLoc handle value from ``meta`` if present."""

    meta = meta or {}
    try:
        val = meta.get(key, default)
    except Exception:
        val = default

    if val is None:
        return default

    try:
        return float(val)
    except Exception:
        return default


def _entries_meta_from_metadata_map(metadata_map):
    """Expand metadata_map into a list aligned with storyboard entries."""

    if not metadata_map:
        return []

    entries_meta = []
    for _key, meta in sorted(metadata_map.items(), key=lambda item: item[1]["id"]):
        entries_meta.append(meta)
        if bool(meta.get("midpose", True)):
            mid_handle = _metadata_handle(meta, "mhandle", None)
            entries_meta.append({"copyloc_handle": mid_handle})
    return entries_meta


def _hex_to_rgba(hex_str):
    if not hex_str:
        return None
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    except Exception:
        return None


_SAMPLED_TRACOLOR_PATTERN = re.compile(r"(pre)?sampled_(\d+)", re.IGNORECASE)


def _sample_info_from_tracolor(tracolor: str | None) -> tuple[str | None, int | None]:
    """Return sampling mode and count from a ``tracolor`` string if present."""

    if not isinstance(tracolor, str):
        return None, None

    match = _SAMPLED_TRACOLOR_PATTERN.fullmatch(tracolor.strip())
    if not match:
        return None, None

    mode = "presampled" if match.group(1) else "sampled"

    try:
        return mode, max(1, int(match.group(2)))
    except Exception:
        return mode, None


def _colors_with_black_endpoints(colors):
    black = (0.0, 0.0, 0.0, 1.0)
    normalized = []
    for color in colors or []:
        normalized.append(tuple(_normalize_float_sequence(color, 4, 1.0)))
    if not normalized:
        normalized = [black]
    if normalized[0] != black:
        normalized.insert(0, black)
    if normalized[-1] != black:
        normalized.append(black)
    return normalized


def _apply_colors_to_color_ramp(
    light_effect_entry,
    colors: list[tuple[float, float, float, float]],
    *,
    black_endpoints: bool = False,
):
    """Configure or create a color ramp on ``light_effect_entry`` using ``colors``."""

    if black_endpoints:
        colors = _colors_with_black_endpoints(colors)

    if not colors:
        return False

    tex = getattr(light_effect_entry, "texture", None)
    if tex is None:
        try:
            tex = bpy.data.textures.new(
                name=f"{getattr(light_effect_entry, 'name', 'LE')}_ColorTex", type="IMAGE"
            )
            light_effect_entry.texture = tex
        except Exception:
            return False

    try:
        ramp = tex.color_ramp
    except Exception:
        ramp = None

    if ramp is None:
        return False

    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])

    if len(ramp.elements) == 0:
        ramp.elements.new(0.0)

    ramp.elements[0].position = 0.0
    ramp.elements[0].color = colors[0]

    num_colors = len(colors)
    if num_colors == 1:
        return True

    last_index = max(num_colors - 1, 1)
    for idx, color in enumerate(colors[1:], start=1):
        elem = ramp.elements.new(idx / last_index)
        elem.color = color

    return True


def _apply_transition_metadata(
    light_effect_entry,
    meta: dict | None,
    *,
    ramp_colors: list[tuple[float, float, float, float]] | None = None,
    black_endpoints: bool = False,
):
    """Apply transition customization stored in ``meta`` to ``light_effect_entry``."""

    meta = meta or {}
    applied_colors = None

    if ramp_colors:
        colors_to_apply = (
            _colors_with_black_endpoints(ramp_colors)
            if black_endpoints
            else list(ramp_colors)
        )
        if _apply_colors_to_color_ramp(
            light_effect_entry, colors_to_apply, black_endpoints=False
        ):
            applied_colors = [tuple(color) for color in colors_to_apply]

    fifo = meta.get("ledfifo", None)
    if fifo is not None:
        value = max(0.0, float(fifo))
        if hasattr(light_effect_entry, "fade_in_duration"):
            light_effect_entry.fade_in_duration = int(value)
        if hasattr(light_effect_entry, "fade_out_duration"):
            light_effect_entry.fade_out_duration = int(value)

    loop_count = meta.get("ledloop", None)
    if loop_count is not None and hasattr(light_effect_entry, "loop_count"):
        light_effect_entry.loop_count = max(0, int(loop_count))

    randomness = meta.get("ledrandom", None)
    if randomness is not None and hasattr(light_effect_entry, "randomness"):
        light_effect_entry.randomness = float(randomness)

    mode = meta.get("ledmode")
    if mode and hasattr(light_effect_entry, "output"):
        light_effect_entry.output = mode
            

    return applied_colors


def _adjust_transition_timing(frame_start: int, duration: int, meta: dict | None):
    """Adjust transition timing using ``ledsubdur`` metadata if present."""

    meta = meta or {}
    subdur = max(0, int(meta.get("ledsubdur", 0) or 0))

    if subdur <= 0:
        return frame_start, duration

    adjusted_start = int(frame_start) + int(round(subdur * 0.5))
    adjusted_duration = max(1, int(duration) - subdur)
    return adjusted_start, adjusted_duration


def _sample_colors_from_cat_effect(context, candidates: list[str], sample_count: int):
    """Return up to ``sample_count`` colors sampled from the CAT image of a target effect."""

    for candidate in candidates:
        le_entry = _find_light_effect_entry(context.scene, candidate)
        tex = getattr(le_entry, "texture", None) if le_entry is not None else None
        image = getattr(tex, "image", None) if tex is not None else None
        if image is None:
            continue

        pixels = np.array(image.pixels[:], dtype=float)

        if pixels.size == 0:
            continue

        pixels = pixels.reshape(-1, 4)
        rounded = np.clip(np.round(pixels[:, :4], 4), 0.0, 1.0)
        unique_colors, counts = np.unique(rounded, axis=0, return_counts=True)
        if unique_colors.size == 0:
            continue

        order = np.argsort(counts)[::-1]
        selected = unique_colors[order][:sample_count]
        colors = [tuple(color.tolist()) for color in selected]
        if colors:
            last_color = colors[-1]
            while len(colors) < sample_count:
                colors.append(last_color)
            return colors

    return None

def build_tracks_from_folder(folder, delimiter="auto"):
    files = []
    for name in os.listdir(folder):
        low = name.lower()
        if low.endswith(".csv") or low.endswith(".tsv"):
            files.append(os.path.join(folder, name))
    files.sort()
    tracks = []
    for p in files:
        data = load_csv(p, delimiter=delimiter)
        if not data:
            continue
        tracks.append({"name": os.path.splitext(os.path.basename(p))[0], "data": data})
    return tracks


def _parse_bounds_suffix(suffix: str):
    tokens = suffix.split("_")
    if len(tokens) != 8 or tokens[0] != "Start" or tokens[4] != "End":
        return None
    try:
        pos_min = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
        pos_max = (float(tokens[5]), float(tokens[6]), float(tokens[7]))
        return pos_min, pos_max
    except Exception:
        return None


def _find_vat_cat_set(folder: str):
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        if ext.lower() != ".exr" or "_VAT_" not in name.upper():
            continue
        if not name.endswith("_Pos"):
            continue

        base_and_suffix = name.split("_VAT_", 1)
        if len(base_and_suffix) != 2:
            continue
        base_name, suffix = base_and_suffix
        if not suffix.endswith("_Pos"):
            continue
        bounds_suffix = suffix[: -len("_Pos")]
        bounds = _parse_bounds_suffix(bounds_suffix)
        if bounds is None:
            continue

        color_candidate = os.path.join(
            folder, f"{base_name}_Color.png"
        )
        pos_path = os.path.join(folder, filename)

        if os.path.isfile(color_candidate):
            return {
                "base_name": base_name,
                "pos_path": pos_path,
                "color_path": color_candidate,
                "pos_min": bounds[0],
                "pos_max": bounds[1],
            }

    return None


def _load_vat_assets(folder: str):
    found = _find_vat_cat_set(folder)
    if not found:
        return None

    
    pos_img = bpy.data.images.load(found["pos_path"])
    color_img = bpy.data.images.load(found["color_path"])

    duration = max(int(getattr(pos_img, "size", (1,))[0] or 1) - 1, 0)
    return pos_img, color_img, found["pos_min"], found["pos_max"], duration


def calculate_duration_from_tracks(tracks, fps):
    """Return the duration in frames for ``tracks`` at the given ``fps``."""

    if not tracks:
        return 0
    max_t_ms = max(tr["data"][-1]["t_ms"] for tr in tracks if tr["data"])
    return int(ms_to_frame(max_t_ms, fps))


def _has_vat_cat_images(folder: str) -> bool:
    """Return True if the folder appears to contain VAT/CAT images."""

    names = os.listdir(folder)

    for name in names:
        lower = name.lower()
        if lower.endswith(".exr") or lower.endswith(".png"):
            if "vat" in lower or lower.endswith("_Color.png") or "_vat_" in lower:
                return True
    return False


def tracks_to_keydata(tracks, fps):
    """Convert loaded ``tracks`` to Skybrush key data format."""
    key_data = []
    for tr in tracks:
        if not tr["data"]:
            continue
        first = tr["data"][0]
        keys = {0: [], 1: [], 2: []}
        for row in tr["data"]:
            frame = ms_to_frame(row["t_ms"], fps)
            keys[0].append((frame, row["r"]))
            keys[1].append((frame, row["g"]))
            keys[2].append((frame, row["b"]))
        key_data.append({
            "name": tr["name"],
            "location": [first["x"], first["y"], first["z"]],
            "keys": keys,
        })
    return key_data

def ensure_mesh_with_armature(name="CSV_Tracks", count=1, first_positions=None):
    # Create an armature with one bone per vertex
    arm = bpy.data.armatures.new(name + "_arm")
    arm_obj = bpy.data.objects.new(name + "_Arm", arm)
    bpy.context.scene.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for i in range(count):
        bone = arm.edit_bones.new(f"Bone_{i}")
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 0.1)
    bpy.ops.object.mode_set(mode="POSE")
    for i in range(count):
        pb = arm_obj.pose.bones[f"Bone_{i}"]
        if first_positions and len(first_positions) == count:
            pb.location = first_positions[i]
        else:
            pb.location = (0.0, 0.0, 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Create mesh with vertices at origin
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    verts = [(0.0, 0.0, 0.0) for _ in range(count)]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    # Parent mesh to armature with modifier
    obj.parent = arm_obj
    mod = obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

    # Vertex groups per bone
    for i in range(count):
        vg = obj.vertex_groups.new(name=f"Bone_{i}")
        vg.add([i], 1.0, "REPLACE")

    # Match original orientation without manual rotation
    arm_obj.rotation_euler[0] = math.radians(-90)

    return obj


def _create_grid_positions(count, *, spacing=1.0, bounds=None, layers=1, layer_depth=None):
    """Return grid positions for ``count`` items, optionally stacked in layers."""

    if count <= 0:
        return []

    layers = max(1, int(layers))
    base_per_layer = count // layers
    remainder = count % layers

    center = Vector((0.0, 0.0, 0.0))
    dims = Vector((0.0, 0.0, 0.0))
    if bounds:
        min_v, max_v = bounds
        center = Vector(((min_v.x + max_v.x) * 0.5, 0.0, (min_v.z + max_v.z) * 0.5))
        dims = Vector((max_v.x - min_v.x, 0.0, max_v.z - min_v.z))

    if layers > 1 and layer_depth is not None:
        layer_spacing = float(layer_depth) / float(max(layers - 1, 1))
    else:
        layer_spacing = spacing
    start_layer = -0.5 * layer_spacing * (layers - 1)

    positions = []
    for layer_idx in range(layers):
        layer_count = base_per_layer + (1 if layer_idx < remainder else 0)
        if layer_count == 0:
            continue

        cols = max(1, math.ceil(math.sqrt(layer_count)))
        rows = math.ceil(layer_count / cols)

        spacing_x = spacing
        spacing_z = spacing
        if dims.x > 0 and dims.z > 0:
            ideal_cols = max(1, int(round(math.sqrt(layer_count * dims.x / dims.z))))
            candidates = {cols, ideal_cols, ideal_cols + 1, ideal_cols - 1}
            best_cols = cols
            best_rows = rows
            best_diff = float("inf")
            for cand in candidates:
                cand_cols = max(1, min(layer_count, cand))
                cand_rows = max(1, math.ceil(layer_count / cand_cols))
                cand_spacing_x = (
                    dims.x / (cand_cols - 1) if cand_cols > 1 and dims.x > 0 else spacing
                )
                cand_spacing_z = (
                    dims.z / (cand_rows - 1) if cand_rows > 1 and dims.z > 0 else spacing
                )
                diff = abs(cand_spacing_x - cand_spacing_z)
                if diff < best_diff:
                    best_diff = diff
                    best_cols = cand_cols
                    best_rows = cand_rows
                    spacing_x = cand_spacing_x
                    spacing_z = cand_spacing_z
            cols = best_cols
            rows = best_rows

        start_x = -0.5 * spacing_x * (cols - 1)
        start_z = -0.5 * spacing_z * (rows - 1)
        layer_y = start_layer + layer_idx * layer_spacing

        for idx in range(layer_count):
            row = idx // cols
            col = idx % cols
            x = start_x + col * spacing_x
            z = start_z + row * spacing_z
            positions.append((x + center.x, layer_y, z + center.z))
            if len(positions) >= count:
                return positions

    return positions


def _modifier_input_value(mod, socket_name):
    """Best-effort fetch of a Geometry Nodes input value by socket name."""

    if socket_name in mod:
        return mod.get(socket_name)

    try:
        return getattr(mod, socket_name)
    except Exception:
        pass

    group = getattr(mod, "node_group", None)
    sockets = []
    if group is not None:
        sockets = list(getattr(group, "inputs", []) or [])
    for idx, socket in enumerate(sockets):
        if getattr(socket, "name", None) != socket_name:
            continue
        for candidate in (
            getattr(socket, "identifier", None),
            f"Input_{idx + 1}",
        ):
            if not candidate:
                continue
            return mod.get(candidate)
    return None


def _geometry_node_bounds(obj):
    if obj is None:
        return None

    # Prefer explicit bounds exposed on a Geometry Nodes modifier
    for mod in getattr(obj, "modifiers", []):
        if getattr(mod, "type", None) != "NODES":
            continue

        min_pos = _modifier_input_value(mod, "MinPos")
        max_pos = _modifier_input_value(mod, "MaxPos")
        if min_pos is not None and max_pos is not None:
            min_v = Vector(min_pos)
            max_v = Vector(max_pos)
            return min_v, max_v

    # Fallback to the object's bounding box if no GN bounds are present
    try:
        bb = getattr(obj, "bound_box", None)
        if bb:
            corners = [Vector(c) for c in bb]
            if hasattr(obj, "matrix_world"):
                corners = [obj.matrix_world @ c for c in corners]
            min_v = Vector(
                (
                    min(c.x for c in corners),
                    min(c.y for c in corners),
                    min(c.z for c in corners),
                )
            )
            max_v = Vector(
                (
                    max(c.x for c in corners),
                    max(c.y for c in corners),
                    max(c.z for c in corners),
                )
            )
            return min_v, max_v
    except Exception:
        pass
    return None


def _bounds_from_tracks(tracks):
    """Compute XY bounds from track data (using x,z axes)."""

    if not tracks:
        return None

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for tr in tracks:
        for row in tr.get("data", []):
            x, y, z = row.get("x"), row.get("y"), row.get("z")
            if x is None or y is None or z is None:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    if min_x == float("inf"):
        return None

    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))


def _combine_bounds(b1, b2):
    """Return midpoint bounds averaged from two bounds."""

    if not b1:
        return b2
    if not b2:
        return b1

    min_v1, max_v1 = b1
    min_v2, max_v2 = b2
    min_v = Vector(((min_v1.x + min_v2.x) * 0.5, (min_v1.y + min_v2.y) * 0.5, (min_v1.z + min_v2.z) * 0.5))
    max_v = Vector(((max_v1.x + max_v2.x) * 0.5, (max_v1.y + max_v2.y) * 0.5, (max_v1.z + max_v2.z) * 0.5))
    return min_v, max_v


def create_grid_mid_pose(
    context,
    frame_start,
    base_name="MidPose",
    spacing=1.0,
    reference_obj=None,
    layers=1,
    duration=1,
    meta=None,
    next_bounds=None,
    layer_depth=None,
    transition_le_meta=None,
):
    """Create and add a grid-formation storyboard entry at ``frame_start``.

    The grid will have as many vertices as there are drone objects in the
    "Drones" collection. The formation is appended to the storyboard with the
    given ``duration`` (default: 1 frame).
    """

    drones = bpy.data.collections.get("Drones")
    drone_count = len(drones.objects) if drones else 0
    if drone_count == 0:
        return None

    ref_spacing = spacing
    bounds = _geometry_node_bounds(reference_obj)
    if bounds and next_bounds:
        bounds = _combine_bounds(bounds, next_bounds)
    elif next_bounds and not bounds:
        bounds = next_bounds
    if bounds:
        min_v, max_v = bounds
        dims = Vector((max_v.x - min_v.x, 0.0, max_v.z - min_v.z))
        max_dim = max(dims.x, dims.z)
        side = max(1, math.ceil(math.sqrt(drone_count)))
        if side > 1 and max_dim > 0:
            ref_spacing = max_dim / (side - 1)
        elif max_dim > 0:
            ref_spacing = max_dim

    positions = _create_grid_positions(
        drone_count,
        spacing=ref_spacing,
        bounds=bounds,
        layers=layers,
        layer_depth=layer_depth,
    )
    mesh = bpy.data.meshes.new(f"{base_name}_mesh")
    mesh.from_pydata(positions, [], [])
    mesh.update()

    obj = bpy.data.objects.new(base_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.location = (0.0, 0.0, 0.0)

    vg = obj.vertex_groups.new(name="Drones")
    vg.add(range(len(mesh.vertices)), 1.0, "REPLACE")
    if hasattr(obj, "skybrush"):
        obj.skybrush.formation_vertex_group = "Drones"

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj

    try:
        bpy.ops.skybrush.create_formation(name=obj.name, contents="SELECTED_OBJECTS")
        bpy.ops.skybrush.append_formation_to_storyboard()
        storyboard = context.scene.skybrush.storyboard
        entry = storyboard.entries[-1]
        entry.name = base_name
        entry.frame_start = frame_start
        entry.duration = max(1, int(duration))
        if meta is not None:
            try:
                entry["metadata"] = json.dumps(meta)
            except Exception:
                pass
        if transition_le_meta:
            try:
                entry["transition_le"] = json.dumps(transition_le_meta)
            except Exception:
                pass
        return entry
    except Exception:
        return None

# ---------- Utilities for Replacement ----------

def clear_drone_keys(start_frame, duration):
    """Remove color keyframes on drones within the given frame range."""
    drones_col = bpy.data.collections.get("Drones")
    if not drones_col:
        return
    end_frame = start_frame + duration
    for obj in drones_col.objects:
        mat = obj.active_material
        if not mat or not mat.node_tree:
            continue
        anim = mat.node_tree.animation_data
        if not anim or not anim.action:
            continue
        for fcurve in anim.action.fcurves:
            for key in reversed(fcurve.keyframe_points):
                if start_frame <= key.co.x <= end_frame:
                    fcurve.keyframe_points.remove(key)


def _update_frame_range_from_storyboard(context):
    """Extend the scene frame range to cover all storyboard entries."""

    scene = context.scene
    storyboard = getattr(getattr(scene, "skybrush", None), "storyboard", None)
    entries = getattr(storyboard, "entries", None)
    if not entries:
        return

    max_end = 0
    for entry in entries:
        try:
            start = int(getattr(entry, "frame_start", 0))
            duration = int(getattr(entry, "duration", 0))
        except (TypeError, ValueError):
            continue
        max_end = max(max_end, start + max(duration, 0))

    if max_end > 0:
        scene.frame_end = max(scene.frame_end, max_end)


def _shift_subsequent_storyboard_entries(storyboard, start_index: int, delta: int):
    """Shift storyboard entries and transitions after ``start_index`` by ``delta`` frames."""

    if delta == 0 or storyboard is None or start_index is None:
        return

    entries = getattr(storyboard, "entries", None)
    transitions = getattr(storyboard, "transitions", None)

    if entries:
        for idx in range(start_index + 1, len(entries)):
            entry = entries[idx]
            try:
                entry.frame_start = int(getattr(entry, "frame_start", 0)) + delta
            except Exception:
                continue

    if transitions:
        for idx in range(max(start_index, 0), len(transitions)):
            transition = transitions[idx]
            try:
                transition.frame_start = int(getattr(transition, "frame_start", 0)) + delta
            except Exception:
                continue


def _shift_storyboard_after_transition(storyboard, start_index: int, delta: int):
    """Shift storyboard entries and transitions after ``start_index`` by ``delta`` frames."""

    if delta == 0 or storyboard is None:
        return

    entries = getattr(storyboard, "entries", None)
    transitions = getattr(storyboard, "transitions", None)

    if entries:
        for idx in range(start_index + 1, len(entries)):
            entry = entries[idx]
            try:
                entry.frame_start = int(getattr(entry, "frame_start", 0)) + delta
            except Exception:
                continue

    if transitions:
        for idx in range(start_index + 1, len(transitions)):
            transition = transitions[idx]
            try:
                transition.frame_start = int(getattr(transition, "frame_start", 0)) + delta
            except Exception:
                continue


def _apply_transition_durations(storyboard, entries_meta):
    """Assign transition durations from metadata to storyboard transitions."""

    transitions = getattr(storyboard, "transitions", None)
    entries = getattr(storyboard, "entries", None)
    if not transitions or not entries:
        return

    limit = min(len(transitions), len(entries_meta) - 1, len(entries) - 1)
    for idx in range(limit):
        target_meta = entries_meta[idx + 1]
        duration = _metadata_transition_duration(target_meta)
        if duration is None:
            continue
        try:
            current_duration = getattr(transitions[idx], "duration", 0)
            transitions[idx].duration = duration
            delta = duration - current_duration
            if delta:
                _shift_storyboard_after_transition(storyboard, idx, delta)
        except Exception:
            continue


def _apply_copyloc_handles_from_metadata(context, storyboard, entries_meta):
    """Shape CopyLoc influence curves per-entry using metadata handles."""

    entries = getattr(storyboard, "entries", None)
    if not entries or not entries_meta:
        return 0

    drones_collection = bpy.data.collections.get("Drones")
    if not drones_collection:
        return 0

    handle_default = getattr(context.scene, "copyloc_handle_frames", 5.0)
    targets = list(drones_collection.objects)
    updated = 0

    for idx, entry in enumerate(entries):
        meta = entries_meta[idx] if idx < len(entries_meta) else None
        handle = None
        if meta:
            if "copyloc_handle" in meta:
                handle = meta.get("copyloc_handle")
            elif getattr(entry, "name", "").endswith("_MidPose"):
                handle = meta.get("mhandle")
            else:
                handle = meta.get("fhandle")

        try:
            handle_frames = float(handle if handle is not None else handle_default)
        except Exception:
            handle_frames = handle_default

        try:
            start = int(getattr(entry, "frame_start", 0))
            duration = int(getattr(entry, "duration", 0))
        except Exception:
            continue
        frame_min = start - 1
        frame_max = start + max(duration, 0) + 1

        def _key_in_range(key):
            try:
                frame = float(getattr(key.co, "x", None))
            except Exception:
                return False
            return frame_min <= frame <= frame_max

        for obj in targets:
            anim = obj.animation_data
            if not anim or not anim.action:
                continue
            action = anim.action
            for const in obj.constraints:
                if const.type != 'COPY_LOCATION':
                    continue
                fcurve = action.fcurves.find(
                    f'constraints["{const.name}"].influence'
                )
                if not fcurve:
                    continue
                if shape_copyloc_influence_curve(
                    fcurve, handle_frames, key_filter=_key_in_range
                ):
                    updated += 1

    return updated


def _create_light_effect_for_storyboard(
    context,
    mesh_obj,
    storyboard_entry,
    *,
    effect_type: str = "VERTEX_COLOR",
    color_image=None,
    assign_mesh: bool = True,
):
    """Create a light effect aligned with ``storyboard_entry``."""

    skybrush = getattr(context.scene, "skybrush", None)
    light_effects = getattr(skybrush, "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if entries is None:
        return None

    le_entry = None
    append_entry = getattr(light_effects, "append_new_entry", None)
    if callable(append_entry):
        try:
            le_entry = append_entry(
                storyboard_entry.name,
                storyboard_entry.frame_start,
                storyboard_entry.duration,
                select=False,
                context=context,
            )
        except Exception:
            le_entry = None

    if le_entry is None:
        try:
            le_entry = entries.add()
            le_entry.name = storyboard_entry.name
            le_entry.frame_start = storyboard_entry.frame_start
            le_entry.duration = storyboard_entry.duration
        except Exception:
            return None

    if hasattr(le_entry, "type"):
        try:
            le_entry.type = effect_type
        except Exception:
            pass
    if effect_type == "VERTEX_COLOR" and hasattr(le_entry, "output"):
        try:
            le_entry.output = OUTPUT_VERTEX_COLOR
        except Exception:
            pass
    if assign_mesh and hasattr(le_entry, "mesh"):
        try:
            le_entry.mesh = mesh_obj
        except Exception:
            pass
    if hasattr(le_entry, "convert_srgb"):
        try:
            le_entry.convert_srgb = False
        except Exception:
            pass

    if color_image is not None and hasattr(le_entry, "texture"):
        tex = getattr(le_entry, "texture", None)
        if tex is None:
            tex = bpy.data.textures.new(
                name=f"{storyboard_entry.name}_ColorTex", type="IMAGE"
            )
            le_entry.texture = tex

        if tex is not None:
            tex.image = color_image

    return le_entry


def _create_color_light_effect(
    context,
    name,
    frame_start,
    duration,
    color,
    *,
    ramp_colors: list[tuple[float, float, float, float]] | None = None,
    meta: dict | None = None,
    black_endpoints: bool = False,
):
    """Create a simple COLOR_RAMP light effect with a flat color."""

    skybrush = getattr(context.scene, "skybrush", None)
    light_effects = getattr(skybrush, "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if entries is None:
        return None

    append_entry = getattr(light_effects, "append_new_entry", None)
    le_entry = None
    if callable(append_entry):
        le_entry = append_entry(
            name,
            frame_start=int(frame_start),
            duration=int(duration),
            select=False,
            context=context,
        )

    if le_entry is None:
        le_entry = entries.add()
        le_entry.name = name
        le_entry.frame_start = int(frame_start)
        le_entry.duration = int(duration)
        if hasattr(le_entry, "type"):
            le_entry.type = "COLOR_RAMP"
        tex = getattr(le_entry, "texture", None)
        if tex is None:
            tex = bpy.data.textures.new(name=f"{name}_ColorTex", type="IMAGE")
            le_entry.texture = tex
    else:
        tex = getattr(le_entry, "texture", None)
    _apply_colors_to_color_ramp(
        le_entry, ramp_colors if ramp_colors else [color]
    )

    if le_entry:
        _apply_transition_metadata(
            le_entry,
            meta,
            ramp_colors=ramp_colors if ramp_colors else [color],
            black_endpoints=black_endpoints,
        )

    return le_entry


def _apply_pending_sampled_transitions(
    context,
    candidates: set[str],
    pending: list[dict],
    *,
    color_cache: dict[str, list[tuple[float, float, float, float]]] | None = None,
):
    """Apply color ramp updates for transitions waiting on CAT colors."""

    remaining = []
    last_applied_colors = None
    for item in pending:
        target_candidates = item.get("target_candidates", set())
        if not candidates.intersection(target_candidates):
            remaining.append(item)
            continue

        transition_name = item.get("transition_name")
        sample_count = item.get("sample_count")
        black_edges = bool(item.get("black_edges", False))
        if not transition_name or not sample_count:
            continue

        colors = _sample_colors_from_cat_effect(
            context, list(target_candidates), int(sample_count)
        )
        if not colors:
            remaining.append(item)
            continue

        le_entry = _find_light_effect_entry(context.scene, transition_name)
        if le_entry is None:
            remaining.append(item)
            continue

        _apply_transition_metadata(
            le_entry,
            item.get("meta"),
            ramp_colors=colors,
            black_endpoints=black_edges,
        )
        last_applied_colors = _colors_with_black_endpoints(colors) if black_edges else colors
        if color_cache is not None:
            color_cache[transition_name] = last_applied_colors

    pending[:] = remaining
    return last_applied_colors


def _find_light_effect_entry(scene, name: str):
    light_effects = getattr(getattr(scene, "skybrush", None), "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if not entries:
        return None

    for entry in entries:
        if getattr(entry, "name", None) == name:
            return entry
    return None


def _replace_light_effect_texture(light_effect, color_image):
    if light_effect is None:
        return

    tex = getattr(light_effect, "texture", None)
    if tex is None:
        try:
            tex = bpy.data.textures.new(name=f"{light_effect.name}_ColorTex", type="IMAGE")
            light_effect.texture = tex
        except Exception:
            tex = None

    if tex is not None:
        try:
            old_img = getattr(tex, "image", None)
            if old_img is not None:
                bpy.data.images.remove(old_img)
        except Exception:
            pass
        try:
            tex.image = color_image
        except Exception:
            pass


def _remove_light_effect_entries(scene, name: str):
    light_effects = getattr(getattr(scene, "skybrush", None), "light_effects", None)
    entries = getattr(light_effects, "entries", None)
    if not entries:
        return

    for idx in reversed(range(len(entries))):
        entry = entries[idx]
        if getattr(entry, "name", None) != name:
            continue

        tex = getattr(entry, "texture", None)
        if tex is not None:
            img = getattr(tex, "image", None)
            try:
                entries.remove(idx)
            except Exception:
                pass
            if img is not None:
                try:
                    bpy.data.images.remove(img)
                except Exception:
                    pass
            try:
                bpy.data.textures.remove(tex)
            except Exception:
                pass
        else:
            try:
                entries.remove(idx)
            except Exception:
                pass


def _remove_vat_images_for_storyboard(name: str):
    image_names = (
        f"{name}_VAT_Pos",
        f"{name}_VAT_Color",
        f"{name}_CAT_Pos",
        f"{name}_CAT_Color",
    )

    for img_name in image_names:
        img = bpy.data.images.get(img_name)
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass


def _remove_objects_for_storyboard(name: str):
    for candidate_name in (name, f"{name}_CSV"):
        obj = bpy.data.objects.get(candidate_name)
        if obj is None:
            continue

        mesh = getattr(obj, "data", None)
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

        if mesh and mesh.users == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass

# ---------- Import Helper ----------


def _hide_csv_mesh(obj):
    if obj is None:
        return

    try:
        obj.hide_set(True)
    except Exception:
        try:
            obj.hide_viewport = True
        except Exception:
            pass


def import_csv_folder(
    context,
    folder,
    start_frame,
    *,
    use_vat: bool = False,
    image_export_dir: str | None = None,
):
    """Import CSV tracks from ``folder`` and create a mesh with animation.

    Returns a tuple ``(object, duration, key_entries)`` where ``object`` is the
    created mesh or ``None`` if no tracks were found, ``duration`` is the
    animation length in frames and ``key_entries`` contains color keyframe data
    prepared for later application."""
    fps = context.scene.render.fps

    folder_name = os.path.basename(os.path.normpath(folder))
    folder_name, _gap = split_name_and_gap(folder_name, fps)
    tracks = build_tracks_from_folder(folder)
    vat_assets = None
    color_image_for_le = None
    if not tracks and use_vat:
        vat_assets = _load_vat_assets(folder)
    if not tracks and vat_assets is None:
        return None, 0, None


    if vat_assets is not None:
        pos_img, color_img, pos_min, pos_max, duration = vat_assets
        obj = csv_vat_gn.create_vat_animation_from_images(
            pos_img,
            color_img,
            pos_min,
            pos_max,
            start_frame=start_frame,
            base_name=f"{folder_name}_CSV",
        )
        color_image_for_le = color_img
        key_entries = None
    else:
        # Determine total animation duration strictly from the CSV content
        duration = calculate_duration_from_tracks(tracks, fps)


        # Build initial positions array
        first_positions = []
        for tr in tracks:
            d0 = tr["data"][0]
            first_positions.append((d0["x"], d0["y"], d0["z"]))

        if use_vat:
            obj, color_image, pos_image = csv_vat_gn.create_vat_animation_from_tracks(
                tracks,
                fps,
                start_frame=start_frame,
                base_name=f"{folder_name}_CSV",
                storyboard_name=folder_name,
            )
            export_dir = image_export_dir
            if export_dir and pos_image is not None and color_image is not None:
                _export_vat_images(pos_image, color_image, export_dir)
            key_entries = None
            color_image_for_le = color_image
        else:
            # Create mesh and armature with N bones/vertices
            # Name it after the storyboard entry + "_CSV" for clarity
            obj = ensure_mesh_with_armature(
                name=f"{folder_name}_CSV",
                count=len(tracks),
                first_positions=first_positions,
            )

            # Animate bones directly with keyframes
            arm_obj = obj.parent
            for i, tr in enumerate(tracks):
                pb = arm_obj.pose.bones.get(f"Bone_{i}")
                if not pb:
                    continue
                for row in tr["data"]:
                    frame = start_frame + ms_to_frame(row["t_ms"], fps)
                    pb.location = (row["x"], row["y"], row["z"])
                    pb.keyframe_insert(data_path="location", frame=frame)

            if arm_obj.animation_data and arm_obj.animation_data.action:
                for fcurve in arm_obj.animation_data.action.fcurves:
                    for key in fcurve.keyframe_points:
                        key.interpolation = 'LINEAR'

            # Prepare color keyframe data for later application
            key_entries = tracks_to_keydata(tracks, fps)

    if obj is None:
        return None, duration, None

    # Ensure a vertex group exists for formation mapping
    vg = obj.vertex_groups.get("Drones")
    if vg is None:
        vg = obj.vertex_groups.new(name="Drones")
        vg.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')

    if hasattr(obj, "skybrush"):
        obj.skybrush.formation_vertex_group = "Drones"

    # Create a formation and add a storyboard entry
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.skybrush.create_formation(name=obj.name, contents='SELECTED_OBJECTS')
    entry = None
    try:
        bpy.ops.skybrush.append_formation_to_storyboard()
        storyboard = context.scene.skybrush.storyboard
        entry = storyboard.entries[-1]
        entry.name = folder_name
        entry.frame_start = start_frame
        entry.duration = duration
    except Exception:
        entry = None

    if use_vat and entry is not None:
        _create_light_effect_for_storyboard(
            context,
            obj,
            entry,
            effect_type="CAT",
            color_image=color_image_for_le,
            assign_mesh=False,
        )

    _update_frame_range_from_storyboard(context)

    _hide_csv_mesh(obj)

    return obj, duration, key_entries


# ---------- Properties / UI ----------


class CSVVA_PreviewItem(PropertyGroup):
    name: StringProperty(name="Name")
    folder: StringProperty(name="Folder")
    start_frame: IntProperty(name="Start Frame", default=0)
    duration: IntProperty(name="Duration", default=0)
    midpose: BoolProperty(name="Add MidPose", default=True)
    midpose_duration: IntProperty(name="MidPose Duration", default=1, min=1)
    checked: BoolProperty(name="Include", default=False)
    exists: BoolProperty(name="Exists", default=False)
    frame_mismatch: BoolProperty(name="Frame Mismatch", default=False)


class CSVVA_Props(PropertyGroup):
    folder: StringProperty(
        name="CSV Folder",
        description="Folder containing CSV/TSV files with columns: Time[msec], x[m], y[m], z[m], Red, Green, Blue",
        subtype="DIR_PATH",
    )
    use_vat: BoolProperty(
        name="Use VAT",
        description="Generate vertex animation textures instead of bone-based animation",
        default=False,
    )
    export_images: BoolProperty(
        name="Link CAT/VAT images to files",
        description="Save CAT/VAT images next to the .blend file and reference them instead of embedding",
        default=False,
    )
    preview_items: CollectionProperty(type=CSVVA_PreviewItem)
    preview_index: IntProperty(default=0)


class CSVVA_UL_Preview(UIList):
    bl_idname = "CSVVA_UL_Preview"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "checked", text="")
        main = row.row(align=True)
        main.label(text=item.name)
        main.label(text=f"Start: {item.start_frame}")
        main.label(text=f"Dur: {item.duration}")
        status_icon = "CHECKMARK" if not item.exists else "FILE_REFRESH"
        status_text = "New" if not item.exists else "Existing"
        if item.frame_mismatch:
            status_icon = "ERROR"
            status_text = "Frame diff"
        main.label(text=status_text, icon=status_icon)


class CSVVA_OT_PrepareFolders(Operator):
    bl_idname = "csvva.prepare_folders"
    bl_label = "Prep Folders"
    bl_description = (
        "Unzip archives in the selected folder and normalize names"
    )

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)

        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}

        zip_files = [f for f in os.listdir(folder) if f.lower().endswith(".zip")]
        if not zip_files:
            self.report({"INFO"}, "No ZIP archives found")
            return {"CANCELLED"}

        prepared = []
        for zip_name in sorted(zip_files):
            zip_path = os.path.join(folder, zip_name)
            base_name = os.path.splitext(zip_name)[0]
            target_dir = os.path.join(folder, base_name)

            if os.path.isdir(target_dir):
                try:
                    shutil.rmtree(target_dir)
                except Exception as exc:
                    self.report(
                        {"ERROR"},
                        f"Failed to remove existing folder {base_name}: {exc}",
                    )
                    continue

            try:
                os.makedirs(target_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(target_dir)
            except Exception as exc:
                self.report({"ERROR"}, f"Failed to extract {zip_name}: {exc}")
                continue

            try:
                os.remove(zip_path)
            except Exception:
                self.report({"WARNING"}, f"Could not remove {zip_name} after extraction")

            prepared.append(os.path.basename(target_dir))

        if prepared:
            details = ", ".join(prepared)
            self.report({"INFO"}, f"Prepared {len(prepared)} folder(s): {details}")
            return {"FINISHED"}

        self.report({"ERROR"}, "No folders prepared")
        return {"CANCELLED"}


class CSVVA_OT_Import(Operator):
    bl_idname = "csvva.import_setup"
    bl_label = "Import & Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)
        storyboard = context.scene.skybrush.storyboard
        export_dir = None
        if prefs.use_vat and prefs.export_images:
            export_dir = _ensure_export_directory(self.report, "//")
            if not export_dir:
                return {"CANCELLED"}
        base_start = 0
        existing_entry_count = len(storyboard.entries)
        if storyboard.entries:
            last = storyboard.entries[-1]
            base_start = last.frame_start + last.duration
        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}
        metadata_map, metadata_defaults = load_import_metadata(folder, self.report)
        if metadata_defaults.get("start_frame") is not None:
            base_start = metadata_defaults["start_frame"]

        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        ordered_subdirs = list(subdirs)
        if metadata_map:
            ordered_subdirs = []
            subdir_set = set(subdirs)
            for key, meta in sorted(metadata_map.items(), key=lambda item: item[1]["id"]):
                if key in subdir_set:
                    ordered_subdirs.append(key)
                else:
                    self.report({"WARNING"}, f"No folder matched metadata key '{key}'")
            for d in subdirs:
                if d not in metadata_map:
                    ordered_subdirs.append(d)

        if subdirs:
            created = []
            entries_meta = [None] * existing_entry_count
            next_start = base_start
            key_data_collection = []
            pending_sampled_transitions: list[dict] = []
            transition_color_cache: dict[str, list[tuple[float, float, float, float]]] = {}
            last_transition_colors: list[tuple[float, float, float, float]] | None = None
            for idx, d in enumerate(ordered_subdirs):
                sub_path = os.path.join(folder, d)
                base_name, gap_frames = split_name_and_gap(d, context.scene.render.fps)
                meta = metadata_map.get(d, {}) if metadata_map else {}
                display_name = _storyboard_name(base_name, meta)
                mid_layers = meta.get("midlayer", 1) or 1
                midpose_enabled = bool(meta.get("midpose", True))
                mid_duration = max(1, int(meta.get("middur", 1) or 1))
                ydepth = meta.get("ydepth", None)
                mid_handle = _metadata_handle(meta, "mhandle", None)
                traled = bool(meta.get("traled", False))
                tracolor_value = meta.get("tracolor")
                sample_mode, sample_count = _sample_info_from_tracolor(tracolor_value)
                tracolor = _hex_to_rgba(tracolor_value) or (1.0, 1.0, 1.0, 1.0)
                next_meta = (
                    metadata_map.get(ordered_subdirs[idx + 1], {})
                    if idx < len(ordered_subdirs) - 1
                    else {}
                )
                transition_duration = _metadata_transition_duration(next_meta, default=0) or 0
                midpose_disabled = (transition_duration <= 0) or (not midpose_enabled)
                sf_meta = meta.get("start_frame", None) if meta else None
                sf = sf_meta if sf_meta is not None else next_start
                next_bounds = None
                if idx < len(ordered_subdirs) - 1:
                    next_tracks = build_tracks_from_folder(os.path.join(folder, ordered_subdirs[idx + 1]))
                    next_bounds = _bounds_from_tracks(next_tracks)
                obj, dur, key_entries = import_csv_folder(
                    context,
                    sub_path,
                    sf,
                    use_vat=prefs.use_vat,
                    image_export_dir=export_dir,
                )
                if obj:
                    created.append(obj)
                    if key_entries:
                        key_data_collection.append((key_entries, sf, obj, sub_path))

                    applied_colors = _apply_pending_sampled_transitions(
                        context,
                        {display_name, base_name},
                        pending_sampled_transitions,
                        color_cache=transition_color_cache,
                    )
                    if applied_colors:
                        last_transition_colors = applied_colors

                    effective_duration = dur or DEFAULT_FOLDER_DURATION

                    try:
                        entry = storyboard.entries[-1]
                        entry.name = (
                            f"{meta.get('id', '')}_{display_name}"
                            if meta and meta.get("id") is not None
                            else display_name
                        )
                        entry.duration = effective_duration
                        entries_meta.append(meta)
                    except Exception:
                        pass

                    gap_for_next = gap_frames if gap_frames is not None else 0
                    if idx < len(ordered_subdirs) - 1 and not midpose_disabled:
                        mid_offset = max(1, int(round(transition_duration * 0.5)))
                        mid_frame = int(sf or 0) + int(effective_duration or 0) + mid_offset
                        mid_entry = create_grid_mid_pose(
                            context,
                            mid_frame,
                            base_name=f"{display_name}_MidPose",
                            reference_obj=obj,
                            layers=mid_layers,
                            duration=mid_duration,
                            meta={"copyloc_handle": mid_handle} if mid_handle is not None else None,
                            next_bounds=next_bounds,
                            layer_depth=ydepth,
                        )
                        if mid_entry:
                            entries_meta.append({"copyloc_handle": mid_handle})
                    if traled and idx < len(ordered_subdirs) - 1:
                        trans_start = int(sf or 0) + int(effective_duration or 0)
                        trans_duration = max(1, int(gap_for_next or 0) + int(transition_duration or 0))
                        trans_start, trans_duration = _adjust_transition_timing(
                            trans_start, trans_duration, meta
                        )
                        transition_name = f"{display_name}_TransitionLE"
                        le_entry = _create_color_light_effect(
                            context,
                            transition_name,
                            trans_start,
                            trans_duration,
                            tracolor,
                            ramp_colors=[tracolor],
                            meta=meta,
                            black_endpoints=bool(sample_mode),
                        )
                        applied_colors = None
                        if sample_mode == "presampled":
                            source_colors = last_transition_colors or transition_color_cache.get(
                                transition_name
                            )
                            if source_colors:
                                applied_colors = _apply_transition_metadata(
                                    le_entry,
                                    meta,
                                    ramp_colors=source_colors,
                                    black_endpoints=True,
                                )
                        elif sample_mode == "sampled" and sample_count:
                            next_base_name, _ = split_name_and_gap(
                                ordered_subdirs[idx + 1], context.scene.render.fps
                            )
                            next_display_name = _storyboard_name(
                                next_base_name, metadata_map.get(ordered_subdirs[idx + 1], {})
                            )
                            targets = {next_display_name, next_base_name}
                            pending_sampled_transitions.append(
                                {
                                    "transition_name": transition_name,
                                    "sample_count": sample_count,
                                    "target_candidates": targets,
                                    "black_edges": True,
                                    "meta": meta,
                                }
                            )
                            applied_colors = _apply_pending_sampled_transitions(
                                context,
                                targets,
                                pending_sampled_transitions,
                                color_cache=transition_color_cache,
                            )
                        else:
                            applied_colors = [tuple(tracolor)]

                        if applied_colors:
                            transition_color_cache[transition_name] = applied_colors
                            last_transition_colors = applied_colors
                    transition_for_next = (
                        transition_duration if idx < len(ordered_subdirs) - 1 else 0
                    )
                    next_start = max(
                        next_start,
                        int(sf or 0)
                        + int(effective_duration or 0)
                        + int(gap_for_next or 0)
                        + int(transition_for_next or 0),
                    )
            if not created:
                self.report({"ERROR"}, "No CSV/TSV files found in subfolders")
                return {"CANCELLED"}
            try:
                bpy.ops.skybrush.recalculate_transitions(scope="ALL")
            except Exception:
                pass
            _apply_transition_durations(storyboard, entries_meta)
            _apply_copyloc_handles_from_metadata(context, storyboard, entries_meta)
            for key_entries, sf, obj, sub_path in key_data_collection:
                current_frame = context.scene.frame_current
                context.scene.frame_set(sf)
                apply_color_keys_from_key_data(
                    key_entries,
                    sf,
                )
                context.scene.frame_set(current_frame)
            self.report({"INFO"}, f"Setup complete for {len(created)} folders")
            return {"FINISHED"}

        folder_name = os.path.basename(os.path.normpath(folder))
        base_name, _gap = split_name_and_gap(
            folder_name, context.scene.render.fps
        )
        meta = metadata_map.get(folder_name, {}) if metadata_map else {}
        display_name = _storyboard_name(base_name, meta)
        sf_meta = meta.get("start_frame", None) if meta else None
        start_frame = sf_meta if sf_meta is not None else base_start
        storyboard = context.scene.skybrush.storyboard
        tracolor_value = meta.get("tracolor")
        sample_mode, _sample_count = _sample_info_from_tracolor(tracolor_value)
        tracolor = _hex_to_rgba(tracolor_value) or (1.0, 1.0, 1.0, 1.0)
        for idx, sb in enumerate(storyboard.entries):
            if sb.name == display_name:
                old_obj = bpy.data.objects.get(sb.name)
                if old_obj:
                    mesh = old_obj.data
                    bpy.data.objects.remove(old_obj, do_unlink=True)
                    if mesh and mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
                clear_drone_keys(sb.frame_start, sb.duration)
                storyboard.entries.remove(idx)
                break

        obj, _, key_entries = import_csv_folder(
            context, folder, start_frame, use_vat=prefs.use_vat
        )
        if not obj:
            self.report({"ERROR"}, "No CSV/TSV files found in folder")
            return {"CANCELLED"}
        try:
            bpy.ops.skybrush.recalculate_transitions(scope="ALL")
        except Exception:
            pass
        entries_meta = [None] * existing_entry_count + [meta]
        _apply_transition_durations(
            storyboard, entries_meta
        )
        _apply_copyloc_handles_from_metadata(context, storyboard, entries_meta)
        try:
            entry = storyboard.entries[-1]
            entry.name = (
                f"{meta.get('id', '')}_{display_name}"
                if meta and meta.get("id") is not None
                else display_name
            )
        except Exception:
            pass
        if bool(meta.get("traled", False)):
            duration = max(1, int(meta.get("middur", 1) or 1))
            trans_start = int(getattr(entry, "frame_start", start_frame)) + max(
                0, int(getattr(entry, "duration", 0) or 0)
            )
            trans_start, duration = _adjust_transition_timing(
                trans_start, duration, meta
            )
            _create_color_light_effect(
                context,
                f"{display_name}_TransitionLE",
                trans_start,
                duration,
                tracolor,
                ramp_colors=[tracolor],
                meta=meta,
                black_endpoints=bool(sample_mode),
            )
        if key_entries:
            current_frame = context.scene.frame_current
            context.scene.frame_set(start_frame)
            apply_color_keys_from_key_data(
                key_entries,
                start_frame,
            )
            context.scene.frame_set(current_frame)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({"INFO"}, f"Setup complete: {obj.name}")
        return {"FINISHED"}


def _format_bounds_suffix(pos_min, pos_max):
    def _fmt(value: float) -> str:
        return (f"{value:.3f}").rstrip("0").rstrip(".")

    return (
        f"Start_{_fmt(pos_min[0])}_{_fmt(pos_min[1])}_{_fmt(pos_min[2])}_"
        f"End_{_fmt(pos_max[0])}_{_fmt(pos_max[1])}_{_fmt(pos_max[2])}"
    )


def _save_image(image, filepath, file_format):
    image.filepath_raw = filepath
    image.file_format = file_format
    image.save()


def _link_image_to_file(image, filepath, file_format):
    """Save ``image`` to ``filepath`` and keep it linked to the file."""

    _save_image(image, filepath, file_format)
    image.filepath = filepath
    try:
        image.source = 'FILE'
    except Exception:
        pass
    try:
        image.reload()
    except Exception:
        pass


def _ensure_export_directory(report_fn, export_dir):
    if not export_dir:
        return None

    if not bpy.data.filepath:
        report_fn({"ERROR"}, "Please save the .blend file first")
        return None

    resolved = bpy.path.abspath(export_dir)
    if not os.path.isdir(resolved):
        report_fn({"ERROR"}, f"Could not resolve export directory: {resolved}")
        return None

    return resolved


def _default_image_extension(image):
    name = image.name.upper()
    file_format = getattr(image, "file_format", "").upper()
    if "POS" in name or file_format == "OPEN_EXR":
        return ".exr"
    if file_format == "PNG":
        return ".png"
    return ".png"


def _export_vat_images(pos_img, vat_color_img, export_dir):
    os.makedirs(export_dir, exist_ok=True)

    pos_path = os.path.join(export_dir, f"{pos_img.name}{_default_image_extension(pos_img)}")
    vat_color_path = os.path.join(
        export_dir, f"{vat_color_img.name}{_default_image_extension(vat_color_img)}"
    )

    _link_image_to_file(pos_img, pos_path, "OPEN_EXR")
    _link_image_to_file(vat_color_img, vat_color_path, "PNG")

    return pos_path, vat_color_path


def _iter_cat_vat_images():
    for img in bpy.data.images:
        name = img.name.upper()
        if "VAT" in name or name.endswith("_CAT"):
            yield img


class CSVVA_OT_GenerateImages(Operator):
    bl_idname = "csvva.generate_images"
    bl_label = "Generate CAT/VAT Images"
    bl_description = "Create CAT and VAT images for the checked entries in the list"

    def execute(self, context):
        prefs = context.scene.csvva_props
        fps = context.scene.render.fps

        blend_path = bpy.data.filepath
        if not blend_path:
            self.report({"ERROR"}, "Please save the .blend file first")
            return {"CANCELLED"}

        export_dir = bpy.path.abspath("//")
        if not os.path.isdir(export_dir):
            self.report({"ERROR"}, "Could not resolve the .blend directory")
            return {"CANCELLED"}

        targets = [item for item in prefs.preview_items if item.checked]
        if not targets:
            self.report({"ERROR"}, "No checked entries in the list")
            return {"CANCELLED"}

        created = []
        for item in targets:
            tracks = build_tracks_from_folder(item.folder)
            if not tracks:
                self.report({"WARNING"}, f"No tracks found in {item.folder}")
                continue

            try:
                pos_img, vat_col_img, pos_min, pos_max, _duration, _drone_count = (
                    csv_vat_gn.build_vat_images_from_tracks(
                        tracks, fps, image_name_prefix=f"{item.name}_VAT"
                    )
                )
            except Exception as exc:
                self.report({"ERROR"}, f"VAT generation failed for {item.name}: {exc}")
                continue

            bounds_suffix = _format_bounds_suffix(pos_min, pos_max)
            vat_base = f"{item.name}_VAT_{bounds_suffix}"

            pos_img.name = f"{vat_base}_Pos"
            vat_col_img.name = f"{vat_base}_Color"

            pos_path = os.path.join(export_dir, f"{pos_img.name}.exr")
            vat_color_path = os.path.join(export_dir, f"{vat_col_img.name}.png")

            _save_image(pos_img, pos_path, "OPEN_EXR")
            _save_image(vat_col_img, vat_color_path, "PNG")

            created.append((item.name, pos_path, vat_color_path))

        if not created:
            self.report({"ERROR"}, "No images were generated")
            return {"CANCELLED"}

        details = []
        for name, pos_path, vat_color_path in created:
            details.append(
                f"{name}: VAT Pos -> {os.path.basename(pos_path)}, VAT Color -> {os.path.basename(vat_color_path)}"
            )
        self.report({"INFO"}, " | ".join(details))
        return {"FINISHED"}


class CSVVA_OT_PackImages(Operator):
    bl_idname = "csvva.pack_cat_vat_images"
    bl_label = "Pack CAT/VAT Images"
    bl_description = "Embed all CAT/VAT images into the current .blend file"

    def execute(self, context):
        packed = 0
        for img in _iter_cat_vat_images():
            try:
                img.pack()
                packed += 1
            except Exception:
                continue

        if not packed:
            self.report({"ERROR"}, "No CAT/VAT images found to pack")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Packed {packed} CAT/VAT image(s)")
        return {"FINISHED"}


class CSVVA_OT_UnpackImages(Operator):
    bl_idname = "csvva.unpack_cat_vat_images"
    bl_label = "Unpack CAT/VAT Images"
    bl_description = "Save all CAT/VAT images next to the .blend file and link them"

    def execute(self, context):
        export_dir = _ensure_export_directory(self.report, "//")
        if not export_dir:
            return {"CANCELLED"}

        saved = 0
        for img in _iter_cat_vat_images():
            ext = _default_image_extension(img)
            target_path = bpy.path.abspath(img.filepath or img.filepath_raw)
            if not target_path:
                target_path = os.path.join(export_dir, f"{img.name}{ext}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            file_format = "OPEN_EXR" if ext == ".exr" else "PNG"
            _link_image_to_file(img, target_path, file_format)
            try:
                img.unpack(method='USE_ORIGINAL')
            except Exception:
                pass
            saved += 1

        if not saved:
            self.report({"ERROR"}, "No CAT/VAT images found to unpack")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Unpacked {saved} CAT/VAT image(s) to {export_dir}")
        return {"FINISHED"}


class CSVVA_OT_Preview(Operator):
    bl_idname = "csvva.preview"
    bl_label = "Preview"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)
        storyboard = context.scene.skybrush.storyboard
        prefs.preview_items.clear()

        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}

        fps = context.scene.render.fps
        metadata_map, metadata_defaults = load_import_metadata(folder, self.report)
        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        ordered_subdirs = list(subdirs)
        if metadata_map:
            ordered_subdirs = []
            subdir_set = set(subdirs)
            for key, meta in sorted(metadata_map.items(), key=lambda item: item[1]["id"]):
                if key in subdir_set:
                    ordered_subdirs.append(key)
                else:
                    self.report({"WARNING"}, f"No folder matched metadata key '{key}'")
            for d in subdirs:
                if d not in metadata_map:
                    ordered_subdirs.append(d)

        target_folders = []
        if ordered_subdirs:
            target_folders = [os.path.join(folder, d) for d in ordered_subdirs]
        else:
            target_folders = [folder]

        base_start = 0
        if storyboard.entries:
            last = storyboard.entries[-1]
            base_start = last.frame_start + last.duration
        if metadata_defaults.get("start_frame") is not None:
            base_start = metadata_defaults["start_frame"]

        next_start = base_start
        created = 0
        fps = context.scene.render.fps
        for path in target_folders:
            folder_name = os.path.basename(os.path.normpath(path))
            base_name, gap_frames = split_name_and_gap(folder_name, fps)
            meta = metadata_map.get(folder_name, {}) if metadata_map else {}
            if meta:
                gap_frames = _metadata_transition_duration(meta, default=gap_frames)
            display_name = _storyboard_name(base_name, meta)
            sf_meta = meta.get("start_frame", None) if meta else None
            start_frame = sf_meta if sf_meta is not None else next_start
            mid_duration = max(1, int(meta.get("middur", 1) or 1))
            midpose_enabled = bool(meta.get("midpose", True))
            # Preview only; handles are informational, shaping occurs on import/update
            f_handle = _metadata_handle(meta, "fhandle", None)
            m_handle = _metadata_handle(meta, "mhandle", None)
            ydepth = meta.get("ydepth", None)
            traled = bool(meta.get("traled", False))
            tracolor = _hex_to_rgba(meta.get("tracolor"))

            tracks = build_tracks_from_folder(path)
            duration = calculate_duration_from_tracks(tracks, fps)
            if duration == 0:
                vat_assets = _load_vat_assets(path)
                if vat_assets is not None:
                    duration = vat_assets[4]
                elif _has_vat_cat_images(path):
                    duration = DEFAULT_FOLDER_DURATION
            if duration == 0:
                continue

            existing_entry = None
            for sb in storyboard.entries:
                if sb.name == display_name:
                    existing_entry = sb
                    break

            frame_mismatch = False
            checked = False
            display_duration = duration
            if existing_entry:
                start_frame = existing_entry.frame_start
                frame_mismatch = existing_entry.duration != duration
                display_duration = existing_entry.duration
                checked = frame_mismatch
            else:
                checked = True

            item = prefs.preview_items.add()
            item.name = display_name
            item.folder = path
            item.start_frame = start_frame
            item.duration = display_duration
            item.midpose = midpose_enabled
            item.midpose_duration = mid_duration
            # Store handles for display/reference if needed later
            if f_handle is not None:
                item["fhandle"] = f_handle
            if m_handle is not None:
                item["mhandle"] = m_handle
            if ydepth is not None:
                try:
                    item["ydepth"] = float(ydepth)
                except Exception:
                    pass
            item["traled"] = traled
            if tracolor:
                item["tracolor"] = meta.get("tracolor")
            item.checked = checked
            item.exists = existing_entry is not None
            item.frame_mismatch = frame_mismatch
            created += 1

            next_start = max(next_start, start_frame + display_duration + gap_frames)

        if not created:
            self.report({"ERROR"}, "No valid CSV folders found")
            return {"CANCELLED"}

        self.report({"INFO"}, "Preview generated")
        return {"FINISHED"}


class CSVVA_OT_Update(Operator):
    bl_idname = "csvva.update"
    bl_label = "Update"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = context.scene.csvva_props
        storyboard = context.scene.skybrush.storyboard
        checked_items = [item for item in prefs.preview_items if item.checked]

        export_dir = None
        if prefs.use_vat and prefs.export_images:
            export_dir = _ensure_export_directory(self.report, "//")
            if not export_dir:
                return {"CANCELLED"}

        if not checked_items:
            self.report({"ERROR"}, "No items selected for update")
            return {"CANCELLED"}

        processed = 0
        key_data_collection = []
        fps = context.scene.render.fps

        for item in checked_items:
            existing_entry = None
            existing_index = None
            for idx, sb in enumerate(storyboard.entries):
                if sb.name == item.name:
                    existing_entry = sb
                    existing_index = idx
                    break

            target_start = item.start_frame
            keep_start = None
            keep_duration = None
            if existing_entry:
                keep_start = existing_entry.frame_start
                keep_duration = existing_entry.duration
                target_start = keep_start if not item.frame_mismatch else item.start_frame

            tracks = build_tracks_from_folder(item.folder)
            duration = calculate_duration_from_tracks(tracks, fps)
            if duration == 0:
                self.report({"WARNING"}, f"No CSV/TSV files found in {item.folder}")
                continue

            if existing_entry:
                clear_drone_keys(existing_entry.frame_start, existing_entry.duration)
            _remove_objects_for_storyboard(item.name)
            _remove_light_effect_entries(context.scene, item.name)
            _remove_vat_images_for_storyboard(item.name)
            if existing_index is not None:
                try:
                    storyboard.entries.remove(existing_index)
                except Exception:
                    pass

            obj, imported_duration, key_entries = import_csv_folder(
                context,
                item.folder,
                target_start,
                use_vat=prefs.use_vat,
                image_export_dir=export_dir,
            )
            if not obj:
                self.report({"WARNING"}, f"Failed to import from {item.folder}")
                continue

            entry = storyboard.entries[-1]
            entry_index = len(storyboard.entries) - 1
            if existing_entry is not None and existing_index is not None:
                entry_index = min(existing_index, len(storyboard.entries) - 1)
                storyboard.entries.move(len(storyboard.entries) - 1, entry_index)
                if not item.frame_mismatch:
                    entry.frame_start = keep_start
                    entry.duration = keep_duration
                else:
                    entry.frame_start = item.start_frame
                    entry.duration = duration

                old_end = (keep_start or 0) + (keep_duration or 0)
                new_end = entry.frame_start + entry.duration
                _shift_subsequent_storyboard_entries(storyboard, entry_index, new_end - old_end)

            if key_entries:
                key_data_collection.append((key_entries, entry.frame_start))
            processed += 1

        if not processed:
            self.report({"ERROR"}, "Nothing was updated")
            return {"CANCELLED"}

        current_frame = context.scene.frame_current
        for key_entries, start_frame in key_data_collection:
            context.scene.frame_set(start_frame)
            apply_color_keys_from_key_data(
                key_entries,
                start_frame,
            )
        context.scene.frame_set(current_frame)

        _update_frame_range_from_storyboard(context)

        self.report(
            {"INFO"},
            f"Updated {processed} formation(s). Recalculate transitions manually if needed.",
        )
        return {"FINISHED"}

class CSVVA_OT_ApplyTransitionMetadata(Operator):
    bl_idname = "csvva.apply_transition_metadata"
    bl_label = "Apply Transition Metadata"
    bl_description = "Apply transition-related metadata to existing light effects"

    def execute(self, context):
        prefs = context.scene.csvva_props
        folder = bpy.path.abspath(prefs.folder)
        if not os.path.isdir(folder):
            self.report({"ERROR"}, "Invalid CSV folder")
            return {"CANCELLED"}

        metadata_map, _metadata_defaults = load_import_metadata(folder, self.report)
        if not metadata_map:
            self.report({"WARNING"}, "No metadata found to apply")
            return {"CANCELLED"}

        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        ordered_subdirs = []
        subdir_set = set(subdirs)
        for key, meta in sorted(metadata_map.items(), key=lambda item: item[1]["id"]):
            if key in subdir_set:
                ordered_subdirs.append(key)
            else:
                self.report({"WARNING"}, f"No folder matched metadata key '{key}'")
        for d in subdirs:
            if d not in metadata_map:
                ordered_subdirs.append(d)

        pending_sampled_transitions: list[dict] = []
        transition_color_cache: dict[str, list[tuple[float, float, float, float]]] = {}
        last_transition_colors: list[tuple[float, float, float, float]] | None = None
        processed = 0

        for idx, d in enumerate(ordered_subdirs):
            meta = metadata_map.get(d, {})
            if not meta.get("traled", False):
                continue

            base_name, _gap_frames = split_name_and_gap(d, context.scene.render.fps)
            display_name = _storyboard_name(base_name, meta)
            applied_from_pending = _apply_pending_sampled_transitions(
                context,
                {display_name, base_name},
                pending_sampled_transitions,
                color_cache=transition_color_cache,
            )
            if applied_from_pending:
                last_transition_colors = applied_from_pending

            transition_name = f"{display_name}_TransitionLE"
            le_entry = _find_light_effect_entry(context.scene, transition_name)
            if le_entry is None:
                continue

            sample_mode, sample_count = _sample_info_from_tracolor(meta.get("tracolor"))
            tracolor = _hex_to_rgba(meta.get("tracolor")) or (1.0, 1.0, 1.0, 1.0)

            start, duration = _adjust_transition_timing(
                getattr(le_entry, "frame_start", 0), getattr(le_entry, "duration", 1), meta
            )
            le_entry.frame_start = start
            le_entry.duration = duration

            applied_colors = None
            if sample_mode == "presampled":
                source_colors = last_transition_colors or transition_color_cache.get(transition_name)
                _apply_transition_metadata(le_entry, meta)
                if source_colors:
                    applied_colors = _apply_transition_metadata(
                        le_entry,
                        meta,
                        ramp_colors=source_colors,
                        black_endpoints=True,
                    )
            elif sample_mode == "sampled" and sample_count:
                _apply_transition_metadata(le_entry, meta)
                if idx < len(ordered_subdirs) - 1:
                    next_base_name, _ = split_name_and_gap(
                        ordered_subdirs[idx + 1], context.scene.render.fps
                    )
                    next_display_name = _storyboard_name(
                        next_base_name, metadata_map.get(ordered_subdirs[idx + 1], {})
                    )
                    targets = {next_display_name, next_base_name}
                    pending_sampled_transitions.append(
                        {
                            "transition_name": transition_name,
                            "sample_count": sample_count,
                            "target_candidates": targets,
                            "black_edges": True,
                            "meta": meta,
                        }
                    )
                    applied_colors = _apply_pending_sampled_transitions(
                        context,
                        targets,
                        pending_sampled_transitions,
                        color_cache=transition_color_cache,
                    )
            else:
                applied_colors = _apply_transition_metadata(
                    le_entry,
                    meta,
                    ramp_colors=[tracolor],
                    black_endpoints=bool(sample_mode),
                )

            if applied_colors:
                transition_color_cache[transition_name] = applied_colors
                last_transition_colors = applied_colors

            processed += 1

        if processed == 0:
            self.report({"WARNING"}, "No transition metadata applied")
            return {"CANCELLED"}

        if pending_sampled_transitions:
            self.report({"INFO"}, "Some sampled transitions will update after target effects are available")

        return {"FINISHED"}


class CSVVA_PT_UI(Panel):
    bl_label = "CSV Vertex Anim"
    bl_idname = "CSVVA_PT_UI"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SBUtil"

    def draw(self, context):
        lay = self.layout
        prefs = context.scene.csvva_props
        col = lay.column(align=True)
        col.prop(prefs, "folder")
        col.prop(prefs, "use_vat")
        col.prop(prefs, "export_images")
        col.operator(CSVVA_OT_PrepareFolders.bl_idname, icon="FILE_FOLDER")
        row = col.row(align=True)
        row.operator(CSVVA_OT_Import.bl_idname, icon="IMPORT")
        row.operator(CSVVA_OT_Preview.bl_idname, icon="VIEWZOOM")
        col.template_list(
            CSVVA_UL_Preview.bl_idname,
            "",
            prefs,
            "preview_items",
            prefs,
            "preview_index",
            rows=4,
        )
        col.operator(CSVVA_OT_GenerateImages.bl_idname, icon="IMAGE_DATA")
        row = col.row(align=True)
        row.operator(CSVVA_OT_PackImages.bl_idname, icon="PACKAGE")
        row.operator(CSVVA_OT_UnpackImages.bl_idname, icon="FILE_IMAGE")
        col.operator(CSVVA_OT_ApplyTransitionMetadata.bl_idname, icon="KEYFRAME_HLT")
        col.operator(CSVVA_OT_Update.bl_idname, icon="FILE_REFRESH")


# ---------- Registration ----------

classes = (
    CSVVA_PreviewItem,
    CSVVA_Props,
    CSVVA_UL_Preview,
    CSVVA_OT_PrepareFolders,
    CSVVA_OT_Import,
    CSVVA_OT_GenerateImages,
    CSVVA_OT_PackImages,
    CSVVA_OT_UnpackImages,
    CSVVA_OT_Preview,
    CSVVA_OT_ApplyTransitionMetadata,
    CSVVA_OT_Update,
    CSVVA_PT_UI,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.csvva_props = bpy.props.PointerProperty(type=CSVVA_Props)


def unregister():
    del bpy.types.Scene.csvva_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
