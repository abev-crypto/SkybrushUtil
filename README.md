# Skybrush Util

Blender add-on for transferring keyframes and light effects.

## Features

- Save keyframe and light effect data to JSON files for reuse.
- Load stored keys and effects back into a Blender scene.
- Append collections and textures from other .blend files.
- Batch import data for multiple storyboard entries with a single command.
- Recalculate transitions while preserving material color keys.
- Drive light effects with ColorRamps or custom Python functions.

## Custom color functions

Custom Python modules may define uppercase constants that show up as
configuration options in Blender. The names of these constants must not clash
with existing RNA properties of the light effect. Names such as `name`, `type`,
`id_data` or `rna_type` are reserved and will be ignored if defined.

Constants may also carry special suffixes that determine how their values are
interpreted:

- `_COLOR` – display the value as a color picker.
- `_POS` – interpret the value as an XYZ position. An additional property
  named `<name>_object` is created to allow referencing a Blender object. When
  set, the object's world position overrides the constant.
- `_ROT` – similar to `_POS`, but uses the object's rotation in Euler angles.
- `_SCL` – similar to `_POS`, but uses the object's scale.
- `_MAT` – similar to `_POS`, but uses the base color of the object's first
  material (if any).

## Updating

In **Edit > Preferences > Add-ons**, open the Skybrush Util entry and use the
**Update Add-on** button to automatically check GitHub for a newer release. If a
newer version is available, it is downloaded, installed, and re-enabled inside
Blender.
