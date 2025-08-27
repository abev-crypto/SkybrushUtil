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

## Updating

In **Edit > Preferences > Add-ons**, open the Skybrush Util entry and use the
**Update Add-on** button to automatically check GitHub for a newer release. If a
newer version is available, it is downloaded, installed, and re-enabled inside
Blender.
