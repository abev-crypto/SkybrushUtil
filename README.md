# SkybrushUtil

SkybrushUtil is a Blender add-on that extends the [Skybrush](https://skybrush.io) ecosystem with advanced utilities for drone show content creation.  
It provides non-destructive JSON workflows, animation transfer tools, LightEffect extensions, and CSV vertex animation export for large-scale drone formations.

---

## 📥 Installation

1. Download the latest release from [Releases](https://github.com/abev-crypto/SkybrushUtil/releases).  
   (ZIP file recommended)  
2. In Blender, go to **Edit → Preferences → Add-ons → Install…**  
3. Select the downloaded `.zip` file.  
4. Enable **SkybrushUtil** in the add-on list.  
5. A new **SBUtil** tab will appear in the Sidebar.

---

## 🚀 Features

### 1. Non-destructive JSON Workflow
- Merge two Blender scenes using JSON data.  
- Supports **Save/Load** operations that preserve animation, duration, and naming conventions.  
- Prevents data loss when transferring between source and target `.blend` files.  
- Prefix handling ensures unique asset names.

### 2. Storyboard Integration
- Import storyboard information directly into the add-on.  
- `Refresh` adjusts keyframes when the **StartFrame** changes.  
- `GotoStart` jumps the current frame to the storyboard’s defined start.

### 3. Shift & Re-Time Animations
- Add storyboard elements to a **ShiftList**.  
- Apply **ShiftFrame** to move animations in bulk.  
- Automatically generates `_Animated` collections for retimed assets.  
- Warning: manual edits to SBUtil’s list are not recommended.

### 4. CSV Export & Vertex Animation
- Export drone motion into `.csv` for **Skybrush Studio**.  
- Supports destructive workflow with **Export to Skybrush .csv**.  
- `CSV Vertex Anim` generates per-vertex animations with independent joints.  
- Root folder batch imports supported.  
- Recommended frame rate: `1 fps`.

### 5. LightEffect Extensions
- Gradient looping and ramp baking.  
- Custom function embedding via external `.py` files.  
- Extended target selection for collections (beyond Inside/Outside Mesh).  
- `Embed` / `Unembed` system for script portability.  
- Supports suffix-based properties (`_COLOR`, `_POS`, `_ROT`, `_SCL`, `_MAT`).

### 6. Utility Tools
- **ApplyProximityLimit**: auto-limit distance during storyboard range.
- **RemoveProximityLimit**: clear all distance constraints.
- **LinearizeCopyLoc**: makes CopyLocation curves linear.
- **Stagger CopyLoc Transition**: offsets Copy Location influence keys per drone to stagger transitions.
- **ReflowVertices**: smooth vertex distribution while respecting axis locks.

---

## ⚠️ Limitations

- Constraint keys are not exportable (auto-generated and unpredictable).  
- Manual addition to SBUtil’s list is not recommended.  
- LightEffect deletion removes required textures.  
- `_Animated` collection names must not be changed.  
- Vertex reflow is limited to non-linked index recalculations.  
- CSV export is the recommended way to share results externally.

---

## 🔄 Updating

1. Open **Preferences → Add-ons → SkybrushUtil**.  
2. Press the **Update** button to refresh files.  
3. If patching fails when starting Blender from a `.blend` file, reload the add-on manually.

---

## 📌 Notes

- Use **CSV export** when sharing scenes with external collaborators.  
- Non-destructive workflows are designed for internal production; destructive CSV pipelines are recommended for delivery.  
- Animation and color data are fully baked during export.

---

## 📂 Repository

- GitHub: [abev-crypto/SkybrushUtil](https://github.com/abev-crypto/SkybrushUtil)
