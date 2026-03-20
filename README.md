# S42 CutFlow — Video Editing Suite for ComfyUI

**42 core + 10 expanded nodes for professional video editing within ComfyUI's node graph.** CapCut-class editing power with zero GPU model dependencies. Trim, split, speed ramp, filter, color grade, overlay text, picture-in-picture, stabilize, beat-sync, background removal, multi-layer compositing, and LTX 2.3 bridge utilities — all within your ComfyUI workflow.

> **DON'T PANIC** — Every node works on standard ComfyUI IMAGE batches. Connect to any video loader (VHS/GGF) and any video output node. No special formats, no VRAM pressure.

---

## Table of Contents

- [Features at a Glance](#features-at-a-glance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Node Reference — Core (42)](#node-reference--core-42)
  - [Clip Operations](#clip-operations)
  - [Filters & Color](#filters--color)
  - [Text & Overlays](#text--overlays)
  - [Composition & Motion](#composition--motion)
  - [Audio Sync](#audio-sync)
  - [Preview & Analysis](#preview--analysis)
  - [Utilities](#utilities)
- [Node Reference — Expanded (10)](#node-reference--expanded-10)
  - [Background Remover](#background-remover)
  - [Layer Composer](#layer-composer)
  - [LTX 2.3 Bridge](#ltx-23-bridge)
- [Example Workflows](#example-workflows)
- [Preview Workflow Tips](#preview-workflow-tips)
- [Dependencies](#dependencies)
- [Compatibility](#compatibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features at a Glance

| Category | Nodes | What You Get |
|----------|-------|-------------|
| **Clip Ops** | 8 | Trim, split, loop, freeze, extract, insert, reverse, speed ramp |
| **Filters** | 7 | Sharpen, blur, denoise, vignette, film grain, chromatic aberration, lens distortion, style presets, color adjust |
| **Temporal FX** | 3 | Echo/ghosting, motion trails, glitch effects, frame blending |
| **Text** | 4 | Text overlay, subtitle burn (SRT), watermarks, animated lower thirds |
| **Composition** | 6 | Picture-in-picture, split screen, Ken Burns, crop/pad, mask wipes, shape masks |
| **Stabilization** | 1 | OpenCV feature-tracking video stabilization |
| **Audio Sync** | 3 | Audio trim, beat detection, audio-video duration matching |
| **Preview** | 5 | Quick preview (480p), filmstrip thumbnails, clip info, A/B compare, histogram/waveform overlay |
| **Utilities** | 5 | Aspect ratio conversion, batch resize, channel ops, image→clip, clip→GIF |
| ***Core Total*** | ***42*** | *Complete video editing pipeline* |
| | | |
| **BG Remover** *(expanded)* | 1 | BiRefNet, RMBG-2.0, rembg, chroma/luma key — batch video with temporal consistency |
| **Layer Composer** *(expanded)* | 1 | 5-layer compositor with 14 blend modes, foreground placement, alpha compositing |
| **LTX 2.3 Bridge** *(expanded)* | 8 | Guide frame prep, frame calculator, subject isolate, background plate, segment prep, audio conditioning, post-process, scene describer |
| ***Expanded Total*** | ***10*** | *AI generation integration* |
| **Grand Total** | **52** | **Complete editing + generation pipeline** |

**Zero GPU Models for core editing.** All 42 core operations run on CPU via torch, numpy, PIL, and OpenCV. Expanded nodes optionally use BiRefNet/rembg for AI background removal.

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for **S42 CutFlow**
3. Click **Install**
4. Restart ComfyUI

### Method 2: Git Clone

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/GeekyGhost/S42-CutFlow.git
```

Then install dependencies:

**Windows (Portable):**
```batch
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\S42-CutFlow\requirements.txt
```

**Linux / venv:**
```bash
pip install -r ComfyUI/custom_nodes/S42-CutFlow/requirements.txt
```

### Method 3: Manual Download

1. Download the ZIP from [GitHub Releases](https://github.com/GeekyGhost/S42-CutFlow/releases)
2. Extract to `ComfyUI/custom_nodes/S42-CutFlow/`
3. Install requirements (see Method 2)
4. Restart ComfyUI

### Verify Installation

After restart, look for nodes under the **S42 CutFlow** category in the node menu. You should see 52 nodes across core and expanded subcategories. The console will print:

```
[S42 CutFlow] Loaded 42 core + 10 expanded = 52 total nodes
```

---

## Quick Start

### Basic Trim + Filter Workflow

```
[Load Video (VHS)] → [S42CF Quick Preview] → [S42CF Smart Trim] → [S42CF Filter Pack] → [S42CF Style Transfer] → [VHS Video Combine]
```

1. **Load** a video with VHS/GGF Load Video
2. **Preview** — add Quick Preview node set to "preview" mode for fast 480p iteration
3. **Trim** — use Smart Trim to cut to your in/out points
4. **Filter** — apply sharpening, denoise, or blur
5. **Style** — add a cinematic color grade
6. **Export** — connect to VHS Video Combine

### Speed Ramp Edit

```
[Load Video] → [S42CF Speed Ramp (preset: bullet_time)] → [S42CF Vignette] → [Video Combine]
```

### Beat-Synced Montage

```
[Load Audio] → [S42CF Beat Snap] → 
[Load Video] → [S42CF Multi Split (split_points from BeatSnap)] → [S42CF Transition] → [Video Combine]
```

---

## Node Reference — Core (42)

Every parameter on every node has a tooltip — hover over any input in ComfyUI to see its description.

---

### Clip Operations

#### ✂ S42CF Smart Trim

Trim a clip to in/out points with frame or timecode precision.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Video clip to trim |
| `trim_unit` | select | frames | `frames` = frame indices, `seconds` = time-based |
| `in_point` | float | 0.0 | Start point (frame index or seconds) |
| `out_point` | float | -1.0 | End point (-1 = end of clip) |
| `fps` | float | 24.0 | FPS for seconds→frames conversion |

**Outputs:** `clip` (IMAGE), `frame_count` (INT), `info` (STRING)

---

#### ✂ S42CF Multi Split

Split a clip at multiple points into segments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Video clip to split |
| `split_points` | string | "24, 48" | Comma-separated frame indices |
| `segment_index` | int | 0 | Which segment to output (0-based) |

**Tip:** Connect the output of Beat Snap directly to `split_points` for beat-synced cuts.

---

#### 🔁 S42CF Loop Bounce

Loop a clip with optional ping-pong and crossfade.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Clip to loop |
| `loops` | int | 2 | Number of plays (1 = passthrough) |
| `mode` | select | loop | `loop` or `ping_pong` (forward-reverse alternating) |
| `crossfade_frames` | int | 0 | Crossfade at seam points (0 = hard cut) |

---

#### ⏸ S42CF Frame Hold

Freeze a specific frame for N frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Source clip |
| `frame_source` | select | first | `first`, `last`, `middle`, or `custom` |
| `frame_index` | int | 0 | Frame index (only for `custom` source) |
| `hold_frames` | int | 24 | Duration of freeze (24 = 1 second at 24fps) |

---

#### 📋 S42CF Frame Extract

Extract specific frames by index, range, or interval.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | interval | `indices`, `interval`, or `range` |
| `index_list` | string | "0, 12, 24" | Frame numbers for `indices` mode |
| `interval` | int | 4 | Every Nth frame for `interval` mode |
| `range_start` / `range_end` | int | 0 / -1 | Range bounds (-1 = end) |

---

#### 📥 S42CF Frame Insert

Insert frames at a specific position within a clip.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Main clip |
| `insert_frames` | IMAGE | — | Frames to insert (auto-resolution matched) |
| `insert_at` | int | 0 | Insertion index (0 = prepend) |

---

#### ⏪ S42CF Clip Reverse

Reverse a clip with optional speed modification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speed` | float | 1.0 | Speed after reversing (0.5 = slow-mo reverse, 2.0 = fast) |
| `interpolation` | select | blend | `blend` (smooth) or `duplicate` (fast) |

---

#### ⏩ S42CF Speed Ramp

CapCut-style speed ramping with presets and custom keyframe curves.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | select | custom | `custom`, `montage`, `bullet_time`, `hero_moment`, `whip_pan`, `ramp_up`, `ramp_down` |
| `speed_curve` | string | "0:1.0, ..." | Keyframe string: `frame:speed:easing` |
| `global_speed` | float | 1.0 | Multiplier on top of curve |
| `interpolation` | select | blend | Frame interpolation method |

**Speed curve format:** `frame:multiplier:easing, frame:multiplier:easing`
- `0:1.0, 12:0.25:ease_in, 36:0.25, 48:1.0:ease_out` = normal → slow-mo → normal
- Speed values: `0.1` = very slow, `0.5` = half, `1.0` = normal, `2.0` = double

**Available easings:** `linear`, `ease_in`, `ease_out`, `ease_in_out`, `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `elastic_in`, `elastic_out`, `bounce`

---

### Filters & Color

#### 🔧 S42CF Filter Pack

Multi-mode filter with 7 operations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | sharpen | `sharpen`, `unsharp_mask`, `gaussian_blur`, `box_blur`, `motion_blur`, `median_denoise`, `bilateral_denoise` |
| `intensity` | float | 1.0 | Strength (meaning varies by mode) |
| `radius` | int | 3 | Kernel radius (must be odd) |
| `angle` | float | 0.0 | Direction for motion_blur (degrees) |

---

#### 🔲 S42CF Vignette

Configurable vignette with shape, color, and offset controls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intensity` | float | 0.5 | Darkness (0-2) |
| `softness` | float | 0.5 | Fade gradient (0.1=sharp, 1.0=soft) |
| `shape` | select | circle | `circle`, `oval`, `rectangle` |
| `offset_x` / `offset_y` | float | 0.0 | Center offset (-1 to 1) |
| `color_r/g/b` | float | 0.0 | Vignette color (default = black) |

---

#### 🎞 S42CF Film Grain

Procedural film grain with size, color mode, and animation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intensity` | float | 0.15 | Grain strength (0.05=subtle, 0.5=heavy) |
| `grain_size` | float | 1.0 | Particle size (1.0=fine, 4.0=coarse) |
| `color_mode` | select | mono | `mono` (classic) or `color` (chromatic) |
| `animated` | select | yes | Per-frame unique grain or static |
| `seed` | int | 42 | Random seed |

---

#### 🌈 S42CF Chromatic Aberration

RGB channel offset simulating lens fringing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strength` | float | 3.0 | Pixel offset (1-3=subtle, 10+=extreme) |
| `mode` | select | radial | `radial` (realistic) or `linear` (artistic) |
| `angle` | float | 0.0 | Direction angle in degrees |

---

#### 🔍 S42CF Lens Distortion

Barrel/pincushion distortion and tilt-shift miniature effect.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | barrel | `barrel`, `pincushion`, `tilt_shift` |
| `strength` | float | 0.3 | Distortion amount |
| `focus_position` | float | 0.5 | Tilt-shift focus band Y (0=top, 1=bottom) |
| `focus_width` | float | 0.3 | Sharp band width |

---

#### 🎨 S42CF Style Transfer

8 LUT-free color style presets using pure numpy math.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | select | cinematic_teal_orange | `cinematic_teal_orange`, `vintage`, `noir`, `polaroid`, `cyberpunk`, `pastel`, `bleach_bypass`, `cross_process` |
| `strength` | float | 1.0 | Blend (0=original, 1=full effect) |

---

#### ☀ S42CF Color Adjust

Fundamental brightness, contrast, saturation, hue, gamma, and temperature controls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `brightness` | float | 0.0 | Brightness offset (-1 to +1). Typical: -0.2 to +0.2 |
| `contrast` | float | 1.0 | Contrast multiplier (0=flat, 1=normal, 1.5=punchy) |
| `saturation` | float | 1.0 | Color saturation (0=gray, 1=normal, 1.5=vivid) |
| `hue_shift` | float | 0.0 | Hue rotation in degrees (-180 to +180) |
| `gamma` | float | 1.0 | Gamma correction (<1=lift shadows, >1=darken mids) |
| `temperature` | float | 0.0 | Warm/cool shift (-1=cool/blue, +1=warm/orange) |

---

### Temporal FX

#### ⏳ S42CF Temporal FX

Time-domain effects across frame windows.

| Mode | Description |
|------|-------------|
| `echo` | Semi-transparent past frames layered on current |
| `motion_trail` | Accumulating trail from movement |
| `time_displacement` | Vertical rows sample different time offsets |
| `frame_average` | Average N adjacent frames (dreamy/ghostly) |
| `strobe` | Alternating frame hold (strobe/flash) |

---

#### 💥 S42CF Glitch FX

Digital glitch effects with 5 modes.

| Mode | Description |
|------|-------------|
| `datamosh` | Frame bleed between consecutive frames |
| `scan_lines` | Horizontal line shift artifacts |
| `pixel_sort` | Sort pixels by brightness in bands |
| `rgb_split` | Random RGB channel displacement |
| `block_corrupt` | Random block displacement |

---

#### 🔄 S42CF Frame Blend

Blend adjacent frames for motion blur or AI artifact smoothing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window` | int | 3 | Frames to blend (2-3=subtle, 5+=heavy) |
| `weights` | select | center_weighted | `uniform`, `center_weighted`, `exponential_decay` |
| `strength` | float | 1.0 | Mix (0=original, 1=fully blended) |

---

### Text & Overlays

#### 📝 S42CF Text Overlay

Rich text rendering with stroke, shadow, background, and positioning.

| Key Parameters | Description |
|----------------|-------------|
| `text` | Multi-line text content |
| `font_size` | 8-400px |
| `position_x/y` | 0-1 normalized position |
| `stroke_width` | Outline thickness (0=none) |
| `shadow_offset` | Drop shadow (0=none) |
| `bg_padding` | Background box (0=none) |

---

#### 💬 S42CF Subtitle Burn

Burn SRT subtitles onto frames with style presets.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `srt_text` | string | — | Standard SRT format text |
| `fps` | float | 24.0 | For timestamp→frame conversion |
| `style` | select | default | `default`, `cinematic`, `youtube`, `outline_only` |
| `position_y` | float | 0.9 | Vertical position (0.9=near bottom) |

---

#### 🔖 S42CF Watermark

Image or text watermark with tiling support.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | text | `text` or `image` |
| `position` | select | bottom_right | 6 positions + `tile` for repeat |
| `opacity` | float | 0.3 | Watermark transparency |
| `watermark_image` | IMAGE (optional) | — | Image for `image` mode |

---

#### 📺 S42CF Lower Third

Animated lower-third title cards with slide-in/out.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "John Smith" | Primary name |
| `title` | string | "Lead Designer" | Subtitle |
| `style` | select | modern | `modern`, `minimal`, `broadcast`, `cinematic` |
| `start_frame` / `duration_frames` | int | 0 / 72 | Timing |
| `animate_in` / `animate_out` | int | 12 / 12 | Animation duration |
| `accent_r/g/b` | int | 255/153/0 | Accent color |

---

### Composition & Motion

#### 🖼 S42CF Picture-in-Picture

Overlay one clip onto another with border, shadow, and rounded corners.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position_x/y` | float | 0.75/0.25 | PiP center position |
| `scale` | float | 0.3 | PiP size (0.3 = 30% of background) |
| `border_width` | int | 2 | Border in pixels |
| `corner_radius` | int | 0 | Rounded corners |
| `shadow` | int | 0 | Drop shadow size |

---

#### ▦ S42CF Split Screen

2/3/4 way split with configurable layout.

| Layout | Description |
|--------|-------------|
| `horizontal_2` | Top/bottom split |
| `vertical_2` | Left/right split |
| `grid_4` | 2×2 grid (connect clip_c and clip_d) |
| `diagonal` | Diagonal split line |

---

#### 🎬 S42CF Ken Burns

Pan and zoom effect for photos or video.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_x/y` | float | 0.0 | Start pan position |
| `start_zoom` | float | 1.0 | Start zoom (1.0=full, 2.0=2× zoom) |
| `end_x/y` | float | 0.3/0.2 | End pan position |
| `end_zoom` | float | 1.5 | End zoom |
| `easing` | select | ease_in_out | Animation curve |

---

#### ✂ S42CF Crop & Pad

Crop or pad to dimensions/aspect ratio.

| Mode | Description |
|------|-------------|
| `crop` | Cut to exact dimensions from center |
| `pad_letterbox` | Add bars top/bottom |
| `pad_pillarbox` | Add bars left/right |
| `aspect_ratio` | Auto-detect best fit |

Presets: `16:9`, `9:16`, `4:3`, `1:1`, `21:9`, `2.35:1`

---

#### 🎭 S42CF Mask Wipe

Animated mask-based transitions between two clips.

| Wipe Type | Description |
|-----------|-------------|
| `gradient_left/right/up/down` | Linear wipe |
| `radial_in/out` | Circle from/to center |
| `diamond` | Diamond shape reveal |
| `clock` | Rotating clock wipe |

---

#### ⬡ S42CF Shape Mask

Generate animated shape masks for compositing.

| Shape | Description |
|-------|-------------|
| `rectangle` | Rounded rectangle mask |
| `ellipse` | Oval mask |
| `star` | 5-pointed star |
| `diamond` | Diamond shape |

Supports keyframe animation for `rotation` and `size` via keyframe strings.

---

#### 📐 S42CF Stabilize

Software video stabilization using OpenCV optical flow tracking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smoothing` | int | 15 | Smoothing window (larger=smoother) |
| `crop_mode` | select | auto_crop | `auto_crop`, `fill_border`, `none` |
| `max_shift` | float | 0.1 | Max correction (fraction of frame) |

**Requires:** `opencv-python`

---

### Audio Sync

#### 🔊 S42CF Audio Trim

Trim audio to match clip length with fade in/out.

| Mode | Description |
|------|-------------|
| `match_frames` | Trim to match `target_frames` at given `fps` |
| `custom_time` | Trim to `in_seconds` / `out_seconds` |
| `custom_frames` | Trim by frame numbers |

---

#### 🥁 S42CF Beat Snap

Detect beats and output frame indices for synced cuts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity` | float | 0.5 | Detection sensitivity (lower=fewer beats) |
| `min_interval` | float | 0.3 | Min seconds between beats |
| `output_format` | select | frame_indices | `frame_indices`, `split_points`, `keyframe_string` |

**Workflow:** Beat Snap → `beat_data` output → Multi Split `split_points` input

---

#### 🔗 S42CF Audio-Video Sync

Match audio duration to video duration.

| Method | Description |
|--------|-------------|
| `trim_pad` | Cut or silence-pad (preserves pitch) |
| `time_stretch` | Resample to fit (may change pitch slightly) |

---

### Preview & Analysis

#### 👁 S42CF Quick Preview

Downsample for fast iteration, passthrough for final render.

| Mode | Description |
|------|-------------|
| `preview` | Downsample to `preview_height` (default 480p) + optional frame skip |
| `passthrough` | No-op, full resolution passes through |

**Workflow tip:** Place at start of chain. Work in preview mode, switch to passthrough for final render.

---

#### 🎞 S42CF Thumbnail Strip

Generate a filmstrip contact sheet for visual review.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | int | 8 | Thumbnails per row |
| `thumb_height` | int | 120 | Thumbnail height (px) |
| `max_frames` | int | 24 | Max thumbnails (evenly sampled) |
| `show_frame_numbers` | select | yes | Overlay frame indices |

---

#### ℹ S42CF Clip Info

Output metadata: frame count, dimensions, duration, aspect ratio, brightness stats.

**Outputs:** `frame_count`, `width`, `height`, `duration_seconds`, `aspect_ratio`, `info`

---

#### ↔ S42CF Side-by-Side

A/B comparison of two clips with labels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | select | horizontal | `horizontal` or `vertical` |
| `label_a/b` | string | Before/After | Overlay labels |

---

#### 📊 S42CF Histogram Overlay

Burn monitoring graphics onto frames.

| Display | Description |
|---------|-------------|
| `histogram` | RGB histogram |
| `waveform` | Luminance waveform |
| `rgb_parade` | Side-by-side RGB waveforms |

---

### Utilities

#### 📐 S42CF Aspect Convert

Convert between aspect ratios.

Presets: `16:9`, `9:16`, `4:3`, `3:4`, `1:1`, `21:9`, `2.35:1`, `custom`

Fit modes: `letterbox`, `crop_center`, `crop_smart`, `stretch`

---

#### ↔ S42CF Batch Resize

Resize with 5 modes and 4 interpolation methods.

| Mode | Description |
|------|-------------|
| `exact` | Set width × height |
| `fit_width` / `fit_height` | Scale proportionally |
| `percentage` | Scale by % |
| `max_dimension` | Cap longest side |

Interpolation: `bilinear`, `nearest`, `bicubic`, `lanczos`

---

#### 🔴 S42CF Channel Ops

13 channel operations: extract, swap, grayscale, invert, threshold, math.

---

#### 🖼 S42CF Image → Clip

Convert a still image to an N-frame video clip.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_mode` | select | seconds | `frames` or `seconds` |
| `duration_seconds` | float | 2.0 | Duration |
| `fps` | float | 24.0 | For time conversion |

---

#### 🎁 S42CF Clip → GIF

Export an IMAGE batch as an optimized GIF.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fps` | float | 12.0 | GIF playback rate |
| `max_colors` | int | 256 | Palette size (16-256) |
| `dither` | select | yes | Floyd-Steinberg dithering |
| `optimize` | select | yes | File size optimization |

Saves to ComfyUI's output directory.

---

## Preview Workflow Tips

ComfyUI executes nodes sequentially, so true real-time preview isn't possible. Here's how to get the fastest iteration loop:

1. **Start with Quick Preview** — place it right after your video loader, set to `preview` mode at 480p. This makes every downstream node 4-8× faster.

2. **Use Thumbnail Strip** — place it at any point in your chain to see a contact sheet of the current state. Great for verifying trim points and transitions.

3. **Switch to passthrough** — when satisfied, change Quick Preview to `passthrough`. Full-res data flows through unchanged for final render.

4. **Use Side-by-Side** — connect your original and edited clips to compare before/after.

**Typical editing speed at 480p:** 48 frames (2 seconds @ 24fps) processes in 1-5 seconds depending on which filters are active.

---

## Dependencies

| Package | Required? | Already in ComfyUI? | Used For |
|---------|-----------|---------------------|----------|
| torch | Yes | ✅ Yes | All tensor ops |
| numpy | Yes | ✅ Yes | Array math, filters |
| Pillow (PIL) | Yes | ✅ Yes | Text, GIF, resampling |
| scipy | Yes | ✅ Yes | Signal processing |
| PyAV (av) | Yes | ✅ Yes (≥14.2) | Audio processing |
| opencv-python | Yes | ❌ Install | Stabilize, bilateral, filters, BG chroma key |
| transformers | Optional | 🔸 Often present | BiRefNet AI background removal |
| rembg | Optional | ❌ Optional | U2Net/ISNet/SILUETA background removal |
| numba | Optional | ❌ Optional | JIT speedup (not required) |

**Only `opencv-python` needs to be installed** for all 42 core nodes. The expanded Background Remover AI methods need `transformers` (often already installed by other nodes) or `rembg` — but traditional methods (chroma key, luma key) work with just opencv.

---

## Compatibility

- **ComfyUI:** Tested with latest portable build (Python 3.12 + PyTorch CUDA 12.6) and desktop (Python 3.13)
- **Python:** 3.10, 3.11, 3.12, 3.13
- **Video Loader:** Works with VHS/GGF Video Helper Suite, or any node that outputs IMAGE batches
- **Video Output:** Works with VHS Video Combine, or any node that accepts IMAGE batches
- **LTX 2.3:** Bridge nodes work with native ComfyUI LTX nodes (LTXVAddGuide, LTXVPreprocess, EmptyLTXVLatentVideo, etc.)
- **Other Nodes:** No conflicts with KJNodes, Impact Pack, ComfyUI-Essentials, ComfyUI-RMBG, or S42 Production Suite
- **Namespace:** All nodes prefixed `S42CF_` to avoid collisions

---

## Troubleshooting

### "opencv-python not installed"
```bash
# Windows Portable
.\python_embeded\python.exe -m pip install opencv-python

# Linux/Mac venv
pip install opencv-python
```

### Nodes not showing up
1. Check the console for import errors
2. Verify the folder is at `ComfyUI/custom_nodes/S42-CutFlow/` (not nested)
3. Ensure `__init__.py` exists in the folder
4. Restart ComfyUI completely (not just reload)

### Out of memory on long clips
- Use Quick Preview at 480p during editing
- Process in shorter segments with Multi Split
- Frame Blend and Temporal FX with large windows are memory-intensive

### GIF export not saving
- Check that ComfyUI has write permissions to its output directory
- Verify PIL/Pillow is installed correctly
- Check console for the exact file path in the info output

### Stabilize produces black borders
- Set `crop_mode` to `auto_crop` (default)
- Reduce `max_shift` if too much is being cropped
- Use `fill_border` mode to extend edges instead of cropping

### Background Remover shows no effect
- **AI methods:** Ensure `transformers` is installed (`pip install transformers`). BiRefNet models auto-download on first use — check console for download progress.
- **rembg methods:** Install rembg (`pip install rembg`). Models download to `~/.u2net/` or ComfyUI's `models/rembg/` folder.
- **Chroma key:** Make sure your footage has an actual green/blue screen. Adjust `chroma_threshold` (lower = stricter) and `chroma_softness`.
- Check `use_gpu` is set to "yes" for faster AI processing.

### Layer Composer output is black
- At least one layer must be connected. If only connecting foreground, make sure `bg_fill_color` is set to something visible (e.g. `#FFFFFF`).
- Check that fg_mask is properly connected — without it, foreground defaults to fully opaque which may obscure the canvas fill.

### LTX Bridge nodes not connecting to LTX nodes
- Bridge nodes output standard IMAGE/INT/STRING types. Connect GuideFramePrep outputs to LTXVPreprocess image inputs manually.
- LTXFrameCalc outputs INT — connect to EmptyLTXVLatentVideo's `length` input.
- The bridge nodes don't replace native LTX nodes, they prepare data for them.

### LTX 2.3 frame count errors
- LTX requires 8n+1 frame counts: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, ...
- Use LTXFrameCalc to automatically compute valid counts from desired duration.
- 121 frames = ~5s at 24fps. 257 frames = ~10.7s. 513 frames = ~21.4s.

---

## Node Reference — Expanded (10)

These nodes extend CutFlow with AI-powered background removal, professional multi-layer compositing, and LTX 2.3 video generation bridge utilities.

### Background Remover

#### 🎭 S42CF Background Remover

Professional background removal for images and video batches. Supports 16 AI methods (BiRefNet, rembg, U2Net, ISNet) plus 4 traditional methods (chroma/luma key). Batch-optimized with temporal consistency for video.

| Key Parameters | Description |
|----------------|-------------|
| `method` | AI model or traditional method. `birefnet-general` = best all-around. `birefnet-portrait` = people. `birefnet-hr` = high-res up to 2048px. `chroma_green/blue/red` = green/blue/red screen. `luma_key` = brightness-based. |
| `use_gpu` | GPU acceleration for AI models. Recommended. |
| `processing_resolution` | AI inference resolution. 1024 = default. 2048 = max quality (birefnet-hr). |
| `mask_blur` | Edge softness. 0 = sharp, 0.8 = smooth, 2+ = very soft. |
| `mask_dilation` / `mask_erosion` | Expand/shrink mask edges (pixels). Helps with hair (dilate) or fringe (erode). |
| `remove_small_objects` | Remove mask noise smaller than N pixels. 300 = good default. |
| `invert_mask` | Swap foreground/background. |
| `temporal_smooth` | **Video feature.** Blend adjacent masks to eliminate flicker. 0 = off, 1-5 = smoothing window. |
| `chroma_threshold` / `chroma_softness` / `spill_suppress` | Chroma key controls. Only for chroma_* methods. |
| `luma_mode` / `shadow_threshold` / `highlight_threshold` | Luma key controls. Remove shadows, highlights, or both extremes. |
| `reference_bg` *(optional)* | Clean background plate. Enhances AI methods with difference information. |

**Outputs:** `result_rgb` (subject on black), `result_rgba` (with alpha), `mask` (MASK), `info`

**AI Model Requirements:**
- **BiRefNet methods:** Require `transformers` package (installed by many ComfyUI nodes). Models auto-download from HuggingFace on first use.
- **rembg methods (u2net, isnet, silueta):** Require `rembg` package. Install: `pip install rembg`.
- **Traditional methods:** No additional packages beyond opencv-python.

---

### Layer Composer

#### 🎬 S42CF Layer Composer

Professional 5-layer video/image compositor with 14 blend modes. Enhanced from the S42P Layer Composer with two additional layers and full alpha-aware compositing.

**Layer stack (bottom to top):**
1. **Background** — always stretched to fill canvas. Connect video backgrounds, solid colors, gradients.
2. **Fill** — stretched to fill. For gradients, color washes, or secondary backgrounds.
3. **FX** — stretched to fill. Connect procedural FX, particles, visualizers. Default blend: screen.
4. **Foreground** — **has full placement controls.** Connect subjects with removed backgrounds. Supports fit/position/custom.
5. **Overlay** — stretched to fill. Top layer for watermarks, grain, borders, vignettes.

| Key Parameters | Description |
|----------------|-------------|
| `output_width` / `output_height` | Canvas dimensions. |
| `bg_fill_color` | Hex color for areas with no content. Default `#000000`. |
| `output_format` | `rgb` (standard) or `rgba` (with alpha channel). |
| Per-layer: `*_image` | IMAGE input for each layer. All optional. |
| Per-layer: `*_mask` | MASK input for each layer. Optional — defaults to fully opaque. |
| Per-layer: `*_opacity` | Layer opacity 0-1. |
| Per-layer: `*_blend` | Blend mode: normal, multiply, screen, overlay, hard_light, soft_light, dodge, burn, darken, lighten, difference, exclusion, add, subtract. |
| `fg_fit_mode` | **Foreground only:** fill, fit, fit_width, fit_height, original, custom. |
| `fg_position` | **Foreground only:** 9 anchor positions (center, top_left, etc.). |
| `fg_custom_w/h/x/y` | **Foreground only:** Exact pixel placement when fit_mode=custom. |

**Outputs:** `composited` (IMAGE), `alpha_mask` (MASK)

**Typical workflow:**
```
[Audio Visualizer] → bg_image
[Procedural FX]    → fx_image (blend: screen, opacity: 0.7)
[BG Removed Subject] → fg_image + fg_mask (fit: fit, position: center)
[Film Grain]       → overlay_image (blend: overlay, opacity: 0.3)
```

---

### LTX 2.3 Bridge

These nodes don't replace native LTX ComfyUI nodes — they bridge CutFlow's editing capabilities with LTX 2.3's generation pipeline. Use them to prepare inputs for LTX, and post-process LTX outputs.

#### 🎯 S42CF Guide Frame Prep

Extract and resize frames for LTX 2.3 image-to-video conditioning (LTXVAddGuide + LTXVPreprocess).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | IMAGE | — | Source video or image batch |
| `mode` | select | first_last | `first_last` = standard FLF2V. `first_only` = simple I2V. `keyframes` = custom indices. `evenly_spaced` = N guides. |
| `keyframe_indices` | string | "0, -1" | Frame indices for keyframes mode. Negative = from end. |
| `target_width/height` | int | 768/512 | Output dimensions matching your LTX empty latent. |

**Outputs:** `first_frame`, `last_frame`, `guide_indices` (string), `info`

---

#### 🔢 S42CF LTX Frame Calculator

Calculate valid LTX 2.3 frame counts. LTX requires frames = 8n+1 (9, 17, 25, ... 121, ... 257, ... 513).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desired_duration` | float | 5.0 | Target duration in seconds |
| `fps` | int | 24 | Target frame rate |
| `round_mode` | select | nearest | How to snap to valid 8n+1 count |
| `max_frames` | int | 257 | VRAM cap. 121≈5s, 257≈10.7s, 513≈21.4s at 24fps |

**Outputs:** `frame_count` (INT — connect to EmptyLTXVLatentVideo), `fps`, `duration_seconds`, `fps_float`, `info`

---

#### 👤 S42CF Subject Isolate

Separate a video into subject plate and background plate for replacement workflows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Source video |
| `subject_mask` | MASK | — | From Background Remover |
| `output_mode` | select | subject_on_black | `subject_on_black` (for LTX guide), `subject_on_white`, `subject_rgba`, `background_only`, `background_inpaint_simple` |
| `expand_mask` | int | 3 | Expand mask edges (pixels) |
| `feather` | float | 2.0 | Feather mask edges |

**Outputs:** `subject_plate`, `background_plate`, `refined_mask`, `info`

**Subject Replacement workflow:** BG Remover → Subject Isolate → use background_plate as LTX guide frames → LTX generates new background → Layer Composer composites original subject onto new background.

---

#### 🏞 S42CF Background Plate

Generate clean background plates for compositing or LTX regeneration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | solid_color | `solid_color`, `gradient_h`, `gradient_v`, `from_video_median`, `from_image` |
| `width/height` | int | 768/512 | Plate dimensions |
| `frames` | int | 1 | Frame count (match your video for animated backgrounds) |
| `color1_hex/color2_hex` | string | #1a1a2e/#16213e | Colors for solid/gradient modes |
| `source_video` *(optional)* | IMAGE | — | For `from_video_median` — computes per-pixel temporal median |
| `source_image` *(optional)* | IMAGE | — | For `from_image` — scales to canvas |

---

#### 📐 S42CF Video Segment Prep

Split long video into LTX-compatible overlapping segments for segment-by-segment generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Long source video |
| `segment_frames` | int | 121 | Frames per segment (must be 8n+1) |
| `overlap_frames` | int | 9 | Overlap between segments for smooth stitching |
| `segment_index` | int | 0 | Which segment to output |

**Outputs:** `segment`, `first_frame`, `last_frame` (both ready for LTXVAddGuide), `segment_count`, `frame_count`, `info`

---

#### 🔊 S42CF Audio Cond Prep

Prepare audio for LTX 2.3 audio conditioning — trim to match video, normalize, fade.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | — | Audio input |
| `target_frames` | int | 121 | Must match your LTX empty latent frame count |
| `fps` | int | 24 | For duration calculation |
| `normalize` | select | yes | Normalize to -3dBFS peak (recommended) |
| `fade_in_ms/fade_out_ms` | int | 50/100 | Prevent click artifacts |

---

#### ✨ S42CF LTX Post-Process

Post-process LTX 2.3 output — crop guide artifacts, auto-brightness, temporal denoise.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | LTX VAE decode output (or RTX upscale output) |
| `crop_last_frames` | int | 2 | Remove N frames from end (LTXVAddGuide artifacts). Matches LTXVCropGuides behavior. |
| `crop_first_frames` | int | 0 | Remove N frames from start |
| `auto_brightness` | select | none | `none`, `subtle` (slight exposure fix), `moderate` (auto-levels) |
| `denoise_strength` | float | 0.0 | Temporal denoise. 0 = off. 0.3 = subtle noise reduction. |

---

#### 📝 S42CF Scene Describer

Analyze video frames to help write better LTX prompts. Reports color palette, lighting, motion level, composition.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Video or image to analyze |
| `detail_level` | select | basic | `basic` = color + brightness + motion. `detailed` = adds composition analysis and prompt suggestions. |

**Outputs:** `scene_description` (human-readable analysis), `prompt_suggestions` (suggested LTX prompt elements)

---

## Example Workflows

All 52 nodes are showcased across 12 workflows in the `workflows/` folder. Drag any JSON file onto ComfyUI to load it.

### 01 — Basic Edit & Color Grade
**File:** `workflows/01_basic_edit_and_grade.json`
**Nodes used:** QuickPreview → SmartTrim → FilterPack → ColorAdjust → Vignette → StyleTransfer → ClipInfo

The bread-and-butter editing workflow. Load a video, trim to your desired segment, sharpen, adjust brightness/contrast/saturation, add a vignette, apply a cinematic color grade, and inspect the final clip metadata. The QuickPreview at the start runs everything at 480p for fast iteration.

---

### 02 — Speed Ramp & Reverse
**File:** `workflows/02_speed_ramp_and_reverse.json`
**Nodes used:** SpeedRamp → ClipReverse → FrameBlend → FilmGrain → ThumbnailStrip

Create dramatic speed effects. The SpeedRamp uses the "bullet_time" preset for a slow-mo center section. A parallel branch shows ClipReverse at half speed. FrameBlend smooths any jitter from speed changes, FilmGrain adds texture, and ThumbnailStrip generates a visual contact sheet of the final result.

---

### 03 — Cut & Assemble
**File:** `workflows/03_cut_and_assemble.json`
**Nodes used:** MultiSplit → FrameExtract → FrameHold → FrameInsert → LoopBounce → SideBySide

Multi-segment editing. MultiSplit divides a clip at beat points. FrameExtract pulls keyframes. FrameHold creates a freeze-frame intro. FrameInsert patches the freeze into the segment. LoopBounce creates a ping-pong loop. SideBySide compares the original vs. the edited result.

---

### 04 — Text & Titles
**File:** `workflows/04_text_and_titles.json`
**Nodes used:** ImageToClip → TextOverlay → SubtitleBurn → LowerThird → Watermark

Full title and text pipeline. ImageToClip turns a still title card into a 3-second clip. TextOverlay renders a large title with stroke and shadow. SubtitleBurn adds SRT subtitles with cinematic styling. LowerThird animates a name/title card with slide-in/out. Watermark stamps the final output.

---

### 05 — Composition & Layout
**File:** `workflows/05_composition_and_layout.json`
**Nodes used:** KenBurns → PictureInPicture → SplitScreen → CropPad → AspectConvert → BatchResize

Multi-clip composition. KenBurns adds a slow zoom/pan to the main clip. PictureInPicture overlays a webcam-style feed with rounded corners. SplitScreen creates a left/right comparison. CropPad adjusts to 16:9. AspectConvert reformats for vertical (9:16). BatchResize outputs at 1920×1080.

---

### 06 — VFX & Glitch Art
**File:** `workflows/06_vfx_and_glitch.json`
**Nodes used:** ChromaticAberration → LensDistortion → TemporalFX → GlitchFX → ChannelOps → HistogramOverlay

Creative effects chain. ChromaticAberration adds radial RGB fringing. LensDistortion applies barrel distortion. TemporalFX creates ghosting echoes. GlitchFX adds RGB split glitch. ChannelOps inverts colors. HistogramOverlay monitors the RGB levels through the chain.

---

### 07 — Transitions, Masks & Stabilize
**File:** `workflows/07_transitions_and_masks.json`
**Nodes used:** Stabilize → MaskWipe → ShapeMask

Two-clip transition with stabilization. Stabilize de-shakes clip A before the transition. MaskWipe performs a radial-out reveal from clip A to clip B with soft edges. ShapeMask generates an animated ellipse mask with size keyframes for use in other compositing scenarios.

---

### 08 — Audio-Synced Edit & GIF Export
**File:** `workflows/08_audio_sync_and_export.json`
**Nodes used:** BeatSnap → AudioTrim → AudioVideoSync → ClipToGIF

Audio-driven editing. BeatSnap analyzes the audio track and outputs beat frame indices (connect to MultiSplit for beat-synced cuts). AudioTrim cuts the audio with fade in/out. AudioVideoSync matches audio duration to video. ClipToGIF exports a separate animated GIF of the clip.

---

### 09 — Background Removal + Layer Compositing *(Expanded)*
**File:** `workflows/09_bg_removal_and_compositing.json`
**Nodes used:** BackgroundRemover → LayerComposer

The foundational compositing workflow. Load a subject video, remove its background with BiRefNet AI, then composite the isolated subject onto a new background video using the 5-layer Layer Composer. The BG remover outputs the mask directly to the composer's foreground mask input. Add FX layers, overlays, or fill layers as needed.

---

### 10 — LTX 2.3 FLF2V + CutFlow Post-Processing *(Expanded)*
**File:** `workflows/10_ltx23_flf2v_post_processing.json`
**Nodes used:** LTXFrameCalc → GuideFramePrep → *(LTX pipeline)* → LTXPostProcess → StyleTransfer → Vignette

Integrates CutFlow with your LTX 2.3 First-Last-Frame-to-Video workflow. GuideFramePrep extracts and resizes guide frames for LTXVAddGuide. FrameCalc computes valid 8n+1 frame counts. After LTX generates, LTXPostProcess crops guide artifacts (last 2 frames), then StyleTransfer and Vignette add cinematic post-processing. Connect the standard LTX 2.3 pipeline (DualCLIPLoader → KSampler → VAEDecodeTiled) in between.

---

### 11 — Subject Replacement with LTX 2.3 *(Expanded)*
**File:** `workflows/11_subject_replacement_ltx23.json`
**Nodes used:** BackgroundRemover → SubjectIsolate → GuideFramePrep → SceneDescriber → LayerComposer

The "replace everything except the subject" workflow. Background Remover isolates the subject. SubjectIsolate separates the subject plate and background plate (with optional simple inpainting). GuideFramePrep extracts frames from the background plate for LTX guide conditioning. SceneDescriber analyzes the scene to help write the regeneration prompt. LTX 2.3 generates a new background. Layer Composer composites the original subject onto the AI-generated background.

---

### 12 — Long-form Segmented LTX 2.3 Generation *(Expanded)*
**File:** `workflows/12_longform_segmented_ltx23.json`
**Nodes used:** VideoSegmentPrep → AudioCondPrep → LTXFrameCalc → LTXPostProcess → BackgroundPlate

For videos longer than LTX 2.3's single-pass limit (~10-20 seconds). VideoSegmentPrep splits the source into overlapping 121-frame segments with 9-frame overlap. Each segment's first and last frames are extracted for guide conditioning. AudioCondPrep normalizes and trims the audio to match. Process each segment through LTX independently, then stitch results back together. BackgroundPlate provides fallback gradient backgrounds for segments without source video.

---

### How to Use

1. Open ComfyUI
2. Drag any `.json` file from `workflows/` onto the ComfyUI canvas
3. Connect your own video source to the Load Video nodes
4. Adjust parameters to taste
5. Queue prompt to render

**Note:** The Load Video and Video Combine nodes use GGF/VHS types. If you use different loader/saver nodes, simply swap them out — all CutFlow nodes use standard IMAGE batches.

---

## Project Structure

```
S42-CutFlow/
├── __init__.py           # Node registration (42 core + 10 expanded = 52 nodes)
├── cf_utils.py           # Shared utilities, easing, keyframes
├── cf_clip_ops.py        # Trim, split, loop, hold, extract, insert, reverse
├── cf_speed_ramp.py      # Speed ramping with curves
├── cf_filters.py         # Filter pack, vignette, grain, CA, distortion, style, color adjust
├── cf_temporal.py        # Temporal FX, glitch FX, frame blend
├── cf_text.py            # Text overlay, subtitles, watermark, lower third
├── cf_composition.py     # PiP, split screen, Ken Burns, crop, masks
├── cf_stabilize.py       # Video stabilization
├── cf_audio_sync.py      # Audio trim, beat snap, AV sync
├── cf_preview.py         # Quick preview, thumbnails, info, compare, histogram
├── cf_utilities.py       # Aspect, resize, channels, img→clip, clip→GIF
├── cf_bg_remover.py      # [EXPANDED] Background removal (AI + traditional)
├── cf_layer_composer.py  # [EXPANDED] 5-layer video compositor
├── cf_ltx_bridge.py      # [EXPANDED] LTX 2.3 bridge (8 nodes)
├── requirements.txt      # opencv-python
├── LICENSE               # MIT
├── README.md             # This file
└── workflows/            # Example workflow JSON files (12 total)
    ├── 01_basic_edit_and_grade.json
    ├── 02_speed_ramp_and_reverse.json
    ├── 03_cut_and_assemble.json
    ├── 04_text_and_titles.json
    ├── 05_composition_and_layout.json
    ├── 06_vfx_and_glitch.json
    ├── 07_transitions_and_masks.json
    ├── 08_audio_sync_and_export.json
    ├── 09_bg_removal_and_compositing.json
    ├── 10_ltx23_flf2v_post_processing.json
    ├── 11_subject_replacement_ltx23.json
    └── 12_longform_segmented_ltx23.json
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Willie Gray Jr](https://github.com/GeekyGhost) as part of the S42 Production Suite ecosystem.*

*52 nodes. 12 workflows. LTX 2.3 ready. Mostly harmless.*