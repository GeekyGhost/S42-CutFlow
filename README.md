# S42 CutFlow — Video Editing Suite for ComfyUI

Workflows are being tweaked, audio editing nodes are coming also. Bringing all my tools to this one suite. 

**42 core + 11 expanded = 53 nodes for professional video editing within ComfyUI's node graph.** CapCut-class editing power with zero GPU model dependencies. Trim, split, speed ramp, filter, color grade, overlay text, picture-in-picture, stabilize, beat-sync, transitions, background removal, multi-layer compositing, and LTX 2.3 bridge utilities — all within your ComfyUI workflow.

> **DON'T PANIC** — Every node works on standard ComfyUI IMAGE batches. Connect to any video loader (VHS/GGF) and any video output node. No special formats, no VRAM pressure. Your towel is optional but recommended.

---

## Table of Contents

- [Features at a Glance](#features-at-a-glance)
- [Why S42 CutFlow?](#why-s42-cutflow)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Node Reference — Core (42 Nodes)](#node-reference--core-42-nodes)
  - [Clip Operations (8)](#clip-operations-8)
  - [Filters & Color (7)](#filters--color-7)
  - [Temporal FX (3)](#temporal-fx-3)
  - [Text & Overlays (4)](#text--overlays-4)
  - [Composition & Motion (6)](#composition--motion-6)
  - [Stabilization (1)](#stabilization-1)
  - [Audio Sync (3)](#audio-sync-3)
  - [Preview & Analysis (5)](#preview--analysis-5)
  - [Utilities (5)](#utilities-5)
- [Node Reference — Expanded (11 Nodes)](#node-reference--expanded-11-nodes)
  - [Transitions (1)](#transitions-1)
  - [Background Remover (1)](#background-remover-1)
  - [Layer Composer (1)](#layer-composer-1)
  - [LTX 2.3 Bridge (8)](#ltx-23-bridge-8)
- [Example Workflows](#example-workflows)
- [Preview Workflow Tips](#preview-workflow-tips)
- [Dependencies](#dependencies)
- [Compatibility](#compatibility)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
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
| **Transitions** *(expanded)* | 1 | 20 CapCut-style transitions: dissolve, fade, push, wipe, zoom, spin, iris, slide, glitch |
| **BG Remover** *(expanded)* | 1 | BiRefNet, RMBG-2.0, rembg, chroma/luma key — batch video with temporal consistency |
| **Layer Composer** *(expanded)* | 1 | 5-layer compositor with 14 blend modes, foreground placement, alpha compositing |
| **LTX 2.3 Bridge** *(expanded)* | 8 | Guide frame prep, frame calculator, subject isolate, background plate, segment prep, audio conditioning, post-process, scene describer |
| ***Expanded Total*** | ***11*** | *Transitions + AI generation integration* |
| **Grand Total** | **53** | **Complete editing + generation pipeline** |

**Zero GPU Models for core editing.** All 42 core operations plus transitions run on CPU via torch, numpy, PIL, and OpenCV. Expanded nodes optionally use BiRefNet/rembg for AI background removal.

---

## Why S42 CutFlow?

**The problem:** ComfyUI is incredible for AI image and video generation, but once you have generated footage, you're forced to leave the node graph to perform basic editing — trimming, color grading, adding text, assembling clips. That breaks the creative flow and introduces manual export/import steps.

**The solution:** S42 CutFlow brings CapCut-class editing directly into your ComfyUI graph. Every node operates on standard `IMAGE` batches (the same tensors every ComfyUI node already uses), meaning you can generate a video with LTX, Wan, or any other model, then immediately trim it, color-grade it, add titles, composite it with other clips, and export — all without leaving ComfyUI.

**Who benefits:**

- **AI video creators** — Post-process LTX/Wan/CogVideo output without leaving ComfyUI. Trim artifacts, add cinematic color grades, stabilize jitter, composite subjects onto new backgrounds.
- **Content creators & YouTubers** — Build complete short-form video pipelines: load footage → trim → add lower thirds → burn subtitles → add watermark → export. No need for a separate NLE for simple edits.
- **Motion designers** — Ken Burns effects, picture-in-picture, split screens, mask-based transitions, and glitch effects are all parameterized and keyframeable.
- **AI researchers & experimenters** — Quick Preview mode at 480p means you iterate 4-8× faster. Thumbnail strips and histogram overlays let you visually debug your pipeline. Side-by-side comparison nodes make A/B testing trivial.
- **Automation & batch workflows** — Every node is deterministic and parameterized. Feed 100 clips through the same trim→grade→watermark pipeline via ComfyUI's queue system. Beat Snap + Multi Split enables automated music video cutting.

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

After restart, look for nodes under the **S42 CutFlow** category in the node menu. You should see 53 nodes across core and expanded subcategories. The console will print:

```
[S42 CutFlow] Loaded 42 core + 11 expanded = 53 total nodes
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

### Two-Clip Transition

```
[Load Clip A] → [S42CF Transition (dissolve, 24 frames)] ← [Load Clip B]
                          ↓
                  [VHS Video Combine]
```

---

## Node Reference — Core (42 Nodes)

Every parameter on every node has a tooltip — hover over any input in ComfyUI to see its description. All nodes output an `info` STRING with a human-readable summary of what they did.

---

### Clip Operations (8)

These are the fundamental building blocks of any video edit — cutting, rearranging, and retiming footage. If you've used a timeline-based editor, these are your razor blade, slip tool, and speed controls translated into node form.

---

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

**Why this matters:** Every AI video generation model produces clips with unusable start/end frames — LTX artifacts, Wan warmup frames, CogVideo noise. Smart Trim lets you surgically remove those frames before any downstream processing. The `seconds` mode is especially useful when you know you want "the middle 3 seconds" but don't want to calculate frame indices by hand. Connect the `frame_count` output to downstream nodes that need to know clip duration (Audio Trim, Frame Hold, etc.) to keep everything in sync.

**Use cases beyond video editing:** Trim image sequences from batch renders, extract specific frames from animation sequences, prepare training data by isolating clean segments from recorded footage.

---

#### ✂ S42CF Multi Split

Split a clip at multiple points into segments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Video clip to split |
| `split_points` | string | "24, 48" | Comma-separated frame indices |
| `segment_index` | int | 0 | Which segment to output (0-based) |

**Why this matters:** Multi Split is the engine behind beat-synced editing. Connect the output of Beat Snap directly to `split_points` and your footage is automatically cut on every beat. Change `segment_index` to pick which segment to process — pair multiple Multi Split nodes with the same split points but different indices to process each segment through different effects chains. This is how you build a montage without a timeline.

**Use cases beyond video editing:** Split long screen recordings at scene changes, divide lecture footage into topic segments, extract individual shots from surveillance footage for analysis.

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

**Why this matters:** Short AI-generated clips (2-5 seconds) often need to fill longer timelines. Loop Bounce extends them seamlessly. The `ping_pong` mode is particularly valuable because it creates smooth back-and-forth motion that disguises the loop point — a technique used constantly in social media content, GIF creation, and ambient background videos. The crossfade option eliminates the visible "pop" at loop boundaries.

**Use cases beyond video editing:** Create seamless animated backgrounds for streaming overlays, build looping product showcase videos for e-commerce, generate infinite ambient loops from short nature clips for relaxation apps.

---

#### ⏸ S42CF Frame Hold

Freeze a specific frame for N frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Source clip |
| `frame_source` | select | first | `first`, `last`, `middle`, or `custom` |
| `frame_index` | int | 0 | Frame index (only for `custom` source) |
| `hold_frames` | int | 24 | Duration of freeze (24 = 1 second at 24fps) |

**Why this matters:** Freeze frames are a staple of dramatic editing — the hero moment, the "record scratch" beat, the title card hold. Frame Hold creates these by duplicating a single frame for a specified duration. Use it to create still title cards from a single generated image, to add dramatic pauses in action sequences, or to hold on a key visual while subtitles play. At 24fps, setting `hold_frames` to 72 gives you exactly 3 seconds of freeze — perfect for lower-third name cards.

**Use cases beyond video editing:** Create still-frame thumbnails for video galleries, generate static preview images from video content, produce hold-frame sequences for animation timing reference.

---

#### 📋 S42CF Frame Extract

Extract specific frames by index, range, or interval.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | interval | `indices`, `interval`, or `range` |
| `index_list` | string | "0, 12, 24" | Frame numbers for `indices` mode |
| `interval` | int | 4 | Every Nth frame for `interval` mode |
| `range_start` / `range_end` | int | 0 / -1 | Range bounds (-1 = end) |

**Why this matters:** Frame Extract serves three distinct purposes depending on mode. In `interval` mode, it's a frame rate reducer — extract every 4th frame to go from 24fps to 6fps for a choppy, stylized look or to reduce processing load. In `indices` mode, it's a keyframe picker — pull specific frames for use as LTX guide frames, thumbnail candidates, or reference stills. In `range` mode, it's a lightweight trim that doesn't need FPS information.

**Use cases beyond video editing:** Downsample high-FPS footage for processing efficiency, extract keyframes for storyboard creation, pull specific frames for training data curation, create time-lapse effects from real-time footage.

---

#### 📥 S42CF Frame Insert

Insert frames at a specific position within a clip.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Main clip |
| `insert_frames` | IMAGE | — | Frames to insert (auto-resolution matched) |
| `insert_at` | int | 0 | Insertion index (0 = prepend) |

**Why this matters:** Frame Insert is the complement to Frame Extract and Frame Hold. Use it to splice a freeze-frame title card into the beginning of a clip, insert a transition frame sequence between segments, or patch corrected frames back into a clip after processing them separately. The automatic resolution matching means you don't need to worry about mismatched dimensions — insert a 512×512 title card into a 1920×1080 video and it will be scaled to fit.

**Use cases beyond video editing:** Inject watermark frames at specific intervals, insert chapter markers into long recordings, splice corrected/inpainted frames back into sequences.

---

#### ⏪ S42CF Clip Reverse

Reverse a clip with optional speed modification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speed` | float | 1.0 | Speed after reversing (0.5 = slow-mo reverse, 2.0 = fast) |
| `interpolation` | select | blend | `blend` (smooth) or `duplicate` (fast) |

**Why this matters:** Reverse playback is more than a novelty effect. At 0.5× speed with blend interpolation, it creates dramatic slow-motion reverse sequences — think of an explosion reassembling itself, or water flowing upward. Combined with Loop Bounce in ping-pong mode, you get seamless forward-reverse loops. The `blend` interpolation generates smooth intermediate frames when the speed isn't 1.0, avoiding the stuttery look of simple frame duplication.

**Use cases beyond video editing:** Create reverse-motion effects for product reveals, generate boomerang-style social media clips, produce artistic time-reversal sequences.

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

Example: `0:1.0, 12:0.25:ease_in, 36:0.25, 48:1.0:ease_out` = normal → slow-mo → normal

**Preset descriptions:**

| Preset | Behavior | Best For |
|--------|----------|----------|
| `bullet_time` | Normal → very slow (0.15×) → normal | Action hero moments, dramatic reveals |
| `montage` | Fast → slow → fast | Dynamic montage edits, travel videos |
| `hero_moment` | Slow build → near-freeze → resume | Sports highlights, climactic beats |
| `whip_pan` | Slow → very fast (3×) → slow | Quick transitions, energetic cuts |
| `ramp_up` | 0.3× → 2.5× | Building energy, acceleration reveals |
| `ramp_down` | 2.5× → 0.3× | Dramatic landing, arrival emphasis |

**Available easings:** `linear`, `ease_in`, `ease_out`, `ease_in_out`, `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `elastic_in`, `elastic_out`, `bounce`

**Why this matters:** Speed ramping is the single most popular effect in modern short-form video editing. It's what gives TikToks and Reels their signature kinetic energy. The presets replicate CapCut's most-used speed curves — `bullet_time` alone covers 80% of dramatic slow-motion use cases. The custom keyframe string gives you frame-level control with proper easing functions, something that usually requires a full NLE's curve editor.

**Use cases beyond video editing:** Create dynamic product demos with dramatic pauses on key features, build sports highlight reels with automatic slow-motion on impact moments, generate stylized AI video output with variable-speed playback.

---

### Filters & Color (7)

Color grading and image filtering — the tools that transform raw footage into something with visual character. These nodes cover everything from basic sharpening to full cinematic color science.

---

#### 🔧 S42CF Filter Pack

Multi-mode filter with 7 operations covering the most common image processing needs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | sharpen | `sharpen`, `unsharp_mask`, `gaussian_blur`, `box_blur`, `motion_blur`, `median_denoise`, `bilateral_denoise` |
| `intensity` | float | 1.0 | Strength (meaning varies by mode) |
| `radius` | int | 3 | Kernel radius (must be odd) |
| `angle` | float | 0.0 | Direction for motion_blur (degrees) |

**Why this matters:** AI-generated video often has specific artifacts that traditional filters address directly. `sharpen` and `unsharp_mask` counteract the slight softness common in VAE-decoded output. `bilateral_denoise` smooths noise while preserving edges — ideal for cleaning up LTX or Wan output without losing detail. `motion_blur` at specific angles can simulate camera movement in generated footage that looks too static. Having all seven operations in one node with a mode selector keeps your graph clean.

**Use cases beyond video editing:** Pre-process input images for AI generation (sharpen blurry reference photos), post-process AI output to match real footage characteristics, apply depth-of-field simulation via selective blur, denoise surveillance or low-light footage.

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

**Why this matters:** Vignetting is one of the simplest ways to add production value to any footage. It naturally draws the viewer's eye toward the center of frame and mimics the optical falloff of real cinema lenses. The offset controls let you shift the "bright spot" to track an off-center subject. Non-black vignette colors (warm orange, cool blue) add mood. The rectangle shape option creates a framed look useful for title cards and presentations.

**Use cases beyond video editing:** Add lens-like optical character to AI-generated images, create themed frames for social media content, simulate vintage camera optics, draw attention to specific regions in presentations or tutorials.

---

#### 🎞 S42CF Film Grain

Procedural film grain with size, color mode, and animation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intensity` | float | 0.1 | Grain amount (0.05=subtle, 0.3=heavy) |
| `grain_size` | float | 1.5 | Grain scale (1=fine pixel, 3=coarse) |
| `color_mode` | select | mono | `mono` (classic film) or `color` (chromatic) |
| `animated` | select | yes | `yes` = unique per frame, `no` = static |
| `seed` | int | 42 | Random seed for reproducibility |

**Why this matters:** AI-generated video has a telltale "too clean" quality — perfectly smooth gradients, zero noise. Film grain breaks up that artificial cleanliness and makes the output feel organic. The `mono` mode replicates classic photographic grain; `color` mode adds per-channel noise that looks more like digital sensor noise. The `animated` flag ensures each frame gets unique grain (essential for video — static grain looks wrong in motion). The seed makes results reproducible for consistent batch processing.

**Use cases beyond video editing:** Add film-like texture to AI art, make digital renders feel more photographic, mask compression artifacts in low-bitrate output, create vintage or aged film effects for storytelling.

---

#### 🌈 S42CF Chromatic Aberration

RGB channel offset for chromatic aberration effect.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intensity` | float | 3.0 | Pixel offset amount |
| `mode` | select | radial | `radial` (lens-like) or `directional` (uniform shift) |
| `angle` | float | 0.0 | Shift angle for directional mode |

**Why this matters:** Chromatic aberration — the colored fringing at the edges of an image — is an optical imperfection that paradoxically makes footage feel more "real." High-end cinema cameras exhibit subtle CA, and audiences have learned to associate it with expensive glass. In `radial` mode, the effect increases toward frame edges (matching real lens behavior). In `directional` mode, it creates a uniform RGB split useful for stylized, glitchy looks. Subtle settings (1-2px) add realism; heavy settings (5+px) create intentional VFX.

**Use cases beyond video editing:** Add optical realism to AI renders, create glitch/cyberpunk aesthetics, simulate vintage lens characteristics, add urgency or unease to thriller/horror content.

---

#### 🔮 S42CF Lens Distortion

Barrel or pincushion lens distortion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distortion` | float | 0.0 | Negative = barrel, positive = pincushion |
| `crop_to_fit` | select | yes | Auto-crop to remove black edges |

**Why this matters:** Lens distortion is another optical imperfection that adds character. Barrel distortion (negative values) mimics wide-angle/fisheye lenses, giving footage an immersive, POV quality. Pincushion distortion (positive values) creates a telescopic or compressed look. Combined with Chromatic Aberration and Vignette, you can fully simulate specific lens personalities — from a GoPro's fisheye to a vintage anamorphic portrait lens.

**Use cases beyond video editing:** Simulate camera lens effects on AI-generated content, correct lens distortion in captured footage, create fisheye/wide-angle looks for action sequences, add a documentary or surveillance aesthetic.

---

#### 🎨 S42CF Style Transfer

8 cinematic color grading presets with adjustable strength.

| Preset | Look |
|--------|------|
| `cinematic_teal_orange` | Hollywood blockbuster (warm highlights, cool shadows) |
| `vintage` | Faded warm tones, lifted blacks |
| `noir` | High-contrast desaturated |
| `polaroid` | Warm shadows, cool highlights |
| `cyberpunk` | Neon-shifted, high saturation |
| `pastel` | Soft, desaturated pastels |
| `bleach_bypass` | Silver-retention film look (high contrast, low saturation) |
| `cross_process` | Shifted color channels, retro film lab accident |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | select | cinematic_teal_orange | Color grade preset |
| `strength` | float | 1.0 | Blend (0 = original, 1 = fully styled) |

**Why this matters:** Color grading is what separates "footage" from "cinema." These presets encode the color science of specific visual styles — `cinematic_teal_orange` is the complementary color scheme used in most Hollywood blockbusters, `bleach_bypass` replicates a physical film development technique that gives Saving Private Ryan its iconic look. The `strength` slider lets you dial in exactly how much grade to apply, so you can go from subtle enhancement (0.3) to full stylization (1.0).

**Use cases beyond video editing:** Apply consistent color grades across AI image batches, create mood-specific visual styles for brand content, replicate specific cinematic looks without manual color wheel adjustment, batch-process generated images with a unified aesthetic.

---

#### 🎛 S42CF Color Adjust

Fundamental brightness, contrast, saturation, hue, gamma, and temperature adjustment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `brightness` | float | 0.0 | Brightness offset (-1 to +1). Typical: -0.2 to +0.2 |
| `contrast` | float | 1.0 | Contrast multiplier (0=flat, 1=normal, 1.5=punchy) |
| `saturation` | float | 1.0 | Color saturation (0=gray, 1=normal, 1.5=vivid) |
| `hue_shift` | float | 0.0 | Hue rotation in degrees (-180 to +180) |
| `gamma` | float | 1.0 | Gamma correction (<1=lift shadows, >1=darken mids) |
| `temperature` | float | 0.0 | Warm/cool shift (-1=cool/blue, +1=warm/orange) |

**Why this matters:** Where Style Transfer applies an opinionated look, Color Adjust gives you direct, independent control over every fundamental image property. This is the node you use for technical correction — fixing underexposed AI output (brightness + gamma), boosting washed-out colors (saturation + contrast), or shifting the overall mood (temperature). Gamma is particularly powerful for AI video: a value of 0.85 lifts shadow detail that VAE decoding often crushes, recovering hidden information.

**Use cases beyond video editing:** Correct exposure and white balance in generated images, prepare reference images for img2img pipelines, normalize brightness across frames with inconsistent AI output, shift mood for storytelling through temperature adjustment.

---

### Temporal FX (3)

Time-domain effects that operate across multiple frames to create motion-based looks. These nodes look at the relationship between frames rather than treating each frame independently.

---

#### ⏳ S42CF Temporal FX

Time-domain effects across frame windows.

| Mode | Description |
|------|-------------|
| `echo` | Semi-transparent past frames layered on current — ghostly trails |
| `motion_trail` | Accumulating trail from movement — speed streaks |
| `time_displacement` | Vertical rows sample different time offsets — the "slit-scan" effect |
| `frame_average` | Average N adjacent frames — dreamy, ghostly, long-exposure |
| `strobe` | Alternating frame hold — strobe/flash effect |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | echo | Effect type |
| `window` | int | 5 | Number of frames in effect window |
| `intensity` | float | 0.5 | Effect strength |
| `decay` | float | 0.8 | Echo/trail fade rate per frame |

**Why this matters:** Temporal effects create visuals that are impossible to achieve with single-frame processing. The `echo` mode creates ghosting that suggests speed and energy — used in every music video since the 1980s. `time_displacement` produces the "slit-scan" effect from 2001: A Space Odyssey. `frame_average` creates a dreamy long-exposure look that's particularly stunning with AI-generated footage, smoothing out minor frame-to-frame inconsistencies while creating an ethereal quality.

**Use cases beyond video editing:** Visualize motion patterns in surveillance footage, create artistic long-exposure effects from standard video, generate abstract visual art from mundane footage, smooth out AI generation inconsistencies through frame averaging.

---

#### 💥 S42CF Glitch FX

Digital glitch effects with 5 modes.

| Mode | Description |
|------|-------------|
| `datamosh` | Frame bleed between consecutive frames — the hallmark of compressed video errors |
| `scan_lines` | Horizontal line shift artifacts — CRT television malfunction |
| `pixel_sort` | Sort pixels by brightness in bands — the signature glitch art technique |
| `rgb_split` | Random RGB channel displacement — digital interference |
| `block_corrupt` | Random block displacement — MPEG compression artifacts |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | rgb_split | Glitch type |
| `intensity` | float | 0.5 | Effect amount |
| `seed` | int | 42 | Reproducible randomness |
| `block_size` | int | 16 | Block size for block_corrupt |

**Why this matters:** Glitch art has become a major visual style in music videos, cyberpunk aesthetics, and horror content. Each mode replicates a specific type of digital failure: `datamosh` looks like video compression gone wrong, `pixel_sort` creates the distinctive sorted-pixel look popularized by artists like Sabato Visconti, and `block_corrupt` mimics MPEG macroblocking. The seed parameter ensures reproducibility — essential for professional work where you need the same glitch on every render.

**Use cases beyond video editing:** Generate glitch art from AI images, create cyberpunk/synthwave aesthetics, simulate corrupted data for storytelling, build visual effects for music videos and live performances.

---

#### 🔄 S42CF Frame Blend

Blend adjacent frames for motion blur or AI artifact smoothing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window` | int | 3 | Frames to blend (2-3=subtle, 5+=heavy) |
| `weights` | select | center_weighted | `uniform`, `center_weighted`, `exponential_decay` |
| `strength` | float | 1.0 | Mix (0=original, 1=fully blended) |

**Why this matters:** Frame Blend is the unsung hero of AI video post-processing. AI-generated video often has frame-to-frame jitter — small inconsistencies in position, color, or detail that create a "vibrating" look. A 3-frame center-weighted blend at 0.3 strength eliminates most of this jitter without significant motion blur. It's also essential after Speed Ramp: when you change speed, frame timing shifts can create visible stuttering, and Frame Blend smooths the result. For creative use, heavy blending (window=7, uniform) creates a dreamy motion-blur look.

**Use cases beyond video editing:** Smooth AI-generated video jitter, create motion blur for stop-motion animation, reduce flickering in timelapse footage, simulate long-exposure video effects.

---

### Text & Overlays (4)

Professional text rendering for titles, subtitles, lower thirds, and watermarks. All text is rendered via PIL with proper anti-aliasing, stroke, and shadow support.

---

#### 📝 S42CF Text Overlay

Rich text rendering with stroke, shadow, background, and positioning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | "Hello World" | Multi-line text content |
| `font_size` | int | 48 | Font size (8-400px) |
| `bold` | select | no | Bold variant |
| `text_color` | string | #FFFFFF | Text color (hex, with color picker) |
| `position_x` / `position_y` | float | 0.5 / 0.5 | Normalized position (0-1) |
| `alignment` | select | center | `center`, `left`, `right` |
| `stroke_width` | int | 0 | Outline thickness (0=none) |
| `stroke_color` | string | #000000 | Outline color |
| `shadow_offset` | int | 0 | Drop shadow distance (0=none) |
| `opacity` | float | 1.0 | Text opacity |
| `bg_padding` | int | 0 | Background box padding (0=no box) |
| `bg_opacity` | float | 0.5 | Background box opacity |

**Why this matters:** Text Overlay replaces the need to leave ComfyUI for simple title card creation. It supports all the essentials: multi-line text, stroke outlines for readability over busy backgrounds, drop shadows for depth, and semi-transparent background boxes (the "YouTube subtitle" look). The hex color picker integrates directly into the ComfyUI widget, so you can pick colors visually rather than typing hex codes. Position values are normalized (0-1) so they work at any resolution.

**Use cases beyond video editing:** Add branded titles to generated content, create meme-style text overlays, add annotation text to tutorial recordings, generate titled thumbnails from video frames.

---

#### 💬 S42CF Subtitle Burn

Burn SRT subtitles onto frames with style presets.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `srt_text` | string | — | Standard SRT format text |
| `fps` | float | 24.0 | For timestamp→frame conversion |
| `style` | select | default | `default`, `cinematic`, `youtube`, `outline_only` |
| `position_y` | float | 0.9 | Vertical position (0.9=near bottom) |

**Style descriptions:**

| Style | Look |
|-------|------|
| `default` | White text, black outline — universally readable |
| `cinematic` | White text, dark drop shadow — film/TV look |
| `youtube` | White on semi-transparent black box — YouTube standard |
| `outline_only` | Thick black outline, no fill — bold, graphic |

**Why this matters:** Subtitle Burn takes standard SRT format (the universal subtitle file format) and renders each subtitle at the correct timecode directly onto the video frames. This is "hard" or "burned-in" subtitling, which means the text becomes part of the image — essential for social media platforms that strip subtitle tracks. The four style presets cover the most common subtitle aesthetics without requiring manual configuration of font, stroke, shadow, and background parameters.

**Use cases beyond video editing:** Add accessibility subtitles to generated content, create bilingual subtitle tracks, burn timecodes for review copies, add captions to tutorial/educational video.

---

#### 🔖 S42CF Watermark

Image or text watermark with tiling support.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | text | `text` or `image` |
| `text` | string | "© 2026" | Watermark text (text mode) |
| `position` | select | bottom_right | `bottom_right/left`, `top_right/left`, `center`, `tile` |
| `opacity` | float | 0.3 | Watermark opacity (lower = more subtle) |
| `scale` | float | 1.0 | Size multiplier |
| `font_size` | int | 24 | Text mode font size |
| `margin` | int | 20 | Edge margin in pixels |
| `watermark_image` | IMAGE | *(optional)* | Image watermark (image mode) |

**Why this matters:** Watermarking is essential for protecting work-in-progress footage, branding final output, and adding copyright notices. The `tile` mode repeats the watermark across the entire frame — a standard anti-theft technique for preview content. The `image` mode lets you overlay a logo PNG with transparency. At 0.2-0.3 opacity, watermarks are visible but don't distract from the content.

**Use cases beyond video editing:** Brand all output from a generative pipeline, protect preview footage from unauthorized use, add copyright to batch-processed images, create branded templates.

---

#### 📺 S42CF Lower Third

Animated lower-third name/title cards with slide-in/out animation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "John Doe" | Primary name text |
| `title` | string | "Lead Designer" | Secondary title text |
| `style` | select | modern | `modern`, `minimal`, `broadcast`, `cinematic` |
| `accent_color` | string | #FF9900 | Accent color (hex, with picker) |
| `start_frame` | int | 0 | Appearance frame |
| `duration_frames` | int | 72 | Visible duration |
| `animate_in` | int | 12 | Slide-in animation frames |
| `animate_out` | int | 12 | Slide-out animation frames |

**Style descriptions:**

| Style | Look |
|-------|------|
| `modern` | Colored accent bar + clean text — corporate/tech |
| `minimal` | Text only, small — documentary/indie |
| `broadcast` | Full background bar — news/TV |
| `cinematic` | Large text with subtle background — film credits |

**Why this matters:** Lower thirds are the name cards that appear in the bottom third of the screen to identify speakers in interviews, documentaries, and presentations. They're one of the most tedious things to create in a traditional NLE because each one requires positioning, timing, and animation. This node generates them procedurally with proper eased slide-in/out animations, and the four style presets cover the most common broadcast aesthetics. Set `start_frame` and `duration_frames` to control exactly when each card appears.

**Use cases beyond video editing:** Generate interview-style name cards for podcast videos, create speaker introductions for conference recordings, add character identification for narrative content, build branded info cards.

---

### Composition & Motion (6)

Multi-clip composition, camera effects, and masking. These nodes combine multiple visual elements into a single frame.

---

#### 📷 S42CF Ken Burns

Apply animated pan and zoom to still images or video clips.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zoom_start` / `zoom_end` | float | 1.0 / 1.3 | Zoom level at start/end |
| `pan_x_start/end` | float | 0.0 | Horizontal pan (-1 to +1) |
| `pan_y_start/end` | float | 0.0 | Vertical pan (-1 to +1) |
| `easing` | select | ease_in_out | Animation curve |

**Why this matters:** Named after the documentary filmmaker, the Ken Burns effect transforms still images into dynamic video by slowly panning and zooming. It's the standard technique for turning photographs into video content — every documentary, slideshow, and photo montage uses it. For AI video workflows, it adds camera-like movement to generated stills, making them feel cinematic rather than static. The easing options let you control the acceleration curve of the movement.

**Use cases beyond video editing:** Animate AI-generated images for video slideshows, create dynamic social media stories from still photos, add movement to product photography, generate animated presentations from infographics.

---

#### 🖼 S42CF Picture-in-Picture

Overlay one clip on another with position, scale, and rounded corners.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | select | top_right | 9 anchor positions |
| `scale` | float | 0.3 | PiP size relative to canvas |
| `padding` | int | 20 | Edge margin |
| `corner_radius` | int | 0 | Rounded corners (0=sharp) |
| `border_width` | int | 0 | Optional border |
| `border_color` | string | #FFFFFF | Border color |

**Why this matters:** Picture-in-picture is essential for reaction videos, tutorial recordings (facecam + screen), commentary content, and multi-angle presentations. The rounded corners and border options create the polished "facecam bubble" look popularized on YouTube and Twitch. Because it operates on standard IMAGE batches, you can PiP an AI-generated clip onto real footage or vice versa.

**Use cases beyond video editing:** Create tutorial layouts with screen capture + facecam, build reaction video formats, overlay reference footage during AI generation reviews, display multiple camera angles simultaneously.

---

#### ↔ S42CF Split Screen

Side-by-side or stacked clip comparison.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | select | side_by_side | `side_by_side` or `stacked` |
| `split_position` | float | 0.5 | Split point (0-1) |
| `divider_width` | int | 2 | Line thickness between clips |
| `divider_color` | string | #FFFFFF | Divider color |

**Why this matters:** Split Screen is the standard format for before/after comparisons, A/B testing different effects chains, and side-by-side model comparisons. In AI video work, it's invaluable for showing "original vs. enhanced," "model A vs. model B," or "with vs. without post-processing." The adjustable split position lets you show more of one clip than the other. Combined with Side-by-Side (Preview node), you have complete A/B comparison capabilities.

**Use cases beyond video editing:** Compare AI model outputs, create before/after demonstrations, build multi-angle sports replays, generate comparison content for social media.

---

#### ✂ S42CF Crop/Pad

Crop, pad, letterbox, or pillarbox to target dimensions or aspect ratios.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | crop | `crop`, `pad_letterbox`, `pad_pillarbox`, `aspect_ratio` |
| `target_width/height` | int | 0 | Target dimensions (0=derive from ratio) |
| `ratio_preset` | select | 16:9 | `keep`, `16:9`, `9:16`, `4:3`, `1:1`, `21:9`, `2.35:1` |
| `pad_color` | string | #000000 | Padding color |

**Why this matters:** Aspect ratio conversion is one of the most common post-processing steps. AI models generate at fixed resolutions (768×512, 1024×576, etc.) that rarely match delivery formats. Crop/Pad handles the conversion: crop 768×512 to clean 16:9, pad to 9:16 vertical for TikTok/Reels/Shorts, letterbox to cinematic 2.35:1, or square-crop to 1:1 for Instagram. The auto-detection mode (`aspect_ratio`) picks the best fit automatically.

**Use cases beyond video editing:** Prepare AI output for specific social media platforms, convert widescreen to vertical for mobile, add letterboxing for cinematic presentation, standardize dimensions across a multi-clip project.

---

#### 🎭 S42CF Mask Wipe

Animated mask-based transitions between two clips.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_a` / `clip_b` | IMAGE | — | Source and destination clips |
| `wipe_type` | select | gradient_right | `gradient_left/right/up/down`, `radial_in/out`, `diamond`, `clock` |
| `transition_frames` | int | 24 | Wipe duration |
| `softness` | float | 0.1 | Edge softness (0=hard, 0.5=very soft) |
| `easing` | select | ease_in_out | Animation curve |
| `custom_mask` | MASK | *(optional)* | Custom wipe mask |

**Why this matters:** Mask Wipe creates traditional broadcast-style reveals using shaped masks. The `radial_out` type is the classic iris-out transition. The `clock` wipe is the rotating reveal seen in Star Wars. The `custom_mask` input lets you use any grayscale image as the wipe pattern — connect a generated mask for completely custom transitions. The softness control creates either hard-edged graphic transitions (0.0) or smooth, filmic blends (0.3+).

**Use cases beyond video editing:** Create chapter transitions in long-form content, build reveal effects for product launches, design custom animated masks for compositing, generate transition effects for presentation software.

---

#### ⬡ S42CF Shape Mask

Generate animated shape masks for compositing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | select | ellipse | `rectangle`, `ellipse`, `star`, `diamond` |
| `width/height` | int | — | Canvas dimensions |
| `size` | float | 0.5 | Shape size relative to canvas |
| `rotation` | float | 0.0 | Rotation in degrees |
| `feather` | float | 0.1 | Edge softness |
| `size_keyframes` | string | — | Animated size keyframes |
| `rotation_keyframes` | string | — | Animated rotation keyframes |

**Why this matters:** Shape Mask generates animated grayscale masks that can be connected to any node with a mask input — Mask Wipe, Layer Composer, or any other compositing node. The keyframe strings let you animate size and rotation over time, creating growing/shrinking/spinning mask reveals without needing to generate them externally. This is the mask generation primitive that powers custom transitions and compositing effects.

**Use cases beyond video editing:** Create animated masks for compositing, generate shape-based reveals for motion graphics, build animated stencils for effects, design custom transition masks.

---

### Stabilization (1)

---

#### 📐 S42CF Stabilize

Software video stabilization using OpenCV optical flow tracking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smoothing` | int | 15 | Smoothing window (larger=smoother) |
| `crop_mode` | select | auto_crop | `auto_crop`, `fill_border`, `none` |
| `max_shift` | float | 0.1 | Max correction (fraction of frame) |

**Requires:** `opencv-python`

**Why this matters:** AI-generated video often has frame-to-frame jitter — subtle position shifts that make the camera appear to "vibrate." Stabilize tracks feature points across frames using OpenCV's optical flow and applies corrective transforms to smooth the camera path. The `smoothing` window controls how aggressively it averages motion (15 frames = gentle, 30+ = locked-off tripod feel). `auto_crop` removes the black borders that stabilization creates; `fill_border` fills them by stretching edge pixels.

**Use cases beyond video editing:** Stabilize shaky handheld footage, smooth AI generation jitter, correct webcam recording wobble, stabilize drone footage, prepare footage for visual effects tracking.

---

### Audio Sync (3)

Audio-aware nodes for matching edits to music, trimming audio to match video, and syncing durations. All audio processing uses PyAV, which ships with ComfyUI.

---

#### 🔊 S42CF Audio Trim

Trim audio to match clip length with fade in/out.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | match_frames | `match_frames`, `custom_time`, `custom_frames` |
| `target_frames` | int | 48 | Frame count to match |
| `fps` | float | 24.0 | For duration calculation |
| `in_seconds` / `out_seconds` | float | — | Custom trim points |
| `fade_in_ms` / `fade_out_ms` | int | 0 / 0 | Fade durations |

**Why this matters:** Audio-video sync is the most tedious part of manual editing. Audio Trim's `match_frames` mode automatically trims (or pads with silence) the audio to exactly match your video's frame count at the specified FPS. Connect your Smart Trim's `frame_count` output to `target_frames` and the audio is always the right length. The fade options prevent the harsh "click" artifacts that occur when audio is cut mid-sample.

**Use cases beyond video editing:** Prepare audio for AI video generation (LTX audio conditioning), create perfectly-timed audio for presentations, trim podcast segments to specific durations, add professional fades to recorded audio.

---

#### 🥁 S42CF Beat Snap

Detect beats and output frame indices for synced cuts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | — | Audio input |
| `fps` | float | 24.0 | For frame calculation |
| `sensitivity` | float | 0.5 | Detection sensitivity (lower=fewer beats) |
| `min_interval` | float | 0.3 | Minimum seconds between beats |
| `output_format` | select | frame_indices | `frame_indices`, `split_points`, `keyframe_string` |

**Workflow:** Beat Snap → `beat_data` output → Multi Split `split_points` input

**Why this matters:** Beat-synced editing — cutting footage on every beat of the music — is the defining technique of music videos, montages, and energetic social media content. Manually marking beat points is time-consuming and error-prone. Beat Snap automates it: feed it an audio track and it returns the frame indices of every beat. Connect `beat_data` directly to Multi Split's `split_points`, and your footage is automatically segmented on every beat. The `sensitivity` and `min_interval` controls let you tune how many beats are detected — lower sensitivity for only the strongest downbeats, higher for every hi-hat.

**Use cases beyond video editing:** Automate music video editing, create beat-synced social media montages, synchronize visual effects to audio, generate rhythmic slideshows from photo collections, analyze music structure for DJ/production work.

---

#### 🔗 S42CF Audio-Video Sync

Match audio duration to video duration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | select | trim_pad | `trim_pad` or `time_stretch` |
| `target_frames` | int | — | Video frame count |
| `fps` | float | 24.0 | For duration calculation |

**Why this matters:** When video and audio come from different sources (common in AI workflows), their durations rarely match. Audio-Video Sync forces them to the same length. `trim_pad` is lossless — it cuts excess audio or adds silence. `time_stretch` resamples the audio to fit, which slightly changes pitch but preserves the entire audio content. Use `trim_pad` when exact audio quality matters; use `time_stretch` when you need every bit of audio to fit the video.

**Use cases beyond video editing:** Sync voiceover to generated video, match background music to clip length, prepare audio for LTX audio conditioning, align commentary tracks to screen recordings.

---

### Preview & Analysis (5)

Monitoring and review tools for efficient editing. These nodes don't modify your footage — they help you see what's happening in your pipeline.

---

#### 👁 S42CF Quick Preview

Downsample for fast iteration, passthrough for final render.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | preview | `preview` (downsample) or `passthrough` (full res) |
| `preview_height` | int | 480 | Downsampled height |
| `frame_skip` | int | 1 | Process every Nth frame (1=all) |

**Why this matters:** This is the single most impactful workflow optimization node. Place it right after your video loader and set to `preview` mode. Every downstream node now processes 480p frames instead of full resolution — that's 4-8× faster for a 1080p source, 16× faster for 4K. When you're happy with your edit, flip it to `passthrough` and queue the final render at full quality. The `frame_skip` option provides additional speed: set to 2 to process every other frame for even faster previewing.

**Workflow tip:** Place at start of chain. Work in preview mode, switch to passthrough for final render.

**Typical editing speed at 480p:** 48 frames (2 seconds @ 24fps) processes in 1-5 seconds depending on which filters are active.

---

#### 🎞 S42CF Thumbnail Strip

Generate a filmstrip contact sheet for visual review.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_frames` | int | 8 | Thumbnails to extract |
| `thumb_height` | int | 120 | Thumbnail height |
| `layout` | select | horizontal | `horizontal` or `grid` |
| `frame_numbers` | select | yes | Show frame index labels |

**Why this matters:** A thumbnail strip (or contact sheet) gives you a bird's-eye view of your entire clip in a single image. It's the visual equivalent of scrubbing through a timeline — you can instantly see where the action is, verify that your trim points are correct, check that transitions look right, and spot any anomalous frames. Place one after your Speed Ramp to verify the speed curve looks correct, or after Multi Split to confirm your segments are cut where you expect.

**Use cases beyond video editing:** Generate visual previews for video libraries, create contact sheets for animation review, produce frame reference sheets for storyboarding, build visual indices of video content.

---

#### ℹ S42CF Clip Info

Display metadata about a clip: resolution, frame count, duration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | IMAGE | — | Clip to analyze |
| `fps` | float | 24.0 | For duration calculation |

**Outputs:** `resolution`, `frame_count`, `duration`, `info` (formatted summary)

**Why this matters:** Clip Info is a diagnostic node that shows you what your data actually looks like at any point in the pipeline. Place it after any node to verify the output resolution, frame count, and calculated duration. This is essential for debugging — if your final export is the wrong length, insert Clip Info nodes throughout your chain to find where frames are being added or dropped.

---

#### 🔍 S42CF Side-by-Side Compare

A/B comparison of two clips at the same resolution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_a` / `clip_b` | IMAGE | — | Clips to compare |
| `layout` | select | side_by_side | Layout mode |
| `label_a` / `label_b` | string | "A" / "B" | Labels |

**Why this matters:** Side-by-Side Compare auto-matches resolution between two clips and composites them for direct visual comparison. Unlike Split Screen (which is a creative composition tool), this is a diagnostic tool — it adds labels and ensures the comparison is fair. Use it to compare different filter settings, different style grades, or original vs. processed footage.

---

#### 📊 S42CF Histogram Overlay

RGB histogram and waveform overlay on the video.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | histogram | `histogram` or `waveform` |
| `position` | select | top_right | Display position |
| `opacity` | float | 0.7 | Overlay opacity |
| `size` | float | 0.25 | Overlay size relative to frame |

**Why this matters:** Histogram monitoring is how professionals verify that their color grading isn't clipping highlights or crushing shadows. The RGB histogram shows the distribution of brightness values per channel — a clip pushed too far left is underexposed, too far right is overexposed. The waveform mode shows brightness distribution spatially. Place this at the end of your color grading chain to monitor your grade in real-time. It's especially useful for AI video, where VAE decoding can produce unexpected tonal ranges.

**Use cases beyond video editing:** Monitor exposure during color grading, verify AI output tonal range, detect clipping in processed footage, quality-check batch-processed images.

---

### Utilities (5)

Format conversion and utility operations that connect different parts of your workflow.

---

#### 📐 S42CF Aspect Convert

Convert between common aspect ratios.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | crop | `crop`, `pad_letterbox`, `pad_pillarbox`, `aspect_ratio` |
| `ratio_preset` | select | 16:9 | `16:9`, `9:16`, `4:3`, `1:1`, `21:9`, `2.35:1` |

**Why this matters:** Quick aspect ratio conversion with presets. While Crop/Pad offers full control, Aspect Convert is streamlined for the most common conversions: 16:9 → 9:16 for vertical social media, 16:9 → 1:1 for Instagram, 16:9 → 2.35:1 for cinematic letterboxing.

---

#### 📏 S42CF Batch Resize

Resize all frames to specific dimensions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_width` / `target_height` | int | 1920 / 1080 | Output dimensions |
| `method` | select | lanczos | `nearest`, `bilinear`, `bicubic`, `lanczos` |
| `maintain_aspect` | select | yes | Preserve aspect ratio |

**Why this matters:** Batch Resize is the resolution control node. Use it to upscale 512p AI output to 1080p for delivery, downscale 4K footage for processing efficiency, or standardize dimensions across clips before compositing. Lanczos resampling provides the highest quality scaling.

---

#### 🔀 S42CF Channel Ops

Per-channel operations: swap, invert, isolate, grayscale.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | select | grayscale | `swap_rb`, `invert`, `isolate_r/g/b`, `grayscale`, `sepia` |

**Why this matters:** Channel Ops provides direct access to individual RGB channels. `invert` creates negative images (useful as creative effects or for debugging masks). `isolate_r/g/b` extracts a single channel as grayscale — invaluable for examining what the AI model is actually generating per channel. `swap_rb` fixes RGB/BGR channel order mismatches that occasionally occur between different node packs. `sepia` adds a classic warm monochrome tone.

---

#### 🖼 S42CF Image to Clip

Convert a single image to a multi-frame clip.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | — | Source image |
| `frames` | int | 24 | Number of frames to generate |

**Why this matters:** Image to Clip bridges still images into the video pipeline. Use it to turn a title card image into a 3-second clip, create a still background for Layer Composer, or convert a generated image into a frame sequence that can be processed through video nodes (like adding Ken Burns motion). At 24fps, 72 frames = 3 seconds.

---

#### 🎬 S42CF Clip to GIF

Export a clip as an animated GIF.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fps` | float | 12.0 | GIF playback rate |
| `max_colors` | int | 256 | Palette size (16-256) |
| `dither` | select | yes | Floyd-Steinberg dithering |
| `optimize` | select | yes | File size optimization |

Saves to ComfyUI's output directory.

**Why this matters:** GIFs remain the universal format for short animated clips on the web — they play everywhere without a video player. Clip to GIF converts any IMAGE batch into a properly optimized animated GIF with dithering and palette control. The `max_colors` parameter trades quality for file size (128 colors is usually sufficient for most content). The `fps` setting defaults to 12 because GIFs look best at lower frame rates — they're a lo-fi format by nature.

**Use cases beyond video editing:** Create shareable previews of AI-generated animations, build animated stickers, generate web-optimized animated thumbnails, export short loops for social media.

---

## Node Reference — Expanded (11 Nodes)

These nodes extend CutFlow with professional transitions, AI-powered background removal, multi-layer compositing, and LTX 2.3 video generation bridge utilities.

---

### Transitions (1)

---

#### 🔀 S42CF Transition

20 CapCut-style transitions between two video clips. Supports blending transitions (dissolve, fade, glitch) that use per-frame overlap, and motion transitions (push, wipe, zoom, spin, iris, slide) that use stable reference frames to prevent jitter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_a` | IMAGE | — | First clip (plays first, transitions out) |
| `clip_b` | IMAGE | — | Second clip (transitions in, plays after) |
| `transition_type` | select | dissolve | See transition table below |
| `transition_frames` | int | 12 | Duration in frames (12 = 0.5s at 24fps) |
| `easing` | select | ease_in_out | Animation curve |
| `glitch_seed` | int | 42 | Random seed for glitch transition |

**Available transitions:**

| Type | Category | Description |
|------|----------|-------------|
| `cut` | — | Hard cut, no blending — just concatenates the two clips |
| `dissolve` | Blending | Classic crossfade between clips |
| `fade_black` | Blending | Fade to black, then from black |
| `fade_white` | Blending | Fade to white, then from white |
| `push_left/right/up/down` | Motion | Push clip A off screen, revealing clip B behind it |
| `wipe_left/right/up/down` | Motion | Edge wipe reveals clip B |
| `zoom_in` | Motion | Clip B scales up from center over clip A |
| `zoom_out` | Motion | Clip A scales down, revealing clip B |
| `glitch` | Blending | RGB channel-shift glitch transition |
| `spin_cw/ccw` | Motion | Rotation transition (clockwise or counter-clockwise) |
| `iris_in` | Motion | Circular iris wipe from center |
| `slide_up/down` | Motion | Slide reveal |

**Frame handling:** Blending transitions (dissolve, fade, glitch) composite the actual overlap frames from both clips, creating a true temporal blend. Motion transitions (push, wipe, zoom, spin, iris, slide) use stable reference frames — the last frame of clip A and the first frame of clip B — to prevent the jittery look that can occur when motion effects are applied to changing source frames. This is the same technique used in the proven S42P Transition node.

**Why this matters:** Transitions are the connective tissue of video editing. Without them, every cut is a hard jump. The Transition node provides 20 types covering every common transition style — from the ubiquitous dissolve to the dramatic iris wipe (think Star Wars) to the modern glitch transition (think music videos). The `easing` parameter controls the animation curve, so a `push_right` with `ease_in_out` accelerates smoothly rather than moving at constant speed. The stable reference frame technique for motion transitions is a critical implementation detail that prevents visual artifacts.

**Use cases beyond video editing:** Create professional multi-clip montages, build slideshow presentations with varied transitions, assemble AI-generated clips into cohesive sequences, produce music video edits with beat-synced transitions.

---

### Background Remover (1)

---

#### 🎭 S42CF Background Remover

Professional background removal for images and video batches. Supports 16 AI methods (BiRefNet, rembg, U2Net, ISNet) plus 4 traditional methods (chroma/luma key). Batch-optimized with temporal consistency for video.

| Key Parameters | Description |
|----------------|-------------|
| `method` | AI model or traditional method (see table below) |
| `use_gpu` | GPU acceleration for AI models. Recommended. |
| `processing_resolution` | AI inference resolution. 1024 = default. 2048 = max (birefnet-hr). |
| `mask_blur` | Edge softness. 0=sharp, 0.8=smooth, 2+=very soft. |
| `mask_dilation` / `mask_erosion` | Expand/shrink mask edges. Helps with hair (dilate) or fringe (erode). |
| `remove_small_objects` | Remove mask noise smaller than N pixels. 300 = good default. |
| `invert_mask` | Swap foreground/background. |
| `temporal_smooth` | **Video feature.** Blend adjacent masks to eliminate flicker. 0=off, 1-5=smoothing window. |
| `chroma_threshold/softness/spill_suppress` | Chroma key controls (chroma_* methods only). |
| `luma_mode/shadow_threshold/highlight_threshold` | Luma key controls. |
| `reference_bg` *(optional)* | Clean background plate for enhanced AI performance. |

**Method reference:**

| Method | Type | Best For |
|--------|------|----------|
| `birefnet-general` | AI | Best all-around subject isolation |
| `birefnet-portrait` | AI | People and faces |
| `birefnet-hr` | AI | High-resolution up to 2048px |
| `rmbg-2.0` | AI | Fast, general purpose |
| `u2net` / `isnet` / `silueta` | AI (rembg) | Alternative models via rembg package |
| `chroma_green/blue/red` | Traditional | Green/blue/red screen keying |
| `luma_key` | Traditional | Brightness-based separation |

**Outputs:** `result_rgb` (subject on black), `result_rgba` (with alpha), `mask` (MASK), `info`

**AI Model Requirements:**
- **BiRefNet methods:** Require `transformers` package (installed by many ComfyUI nodes). Models auto-download from HuggingFace on first use.
- **rembg methods:** Require `rembg` package. Install: `pip install rembg`.
- **Traditional methods:** No additional packages beyond opencv-python.

**Why this matters:** Background removal is the foundation of compositing — you can't place a subject onto a new background without first isolating them. The BiRefNet AI methods provide state-of-the-art edge quality, especially around hair and transparent objects (the hardest cases for traditional methods). The `temporal_smooth` parameter is critical for video: without it, AI masks can flicker frame-to-frame as the model makes slightly different decisions on each frame. A value of 2-3 eliminates this flicker while maintaining responsiveness to actual subject movement. The traditional chroma/luma key methods are included for green screen workflows where AI isn't needed.

**Use cases beyond video editing:** Isolate subjects for AI background replacement, create transparent PNGs for web/design use, remove backgrounds from product photography, prepare footage for virtual production, extract subjects for training data.

---

### Layer Composer (1)

---

#### 🎬 S42CF Layer Composer

Professional 5-layer video/image compositor with 14 blend modes. Enhanced from the S42P Layer Composer with two additional layers and full alpha-aware compositing.

**Layer stack (bottom to top):**

| Layer | Behavior | Typical Use |
|-------|----------|-------------|
| **1. Background** | Stretched to fill canvas | Video backgrounds, solid colors, gradients |
| **2. Fill** | Stretched to fill | Gradients, color washes, secondary backgrounds |
| **3. FX** | Stretched to fill | Procedural effects, particles, visualizers (default blend: screen) |
| **4. Foreground** | **Full placement controls** | Subjects with removed backgrounds (fit/position/custom) |
| **5. Overlay** | Stretched to fill | Watermarks, grain, borders, vignettes |

| Key Parameters | Description |
|----------------|-------------|
| `output_width` / `output_height` | Canvas dimensions |
| `bg_fill_color` | Hex color for empty areas. Default `#000000` |
| `output_format` | `rgb` (standard) or `rgba` (with alpha) |
| Per-layer: `*_image` | IMAGE input (all optional) |
| Per-layer: `*_mask` | MASK input (optional, defaults to fully opaque) |
| Per-layer: `*_opacity` | Layer opacity 0-1 |
| Per-layer: `*_blend` | Blend mode (14 modes) |
| `fg_fit_mode` | **Foreground only:** fill, fit, fit_width, fit_height, original, custom |
| `fg_position` | **Foreground only:** 9 anchor positions (center, top_left, etc.) |
| `fg_custom_w/h/x/y` | **Foreground only:** Exact pixel placement (custom fit_mode) |

**Available blend modes:** `normal`, `multiply`, `screen`, `overlay`, `hard_light`, `soft_light`, `dodge`, `burn`, `darken`, `lighten`, `difference`, `exclusion`, `add`, `subtract`

**Outputs:** `composited` (IMAGE), `alpha_mask` (MASK)

**Typical workflow:**
```
[Video Background]     → bg_image
[Procedural FX]        → fx_image  (blend: screen, opacity: 0.7)
[BG Removed Subject]   → fg_image + fg_mask  (fit: fit, position: center)
[Film Grain]           → overlay_image  (blend: overlay, opacity: 0.3)
```

**Why this matters:** Layer Composer is a full compositing node that replaces the need for external compositing software in many workflows. The 5-layer stack covers the standard compositing hierarchy: background plate, fill/tint, effects, foreground subject, and overlay. The foreground layer's dedicated placement controls (fit mode, position anchors, custom pixel coordinates) handle the precise positioning needed for subject compositing. The 14 blend modes enable everything from simple overlay to advanced light interaction effects (screen for additive glow, multiply for shadows, dodge/burn for highlight/shadow emphasis).

**Use cases beyond video editing:** Build complete green-screen compositing pipelines, layer multiple AI-generated elements, create audio visualizer backgrounds, composite live footage with generated effects, build animated social media templates.

---

### LTX 2.3 Bridge (8)

These nodes don't replace native LTX ComfyUI nodes — they bridge CutFlow's editing capabilities with LTX 2.3's generation pipeline. Use them to prepare inputs for LTX, and post-process LTX outputs.

---

#### 🎯 S42CF Guide Frame Prep

Extract and resize frames for LTX 2.3 image-to-video conditioning (LTXVAddGuide + LTXVPreprocess).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | IMAGE | — | Source video or image batch |
| `mode` | select | first_last | `first_last` (FLF2V), `first_only` (I2V), `last_only`, `keyframes`, `evenly_spaced` |
| `keyframe_indices` | string | "0, -1" | Frame indices for keyframes mode. Negative = from end. |
| `num_guides` | int | 3 | Guides for evenly_spaced mode |
| `target_width/height` | int | 768/512 | Output dimensions matching LTX empty latent |

**Outputs:** `first_frame`, `last_frame`, `guide_indices` (string), `info`

**Why this matters:** LTX 2.3's First-Last-Frame-to-Video (FLF2V) workflow requires specific frame extraction and resizing that must match the LTX latent dimensions. Guide Frame Prep automates this: give it your source video and it extracts the right frames, resizes them to your LTX target resolution, and outputs them ready to connect to LTXVAddGuide. The `evenly_spaced` mode is for multi-guide workflows where you want consistent temporal sampling.

---

#### 🔢 S42CF LTX Frame Calculator

Calculate valid LTX 2.3 frame counts (must follow the 8n+1 rule).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desired_duration` | float | 5.0 | Target duration in seconds |
| `fps` | int | 24 | Frame rate |
| `mode` | select | nearest | `nearest`, `round_up`, `round_down` |

**Outputs:** `frame_count` (INT), `duration` (FLOAT), `info`

**Valid LTX frame counts:** 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121...

**Why this matters:** LTX 2.3 only accepts frame counts that follow the 8n+1 formula. If you pass 120 frames, it fails. LTX Frame Calculator takes your desired duration in seconds and returns the nearest valid frame count. Connect the output directly to EmptyLTXVLatentVideo's frame count input. Reference values: 121 frames ≈ 5s at 24fps, 257 frames ≈ 10.7s, 513 frames ≈ 21.4s.

---

#### 🧍 S42CF Subject Isolate

Separate subject and background plates from a masked video.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Source video with subject |
| `subject_mask` | MASK | — | Subject mask from Background Remover |
| `output_mode` | select | subject_on_black | `subject_on_black`, `subject_on_white`, `subject_rgba`, `background_only`, `background_inpaint_simple` |
| `expand_mask` | int | 3 | Expand mask for cleaner edges |
| `feather` | float | 2.0 | Feather mask edges |

**Outputs:** `subject_plate`, `background_plate`, `refined_mask`, `info`

**Why this matters:** Subject Isolate takes the mask from Background Remover and produces clean plates for compositing. `subject_on_black` is the standard format for LTX guide frames — the subject on a solid background gives the AI model a clean reference. `background_inpaint_simple` fills the subject-shaped hole in the background using edge extension — a quick way to get a usable background plate for reference without running a full inpainting model.

---

#### 🏞 S42CF Background Plate

Extract or generate clean background plates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | select | gradient | `gradient`, `solid`, `from_video` |
| `color_a` / `color_b` | string | — | Gradient/solid colors |
| `width` / `height` | int | — | Output dimensions |

**Why this matters:** When you don't have a real background for compositing, Background Plate generates one. The gradient mode creates smooth color backgrounds sized to your LTX output resolution — useful as fallback plates for segmented generation where some segments lack source video.

---

#### ✂ S42CF Video Segment Prep

Split long video into LTX-compatible segments with overlap for stitching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Long source video |
| `segment_frames` | int | 121 | Frames per segment (must be 8n+1) |
| `overlap_frames` | int | 9 | Overlap between segments |
| `segment_index` | int | 0 | Which segment to output |

**Outputs:** `segment`, `first_frame`, `last_frame` (both ready for LTXVAddGuide), `segment_count`, `frame_count`, `info`

**Why this matters:** LTX 2.3 has a practical single-pass limit of about 10-20 seconds. For longer videos, you need to split the source into overlapping segments, process each through LTX independently, then stitch the results. Video Segment Prep automates the splitting with proper overlap for smooth stitching, and extracts the first/last frames of each segment ready for LTXVAddGuide conditioning.

---

#### 🔊 S42CF Audio Cond Prep

Prepare audio for LTX 2.3 audio conditioning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | — | Audio input |
| `target_frames` | int | 121 | Must match your LTX empty latent |
| `fps` | int | 24 | For duration calculation |
| `normalize` | select | yes | Normalize to -3dBFS peak |
| `fade_in_ms/fade_out_ms` | int | 50/100 | Prevent click artifacts |

**Why this matters:** LTX 2.3's audio conditioning requires audio trimmed to exactly match the video duration. Audio Cond Prep normalizes, trims, and fades the audio in one node, outputting it ready for the LTX audio conditioning pipeline. The normalization ensures consistent audio levels regardless of input volume.

---

#### ✨ S42CF LTX Post-Process

Post-process LTX 2.3 output — crop guide artifacts, auto-brightness, temporal denoise.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | LTX VAE decode output |
| `crop_last_frames` | int | 2 | Remove end-frame artifacts (matches LTXVCropGuides) |
| `crop_first_frames` | int | 0 | Remove start-frame artifacts |
| `auto_brightness` | select | none | `none`, `subtle`, `moderate` |
| `denoise_strength` | float | 0.0 | Temporal denoise (0=off, 0.3=subtle) |

**Why this matters:** LTX 2.3 guide conditioning produces artifacts in the last 1-2 frames of output (the guide frames themselves leak through). LTX Post-Process crops these automatically, matching the behavior of the native LTXVCropGuides node. The auto-brightness and denoise options provide quick one-node cleanup of LTX output before it enters your editing pipeline.

---

#### 📝 S42CF Scene Describer

Analyze video frames to generate scene descriptions for LTX prompting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | IMAGE | — | Video or image to analyze |
| `detail_level` | select | basic | `basic` (color + brightness + motion) or `detailed` (adds composition, prompt suggestions) |

**Outputs:** `scene_description` (human-readable analysis), `prompt_suggestions` (suggested LTX prompt elements)

**Why this matters:** Writing good LTX prompts requires describing the visual properties of your target — color palette, lighting, motion level, composition. Scene Describer analyzes the actual source footage and generates these descriptions for you. Feed the `prompt_suggestions` output into your LTX prompt (or use it as a reference) to get generation that better matches your source material.

---

## Example Workflows

All 53 nodes are showcased across 12 workflows in the `workflows/` folder. Drag any JSON file onto ComfyUI to load it.

---

### 01 — Basic Edit & Color Grade
**File:** `workflows/01_basic_edit_and_grade.json`
**Nodes used:** QuickPreview → SmartTrim → FilterPack → ColorAdjust → Vignette → StyleTransfer → ClipInfo

The bread-and-butter editing workflow. Load a video, trim to your desired segment, sharpen, adjust brightness/contrast/saturation, add a vignette, apply a cinematic color grade, and inspect the final clip metadata. The QuickPreview at the start runs everything at 480p for fast iteration.

**How to customize:** Replace the GGF_LoadVideo source with your own file. Adjust SmartTrim in/out points. Try different StyleTransfer presets (cyberpunk, noir, vintage). Use ClipInfo at the end to verify frame count and duration.

---

### 02 — Speed Ramp & Reverse
**File:** `workflows/02_speed_ramp_and_reverse.json`
**Nodes used:** SpeedRamp → ClipReverse → FrameBlend → FilmGrain → ThumbnailStrip

Create dramatic speed effects. The SpeedRamp uses the "bullet_time" preset for a slow-mo center section. A parallel branch shows ClipReverse at half speed. FrameBlend smooths any jitter from speed changes, FilmGrain adds texture, and ThumbnailStrip generates a visual contact sheet of the final result.

**How to customize:** Change the SpeedRamp preset (try `hero_moment` or `whip_pan`). Adjust ClipReverse speed (0.25 for dramatic slow-mo, 2.0 for fast reverse). Increase FrameBlend window if you see stuttering.

---

### 03 — Cut & Assemble
**File:** `workflows/03_cut_and_assemble.json`
**Nodes used:** MultiSplit → FrameExtract → FrameHold → FrameInsert → LoopBounce → SideBySide

Multi-segment editing. MultiSplit divides a clip at beat points. FrameExtract pulls keyframes. FrameHold creates a freeze-frame intro. FrameInsert patches the freeze into the segment. LoopBounce creates a ping-pong loop. SideBySide compares the original vs. the edited result.

**How to customize:** Change the split_points to match your footage's natural cut points. Adjust FrameHold duration for longer/shorter freeze frames. Try LoopBounce in ping_pong mode with crossfade for seamless loops.

---

### 04 — Text & Titles
**File:** `workflows/04_text_and_titles.json`
**Nodes used:** ImageToClip → TextOverlay → SubtitleBurn → LowerThird → Watermark

Full title and text pipeline. ImageToClip turns a still title card into a 3-second clip. TextOverlay renders a large title with stroke and shadow. SubtitleBurn adds SRT subtitles with cinematic styling. LowerThird animates a name/title card with slide-in/out. Watermark stamps the final output.

**How to customize:** Edit the TextOverlay text and positioning. Paste your own SRT content into SubtitleBurn. Change the LowerThird name/title and accent color. Switch Watermark to image mode and connect your logo.

---

### 05 — Composition & Layout
**File:** `workflows/05_composition_and_layout.json`
**Nodes used:** KenBurns → PictureInPicture → SplitScreen → CropPad → AspectConvert → BatchResize

Multi-clip composition. KenBurns adds a slow zoom/pan to the main clip. PictureInPicture overlays a webcam-style feed with rounded corners. SplitScreen creates a left/right comparison. CropPad adjusts to 16:9. AspectConvert reformats for vertical (9:16). BatchResize outputs at 1920×1080.

**How to customize:** Adjust KenBurns zoom range for more/less dramatic camera movement. Change PiP position and scale. Try CropPad with different ratio presets (1:1 for Instagram, 9:16 for Reels).

---

### 06 — VFX & Glitch Art
**File:** `workflows/06_vfx_and_glitch.json`
**Nodes used:** ChromaticAberration → LensDistortion → TemporalFX → GlitchFX → ChannelOps → HistogramOverlay

Creative effects chain. ChromaticAberration adds radial RGB fringing. LensDistortion applies barrel distortion. TemporalFX creates ghosting echoes. GlitchFX adds RGB split glitch. ChannelOps inverts colors. HistogramOverlay monitors the RGB levels through the chain.

**How to customize:** Reduce ChromaticAberration intensity for subtle realism or increase for heavy VFX. Try different GlitchFX modes (datamosh, pixel_sort, block_corrupt). Remove ChannelOps invert to keep original colors with the other effects.

---

### 07 — Transitions, Masks & Stabilize
**File:** `workflows/07_transitions_and_masks.json`
**Nodes used:** Stabilize → MaskWipe → ShapeMask

Two-clip transition with stabilization. Stabilize de-shakes clip A before the transition. MaskWipe performs a radial-out reveal from clip A to clip B with soft edges. ShapeMask generates an animated ellipse mask with size keyframes for use in other compositing scenarios.

**How to customize:** Increase Stabilize smoothing for a more locked-off look. Try different MaskWipe types (diamond, clock, gradient). Adjust ShapeMask keyframes for animated reveals.

---

### 08 — Audio-Synced Edit & GIF Export
**File:** `workflows/08_audio_sync_and_export.json`
**Nodes used:** BeatSnap → AudioTrim → AudioVideoSync → ClipToGIF

Audio-driven editing. BeatSnap analyzes the audio track and outputs beat frame indices (connect to MultiSplit for beat-synced cuts). AudioTrim cuts the audio with fade in/out. AudioVideoSync matches audio duration to video. ClipToGIF exports a separate animated GIF of the clip.

**How to customize:** Adjust BeatSnap sensitivity (lower = fewer, stronger beats only). Connect beat_data output to a MultiSplit node for automatic beat-synced cuts. Change ClipToGIF fps and max_colors for quality/size tradeoffs.

---

### 09 — Background Removal + Layer Compositing *(Expanded)*
**File:** `workflows/09_bg_removal_and_compositing.json`
**Nodes used:** BackgroundRemover → LayerComposer

The foundational compositing workflow. Load a subject video, remove its background with BiRefNet AI, then composite the isolated subject onto a new background video using the 5-layer Layer Composer. The BG remover outputs the mask directly to the composer's foreground mask input. Add FX layers, overlays, or fill layers as needed.

**How to customize:** Try different BackgroundRemover methods (birefnet-portrait for people, chroma_green for green screen). Adjust temporal_smooth for video (2-3 eliminates flicker). Add layers to the composer: FX layer for particle effects, overlay for grain.

---

### 10 — LTX 2.3 FLF2V + CutFlow Post-Processing *(Expanded)*
**File:** `workflows/10_ltx23_flf2v_post_processing.json`
**Nodes used:** LTXFrameCalc → GuideFramePrep → *(LTX pipeline)* → LTXPostProcess → StyleTransfer → Vignette

Integrates CutFlow with your LTX 2.3 First-Last-Frame-to-Video workflow. GuideFramePrep extracts and resizes guide frames for LTXVAddGuide. FrameCalc computes valid 8n+1 frame counts. After LTX generates, LTXPostProcess crops guide artifacts (last 2 frames), then StyleTransfer and Vignette add cinematic post-processing. Connect the standard LTX 2.3 pipeline (DualCLIPLoader → KSampler → VAEDecodeTiled) in between.

**How to customize:** Change GuideFramePrep target dimensions to match your LTX latent. Use FrameCalc to find valid frame counts for your desired duration. Try different StyleTransfer presets on the LTX output. Adjust LTXPostProcess crop_last_frames if you see guide artifacts.

---

### 11 — Subject Replacement with LTX 2.3 *(Expanded)*
**File:** `workflows/11_subject_replacement_ltx23.json`
**Nodes used:** BackgroundRemover → SubjectIsolate → GuideFramePrep → SceneDescriber → LayerComposer

The "replace everything except the subject" workflow. Background Remover isolates the subject. SubjectIsolate separates the subject plate and background plate (with optional simple inpainting). GuideFramePrep extracts frames from the background plate for LTX guide conditioning. SceneDescriber analyzes the scene to help write the regeneration prompt. LTX 2.3 generates a new background. Layer Composer composites the original subject onto the AI-generated background.

**How to customize:** Use SceneDescriber's prompt_suggestions output to inform your LTX prompt. Adjust SubjectIsolate expand_mask and feather for cleaner edges. Experiment with Layer Composer blend modes on the foreground layer.

---

### 12 — Long-form Segmented LTX 2.3 Generation *(Expanded)*
**File:** `workflows/12_longform_segmented_ltx23.json`
**Nodes used:** VideoSegmentPrep → AudioCondPrep → LTXFrameCalc → LTXPostProcess → BackgroundPlate

For videos longer than LTX 2.3's single-pass limit (~10-20 seconds). VideoSegmentPrep splits the source into overlapping 121-frame segments with 9-frame overlap. Each segment's first and last frames are extracted for guide conditioning. AudioCondPrep normalizes and trims the audio to match. Process each segment through LTX independently, then stitch results back together. BackgroundPlate provides fallback gradient backgrounds for segments without source video.

**How to customize:** Adjust segment_frames based on your VRAM (121 for 8GB, higher for more). Increase overlap_frames for smoother segment transitions. Use BackgroundPlate to fill segments where source video is missing.

---

### How to Use Any Workflow

1. Open ComfyUI
2. Drag any `.json` file from `workflows/` onto the ComfyUI canvas
3. Connect your own video source to the Load Video nodes
4. Adjust parameters to taste
5. Queue prompt to render

**Note:** The Load Video and Video Combine nodes use GGF/VHS types. If you use different loader/saver nodes, simply swap them out — all CutFlow nodes use standard IMAGE batches.

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

**Only `opencv-python` needs to be installed** for all 42 core nodes + transitions. The expanded Background Remover AI methods need `transformers` (often already installed by other nodes) or `rembg` — but traditional methods (chroma key, luma key) work with just opencv.

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
1. Verify installation path: `ComfyUI/custom_nodes/S42-CutFlow/__init__.py` must exist
2. Check the console for import errors at startup
3. Restart ComfyUI completely (not just refresh)
4. If you see "No module named 'cv2'", install opencv-python (see above)

### LTX 2.3 frame count errors
- LTX requires 8n+1 frame counts: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, ...
- Use LTXFrameCalc to automatically compute valid counts from desired duration.
- 121 frames = ~5s at 24fps. 257 frames = ~10.7s. 513 frames = ~21.4s.

### Background Remover model download issues
- BiRefNet models auto-download from HuggingFace (~400MB each). Ensure internet access on first use.
- Models are cached in your HuggingFace cache directory after first download.
- If download fails, check your firewall/proxy settings.

### Color picker not showing
- The color picker widget requires the web extension in `web/js/color_picker.js`.
- If color pickers show as plain text inputs, verify the `web/` directory exists and ComfyUI loaded the extension.

---

## Project Structure

```
S42-CutFlow/
├── __init__.py           # Node registration (42 core + 11 expanded = 53 nodes)
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
├── cf_transitions.py     # [EXPANDED] 20 CapCut-style transitions
├── cf_bg_remover.py      # [EXPANDED] Background removal (AI + traditional)
├── cf_layer_composer.py  # [EXPANDED] 5-layer video compositor
├── cf_ltx_bridge.py      # [EXPANDED] LTX 2.3 bridge (8 nodes)
├── web/
│   └── js/
│       └── color_picker.js  # Hex color picker widget for ComfyUI
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

*53 nodes. 12 workflows. LTX 2.3 ready. Mostly harmless.*
