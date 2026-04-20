# S42 CutFlow — Video Editing Suite for ComfyUI

Workflows are being tweaked, audio editing nodes are coming also. Bringing all my tools to this one suite. 

**42 core + 28 expanded = 70 nodes for professional video editing within ComfyUI's node graph.** CapCut-class editing power with zero GPU model dependencies, combined with professional audio DSP and experimental neural latent hacking. Trim, split, speed ramp, filter, color grade, overlay text, picture-in-picture, stabilize, beat-sync, transitions, background removal, multi-layer compositing, LTX 2.3 bridge utilities, and spectral audio manipulation — all within your ComfyUI workflow.

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
- [Node Reference — Expanded (28 Nodes)](#node-reference--expanded-28-nodes)
  - [Transitions (1)](#transitions-1)
  - [Background Remover (1)](#background-remover-1)
  - [Layer Composer (1)](#layer-composer-1)
  - [LTX 2.3 Bridge (8)](#ltx-23-bridge-8)
  - [Advanced Audio & Neural Latents (17)](#advanced-audio--neural-latents-17)
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
| **LTX Bridge** *(expanded)* | 8 | Guide frame prep, frame calculator, subject isolate, background plate, segment prep, audio conditioning, post-process, scene describer |
| **Adv Audio & Latent** *(exp)*| 17 | 6-track audio mixing, spectral mashups, neural latent dimension hacking |
| ***Expanded Total*** | ***28*** | *Transitions + AI generation integration + Advanced DSP* |
| **Grand Total** | **70** | **Complete editing + generation + audio pipeline** |

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

After restart, look for nodes under the **S42 CutFlow** category in the node menu. You should see 70 nodes across core and expanded subcategories. The console will print:

```
[S42 CutFlow] Loaded 42 core + 28 expanded = 70 total nodes
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

### Audio Mixing & Spectral Mashup

```
[Load Audio A] → [S42CF Spectral Mashup Engine] ← [Load Audio B]
                        ↓
                 [S42CF Gabor Harmonic Transfuser]
```

---

## Node Reference — Core (42 Nodes)

Every parameter on every node has a tooltip — hover over any input in ComfyUI to see its description. All nodes output an `info` STRING with a human-readable summary of what they did.

---

### Clip Operations (8)

These are the fundamental building blocks of any video edit — cutting, rearranging, and retiming footage. If you've used a timeline-based editor, these are your razor blade, slip tool, and speed controls translated into node form.

#### ✂ S42CF Smart Trim
Trim a clip to in/out points with frame or timecode precision.

#### ✂ S42CF Multi Split
Split a clip at multiple points into segments.

#### 🔁 S42CF Loop Bounce
Loop a clip with optional ping-pong and crossfade.

#### ⏸ S42CF Frame Hold
Freeze a specific frame for N frames.

#### 📋 S42CF Frame Extract
Extract specific frames by index, range, or interval.

#### 📥 S42CF Frame Insert
Insert frames at a specific position within a clip.

#### ⏪ S42CF Clip Reverse
Reverse a clip with optional speed modification.

#### ⏩ S42CF Speed Ramp
CapCut-style speed ramping with presets and custom keyframe curves.

---

### Filters & Color (7)

Color grading and image filtering — the tools that transform raw footage into something with visual character. These nodes cover everything from basic sharpening to full cinematic color science.

#### 🔧 S42CF Filter Pack
Multi-mode filter with 7 operations covering the most common image processing needs.

#### 🔲 S42CF Vignette
Configurable vignette with shape, color, and offset controls.

#### 🎞 S42CF Film Grain
Procedural film grain with size, color mode, and animation.

#### 🌈 S42CF Chromatic Aberration
RGB channel offset for chromatic aberration effect.

#### 🔮 S42CF Lens Distortion
Barrel or pincushion lens distortion.

#### 🎨 S42CF Style Transfer
8 cinematic color grading presets with adjustable strength.

#### 🎛 S42CF Color Adjust
Fundamental brightness, contrast, saturation, hue, gamma, and temperature adjustment.

---

### Temporal FX (3)

Time-domain effects that operate across multiple frames to create motion-based looks.

#### ⏳ S42CF Temporal FX
Time-domain effects across frame windows (echo, motion trail, time displacement, etc.).

#### 💥 S42CF Glitch FX
Digital glitch effects with 5 modes (datamosh, scan lines, pixel sort, etc.).

#### 🔄 S42CF Frame Blend
Blend adjacent frames for motion blur or AI artifact smoothing.

---

### Text & Overlays (4)

Professional text rendering for titles, subtitles, lower thirds, and watermarks.

#### 📝 S42CF Text Overlay
Rich text rendering with stroke, shadow, background, and positioning.

#### 💬 S42CF Subtitle Burn
Burn SRT subtitles onto frames with style presets.

#### 🔖 S42CF Watermark
Image or text watermark with tiling support.

#### 📺 S42CF Lower Third
Animated lower-third name/title cards with slide-in/out animation.

---

### Composition & Motion (6)

Multi-clip composition, camera effects, and masking.

#### 📷 S42CF Ken Burns
Apply animated pan and zoom to still images or video clips.

#### 🖼 S42CF Picture-in-Picture
Overlay one clip on another with position, scale, and rounded corners.

#### ↔ S42CF Split Screen
Side-by-side or stacked clip comparison.

#### ✂ S42CF Crop/Pad
Crop, pad, letterbox, or pillarbox to target dimensions or aspect ratios.

#### 🎭 S42CF Mask Wipe
Animated mask-based transitions between two clips.

#### ⬡ S42CF Shape Mask
Generate animated shape masks for compositing.

---

### Stabilization (1)

#### 📐 S42CF Stabilize
Software video stabilization using OpenCV optical flow tracking.

---

### Audio Sync (3)

#### 🔊 S42CF Audio Trim
Trim audio to match clip length with fade in/out.

#### 🥁 S42CF Beat Snap
Detect beats and output frame indices for synced cuts.

#### 🔗 S42CF Audio-Video Sync
Match audio duration to video duration.

---

### Preview & Analysis (5)

#### 👁 S42CF Quick Preview
Downsample for fast iteration, passthrough for final render.

#### 🎞 S42CF Thumbnail Strip
Generate a filmstrip contact sheet for visual review.

#### ℹ S42CF Clip Info
Display metadata about a clip: resolution, frame count, duration.

#### 🔍 S42CF Side-by-Side Compare
A/B comparison of two clips at the same resolution.

#### 📊 S42CF Histogram Overlay
RGB histogram and waveform overlay on the video.

---

### Utilities (5)

#### 📐 S42CF Aspect Convert
Convert between common aspect ratios.

#### 📏 S42CF Batch Resize
Resize all frames to specific dimensions.

#### 🔀 S42CF Channel Ops
Per-channel operations: swap, invert, isolate, grayscale.

#### 🖼 S42CF Image to Clip
Convert a single image to a multi-frame clip.

#### 🎬 S42CF Clip to GIF
Export a clip as an animated GIF.

---

## Node Reference — Expanded (28 Nodes)

These nodes extend CutFlow with professional transitions, AI-powered background removal, multi-layer compositing, LTX 2.3 bridge utilities, and advanced DSP/Latent hacking.

### Transitions (1)

#### 🔀 S42CF Transition
20 CapCut-style transitions between two video clips. Supports blending transitions (dissolve, fade, glitch) and motion transitions (push, wipe, zoom, spin, iris, slide) that use stable reference frames to prevent jitter.

### Background Remover (1)

#### 🎭 S42CF Background Remover
Professional background removal for images and video batches. Supports 16 AI methods (BiRefNet, rembg, U2Net, ISNet) plus 4 traditional methods (chroma/luma key). Batch-optimized with temporal consistency for video.

### Layer Composer (1)

#### 🎬 S42CF Layer Composer
Professional 5-layer video/image compositor with 14 blend modes. Includes full alpha-aware compositing and absolute foreground placement controls.

### LTX 2.3 Bridge (8)

#### 🎯 S42CF Guide Frame Prep
Extract and resize frames for LTX 2.3 image-to-video conditioning.

#### 🔢 S42CF LTX Frame Calculator
Calculate valid LTX 2.3 frame counts (must follow the 8n+1 rule).

#### 🧍 S42CF Subject Isolate
Separate subject and background plates from a masked video.

#### 🏞 S42CF Background Plate
Extract or generate clean background plates.

#### ✂ S42CF Video Segment Prep
Split long video into LTX-compatible segments with overlap for stitching.

#### 🔊 S42CF Audio Cond Prep
Prepare audio for LTX 2.3 audio conditioning.

#### ✨ S42CF LTX Post-Process
Post-process LTX 2.3 output — crop guide artifacts, auto-brightness, temporal denoise.

#### 📝 S42CF Scene Describer
Analyze video frames to generate scene descriptions for LTX prompting.

---

### Advanced Audio & Neural Latents (17)

This suite bridges the gap between traditional Digital Signal Processing (DSP) and Neural Sound Design, operating directly on waveform tensors and compressed semantic VAE latents.

**The Audio Mixer & Dynamics:**
* **🎬 Studio42 Audio Mixer:** A 6-track professional mixer featuring individual track faders, panning, S-curve fades, dynamic range compression, soft-limiting clipping prevention, and LUFS broadcast standardization.
* **Dynamic Range Compressor:** Professional peak/RMS compressor to squash harsh transients and lift nuances.
* **Phase Vocoder Time Stretch:** Stretch or compress audio tempo mathematically without affecting pitch.
* **Gabor Denoiser:** Advanced spectral denoiser utilizing Gabor atom modeling to preserve natural harmonics.

**Spectral DSP & Mashups:**
* **Spectral Mashup Engine:** A deterministic pseudo-latent engine that merges two tracks. Features Auto-BPM detection via spectral flux, automatic phase-vocoder time synchronization, semitone pitch shifting, and dynamic spectral ducking to carve out EQ space automatically.
* **Gabor Harmonic Transfuser:** Morph the acoustic "DNA" of sounds. Uses Source-Filter Cross-Synthesis to apply the rhythm/words of one track to the physical timbre/voice of another. 
* **Vocal Resonance Sculptor:** Precision harmonic exciter that targets sub-harmonics to emulate velvety, vintage baritone profiles.

**Neural Latent Circuit Bending (Experimental):**
* *These nodes bypass waveforms entirely, manipulating the AI's "concept" of sound before decoding.*
* **Audio Latent Encoder / Decoder:** Bridge nodes to encode/decode audio using VAEs (e.g., AceStep, LTX).
* **Neural Latent Mixer:** Mashes up two songs purely in semantic space. Perform *Acoustic Interpolation* (hallucinating hybrid instruments) or *Feature Swaps* (splicing drum dimensions into vocal channels).
* **Latent Disintegration:** Injects structural Gaussian entropy directly into the tensor, causing the acoustic space to dissolve into digital smoke.
* **Audio-Visual Synesthesia:** Forcibly reshapes an **Image Latent** (like an LTX video frame) into a temporal sequence and feeds it into the Audio VAE, allowing you to literally listen to the visual structure of your generated images.
* **AceStep Latent Modifier:** Directly alters AceStep latents to create rhythmic stuttering and quantized digital artifacts.

**Analysis & Sync:**
* **LTX Audio Sync Trigger:** Analyzes audio transients to output float curves for driving LTX motion scales.
* **Onset Envelope Visualizer:** Generates a deterministic visual heatmap (barcode) of audio transients.

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
| scipy | Yes | ✅ Yes | Signal processing, Gabor atoms |
| PyAV (av) | Yes | ✅ Yes (≥14.2) | Audio processing |
| torchaudio | Yes | ✅ Yes | DSP, Sox effects, Resampling |
| opencv-python | Yes | ❌ Install | Stabilize, bilateral, filters |
| transformers | Optional | 🔸 Often present | BiRefNet AI background removal |
| rembg | Optional | ❌ Optional | U2Net/ISNet/SILUETA background removal |

**Only `opencv-python` needs to be installed** for all core nodes to function.

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
├── __init__.py           # Node registration (70 nodes)
├── cf_utils.py           # Shared utilities, easing, keyframes
├── cf_clip_ops.py        # Trim, split, loop, hold, extract, insert, reverse
├── cf_speed_ramp.py      # Speed ramping with curves
├── cf_filters.py         # Filter pack, vignette, grain, CA, distortion, style
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
├── cf_audio_advanced.py  # [EXPANDED] Advanced DSP, Mixers & Neural Latents (17 nodes)
├── web/
│   └── js/
│       └── color_picker.js  # Hex color picker widget for ComfyUI
├── requirements.txt      # opencv-python
├── LICENSE               # MIT
├── README.md             # This file
└── workflows/            # Example workflow JSON files
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Willie Gray Jr](https://github.com/GeekyGhost) as part of the S42 Production Suite ecosystem.*

*70 nodes. LTX 2.3 ready. Mostly harmless.*
