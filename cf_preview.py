"""
S42 CutFlow — Preview & Analysis
===================================
Quick preview, thumbnail strips, clip info, side-by-side compare,
and monitoring overlays (histogram, waveform, vectorscope).

Python 3.12 | ComfyUI Portable | torch + numpy + PIL
"""

import torch
import numpy as np
import logging

from .cf_utils import (
    ensure_rgb, resize_frames, frames_to_np, np_to_frames,
    frame_to_pil, pil_to_frame, PIL_AVAILABLE
)

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Preview"

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw


class S42CF_QuickPreview:
    """Downsample for fast iteration, passthrough for final render."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to preview."}),
                "mode": (["preview", "passthrough"], {
                    "default": "preview",
                    "tooltip": "'preview' = downsample to preview_height for fast processing.\n'passthrough' = no-op, full resolution passes through unchanged.\nSwitch to passthrough for final render."
                }),
                "preview_height": ("INT", {
                    "default": 480, "min": 120, "max": 1080, "step": 8,
                    "tooltip": "Preview resolution height in pixels. Width scales proportionally. 480 = fast, 720 = balanced."
                }),
                "frame_skip": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Process every Nth frame for even faster preview. 1 = all frames, 2 = half, etc."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, preview_height, frame_skip):
        clip = ensure_rgb(clip)
        if mode == "passthrough":
            return (clip, int(clip.shape[0]), f"QuickPreview: passthrough ({clip.shape[0]}f @ {clip.shape[2]}x{clip.shape[1]})")

        n, h, w, c = clip.shape
        if frame_skip > 1:
            indices = list(range(0, n, frame_skip))
            clip = clip[indices]
            n = clip.shape[0]

        if h != preview_height:
            scale = preview_height / h
            new_w = int(w * scale)
            new_w = (new_w // 8) * 8
            clip = resize_frames(clip, preview_height, new_w)

        info = f"QuickPreview: {h}x{w} → {clip.shape[1]}x{clip.shape[2]}, skip={frame_skip}, {clip.shape[0]}f"
        return (clip, int(clip.shape[0]), info)


class S42CF_ThumbnailStrip:
    """Generate a filmstrip thumbnail grid from a clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to create filmstrip from."}),
                "columns": ("INT", {
                    "default": 8, "min": 2, "max": 20, "step": 1,
                    "tooltip": "Number of thumbnails per row."
                }),
                "thumb_height": ("INT", {
                    "default": 120, "min": 32, "max": 480, "step": 8,
                    "tooltip": "Height of each thumbnail in pixels."
                }),
                "max_frames": ("INT", {
                    "default": 24, "min": 2, "max": 200, "step": 1,
                    "tooltip": "Maximum number of thumbnails. Frames are evenly sampled across clip."
                }),
                "show_frame_numbers": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Overlay frame number on each thumbnail."
                }),
                "border": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Border/gap between thumbnails in pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("filmstrip", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, columns, thumb_height, max_frames, show_frame_numbers, border):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape
        num_thumbs = min(max_frames, n)
        indices = [int(i * (n - 1) / max(num_thumbs - 1, 1)) for i in range(num_thumbs)]

        scale = thumb_height / h
        thumb_width = int(w * scale)
        rows = (num_thumbs + columns - 1) // columns
        grid_w = columns * (thumb_width + border) + border
        grid_h = rows * (thumb_height + border) + border

        if PIL_AVAILABLE:
            grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
            draw = ImageDraw.Draw(grid)
            try:
                from PIL import ImageFont
                font = ImageFont.load_default(size=max(10, thumb_height // 8))
            except Exception:
                font = ImageFont.load_default()

            for t, idx in enumerate(indices):
                row_i = t // columns
                col_i = t % columns
                x = col_i * (thumb_width + border) + border
                y = row_i * (thumb_height + border) + border

                frame = frame_to_pil(clip[idx])
                frame = frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                grid.paste(frame, (x, y))

                if show_frame_numbers == "yes":
                    draw.text((x + 4, y + 2), str(idx), font=font, fill=(255, 255, 0))

            result = pil_to_frame(grid).unsqueeze(0)
        else:
            result = clip[:1]

        info = f"ThumbnailStrip: {num_thumbs} frames, {columns} cols, {rows} rows, {grid_w}x{grid_h}"
        return (result, info)


class S42CF_ClipInfo:
    """Output clip metadata and statistics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to analyze."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Assumed FPS for duration calculation."
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame_count", "width", "height", "duration_seconds", "aspect_ratio", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, fps):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape
        duration = n / fps
        aspect = w / h

        mean_brightness = clip.mean().item()
        min_val = clip.min().item()
        max_val = clip.max().item()
        std_val = clip.std().item()

        info = (f"ClipInfo: {n} frames, {w}x{h}, {duration:.2f}s @ {fps}fps\n"
                f"Aspect: {aspect:.3f} ({w}:{h})\n"
                f"Brightness: mean={mean_brightness:.3f}, range=[{min_val:.3f}, {max_val:.3f}], std={std_val:.3f}")

        return (n, w, h, round(duration, 3), round(aspect, 4), info)


class S42CF_SideBySide:
    """Place two clips side-by-side for A/B comparison."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "Clip A (left/top)."}),
                "clip_b": ("IMAGE", {"tooltip": "Clip B (right/bottom)."}),
                "layout": (["horizontal", "vertical"], {
                    "default": "horizontal",
                    "tooltip": "'horizontal' = side by side. 'vertical' = top and bottom."
                }),
                "label_a": ("STRING", {"default": "Before", "tooltip": "Label for clip A."}),
                "label_b": ("STRING", {"default": "After", "tooltip": "Label for clip B."}),
                "gap": ("INT", {"default": 4, "min": 0, "max": 20, "step": 2,
                    "tooltip": "Gap between clips in pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip_a, clip_b, layout, label_a, label_b, gap):
        a = ensure_rgb(clip_a)
        b = ensure_rgb(clip_b)

        h_a, w_a = a.shape[1], a.shape[2]
        h_b, w_b = b.shape[1], b.shape[2]

        if layout == "horizontal":
            target_h = max(h_a, h_b)
            a = resize_frames(a, target_h, w_a) if h_a != target_h else a
            b = resize_frames(b, target_h, w_b) if h_b != target_h else b
        else:
            target_w = max(w_a, w_b)
            a = resize_frames(a, h_a, target_w) if w_a != target_w else a
            b = resize_frames(b, h_b, target_w) if w_b != target_w else b

        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]

        if layout == "horizontal":
            gap_tensor = torch.zeros(n, a.shape[1], gap, 3, dtype=a.dtype) if gap > 0 else None
            parts = [a, gap_tensor, b] if gap_tensor is not None else [a, b]
            result = torch.cat([p for p in parts if p is not None], dim=2)
        else:
            gap_tensor = torch.zeros(n, gap, a.shape[2], 3, dtype=a.dtype) if gap > 0 else None
            parts = [a, gap_tensor, b] if gap_tensor is not None else [a, b]
            result = torch.cat([p for p in parts if p is not None], dim=1)

        if PIL_AVAILABLE and (label_a or label_b):
            result_list = []
            for i in range(n):
                pil_frame = frame_to_pil(result[i])
                draw = ImageDraw.Draw(pil_frame)
                try:
                    font = __import__('PIL').ImageFont.load_default(size=16)
                except Exception:
                    font = __import__('PIL').ImageFont.load_default()
                if label_a:
                    draw.text((8, 8), label_a, font=font, fill=(255, 255, 0))
                if label_b:
                    if layout == "horizontal":
                        draw.text((a.shape[2] + gap + 8, 8), label_b, font=font, fill=(255, 255, 0))
                    else:
                        draw.text((8, a.shape[1] + gap + 8), label_b, font=font, fill=(255, 255, 0))
                result_list.append(pil_to_frame(pil_frame))
            result = torch.stack(result_list)

        info = f"SideBySide({layout}): {result.shape[2]}x{result.shape[1]}, {n}f"
        return (result, n, info)


class S42CF_HistogramOverlay:
    """Burn RGB histogram or waveform onto frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to overlay monitoring graphics onto."}),
                "display": (["histogram", "waveform", "rgb_parade"], {
                    "default": "histogram",
                    "tooltip": "'histogram' = RGB histogram overlay.\n'waveform' = luminance waveform.\n'rgb_parade' = side-by-side RGB waveforms."
                }),
                "position": (["top_right", "top_left", "bottom_right", "bottom_left"], {
                    "default": "top_right",
                    "tooltip": "Overlay position on frame."
                }),
                "size": ("FLOAT", {
                    "default": 0.25, "min": 0.1, "max": 0.5, "step": 0.05,
                    "tooltip": "Size of overlay relative to frame. 0.25 = quarter width."
                }),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Overlay opacity."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, display, position, size, opacity):
        if not PIL_AVAILABLE:
            return (clip, int(clip.shape[0]), "HistogramOverlay: PIL not available")

        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape
        ow = int(w * size)
        oh = int(ow * 0.75)

        pos_map = {
            "top_right": (w - ow - 10, 10),
            "top_left": (10, 10),
            "bottom_right": (w - ow - 10, h - oh - 10),
            "bottom_left": (10, h - oh - 10),
        }
        ox, oy = pos_map.get(position, (w - ow - 10, 10))

        results = []
        for i in range(n):
            frame_np = (clip[i].cpu().numpy() * 255).astype(np.uint8)
            overlay = Image.new("RGBA", (ow, oh), (0, 0, 0, int(180 * opacity)))
            draw = ImageDraw.Draw(overlay)

            if display == "histogram":
                for ch_idx, color in enumerate([(255, 60, 60), (60, 255, 60), (60, 60, 255)]):
                    hist, _ = np.histogram(frame_np[:, :, ch_idx], bins=64, range=(0, 255))
                    hist = hist.astype(np.float32) / (hist.max() + 1)
                    bar_w = max(1, ow // 64)
                    for b in range(64):
                        bar_h = int(hist[b] * (oh - 10))
                        x = b * bar_w
                        if bar_h > 0:
                            draw.rectangle([x, oh - bar_h, x + bar_w - 1, oh],
                                           fill=(*color, int(150 * opacity)))

            elif display == "waveform":
                for col in range(0, ow, 2):
                    src_col = int(col * w / ow)
                    if src_col < w:
                        lum = (0.299 * frame_np[:, src_col, 0] + 0.587 * frame_np[:, src_col, 1] + 0.114 * frame_np[:, src_col, 2])
                        for val in lum[::max(1, len(lum) // oh)]:
                            y = oh - 1 - int(val / 255 * (oh - 1))
                            y = max(0, min(y, oh - 1))
                            draw.point((col, y), fill=(0, 255, 0, int(100 * opacity)))

            elif display == "rgb_parade":
                third = ow // 3
                for ch_idx, color in enumerate([(255, 60, 60), (60, 255, 60), (60, 60, 255)]):
                    x_off = ch_idx * third
                    for col in range(0, third, 2):
                        src_col = int(col * w / third)
                        if src_col < w:
                            vals = frame_np[:, src_col, ch_idx]
                            for val in vals[::max(1, len(vals) // oh)]:
                                y = oh - 1 - int(val / 255 * (oh - 1))
                                y = max(0, min(y, oh - 1))
                                draw.point((x_off + col, y), fill=(*color, int(80 * opacity)))

            pil_frame = frame_to_pil(clip[i])
            pil_frame.paste(overlay, (ox, oy), overlay)
            results.append(pil_to_frame(pil_frame))

        result = torch.stack(results)
        info = f"HistogramOverlay({display}): position={position}, size={size}"
        return (result, n, info)


NODE_CLASS_MAPPINGS = {
    "S42CF_QuickPreview": S42CF_QuickPreview,
    "S42CF_ThumbnailStrip": S42CF_ThumbnailStrip,
    "S42CF_ClipInfo": S42CF_ClipInfo,
    "S42CF_SideBySide": S42CF_SideBySide,
    "S42CF_HistogramOverlay": S42CF_HistogramOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_QuickPreview": "👁 S42 CutFlow Quick Preview",
    "S42CF_ThumbnailStrip": "🎞 S42 CutFlow Thumbnail Strip",
    "S42CF_ClipInfo": "ℹ S42 CutFlow Clip Info",
    "S42CF_SideBySide": "↔ S42 CutFlow Side-by-Side",
    "S42CF_HistogramOverlay": "📊 S42 CutFlow Histogram Overlay",
}
