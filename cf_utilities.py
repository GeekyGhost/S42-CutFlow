"""
S42 CutFlow — Utilities
=========================
Aspect ratio conversion, batch resize, channel operations,
image-to-clip, clip-to-GIF export.

Python 3.12 | ComfyUI Portable | torch + numpy + PIL
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import logging

from .cf_utils import ensure_rgb, resize_frames, frame_to_pil, PIL_AVAILABLE

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Utilities"

if PIL_AVAILABLE:
    from PIL import Image


class S42CF_AspectConvert:
    """Convert between aspect ratios with letterbox/pillarbox/crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to convert."}),
                "target_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "2.35:1", "custom"], {
                    "default": "16:9",
                    "tooltip": "Target aspect ratio.\n'16:9' = widescreen.\n'9:16' = vertical/phone.\n'1:1' = square.\n'custom' = use custom_ratio value."
                }),
                "custom_ratio": ("FLOAT", {
                    "default": 1.778, "min": 0.2, "max": 5.0, "step": 0.001,
                    "tooltip": "Custom aspect ratio (width/height). 1.778 = 16:9. Only used when target_ratio='custom'."
                }),
                "fit_mode": (["letterbox", "crop_center", "crop_smart", "stretch"], {
                    "default": "letterbox",
                    "tooltip": "'letterbox' = add black bars to preserve content.\n'crop_center' = crop from center to fill.\n'crop_smart' = center-weighted crop.\n'stretch' = distort to fit."
                }),
                "bg_color": ("STRING", {"default": "#000000",
                    "tooltip": "Background/letterbox color (hex). Click swatch to pick."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, target_ratio, custom_ratio, fit_mode, bg_color):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        ratios = {"16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4, "1:1": 1.0, "21:9": 21/9, "2.35:1": 2.35}
        ratio = ratios.get(target_ratio, custom_ratio)

        if w / h > ratio:
            tw = int(h * ratio)
            th = h
        else:
            tw = w
            th = int(w / ratio)
        tw = max(8, (tw // 8) * 8)
        th = max(8, (th // 8) * 8)

        if fit_mode == "stretch":
            result = resize_frames(clip, th, tw)
        elif fit_mode in ("crop_center", "crop_smart"):
            scale = max(tw / w, th / h)
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            scaled = resize_frames(clip, scaled_h, scaled_w)
            cx, cy = scaled_w // 2, scaled_h // 2
            x0 = max(0, cx - tw // 2)
            y0 = max(0, cy - th // 2)
            result = scaled[:, y0:y0+th, x0:x0+tw]
        else:
            from .cf_utils import hex_to_rgb as _hex3
            ar, ag, ab = _hex3(bg_color)
            bg_t = torch.tensor([ar, ag, ab])
            result = bg_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, th, tw, 3).clone()
            scale = min(tw / w, th / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = resize_frames(clip, new_h, new_w)
            y_off = (th - new_h) // 2
            x_off = (tw - new_w) // 2
            result[:, y_off:y_off+new_h, x_off:x_off+new_w] = resized

        info = f"AspectConvert: {w}x{h} → {result.shape[2]}x{result.shape[1]} ({target_ratio}, {fit_mode})"
        return (result, int(result.shape[0]), info)


class S42CF_BatchResize:
    """Resize all frames with multiple mode options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to resize."}),
                "mode": (["exact", "fit_width", "fit_height", "percentage", "max_dimension"], {
                    "default": "exact",
                    "tooltip": "'exact' = resize to exact width x height.\n'fit_width' = set width, height scales proportionally.\n'fit_height' = set height, width scales.\n'percentage' = scale by percentage.\n'max_dimension' = scale so longest side = max_dim."
                }),
                "width": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Target width (exact, fit_width modes)."}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Target height (exact, fit_height modes)."}),
                "percentage": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 400.0, "step": 1.0,
                    "tooltip": "Scale percentage (percentage mode). 50 = half size, 200 = double."}),
                "max_dim": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Max dimension for longest side (max_dimension mode)."}),
                "interpolation": (["bilinear", "nearest", "bicubic", "lanczos"], {
                    "default": "bilinear",
                    "tooltip": "Resize interpolation method.\n'bilinear' = smooth (default).\n'nearest' = pixel-art safe.\n'bicubic'/'lanczos' = high quality (uses PIL)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, width, height, percentage, max_dim, interpolation):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        if mode == "exact":
            tw, th = width, height
        elif mode == "fit_width":
            tw = width
            th = int(h * width / w)
        elif mode == "fit_height":
            th = height
            tw = int(w * height / h)
        elif mode == "percentage":
            tw = int(w * percentage / 100)
            th = int(h * percentage / 100)
        elif mode == "max_dimension":
            scale = max_dim / max(w, h)
            tw = int(w * scale)
            th = int(h * scale)
        else:
            tw, th = width, height

        tw = max(8, (tw // 8) * 8)
        th = max(8, (th // 8) * 8)

        if interpolation in ("lanczos", "bicubic") and PIL_AVAILABLE:
            resampler = Image.Resampling.LANCZOS if interpolation == "lanczos" else Image.Resampling.BICUBIC
            results = []
            for i in range(n):
                pil = frame_to_pil(clip[i])
                resized = pil.resize((tw, th), resampler)
                results.append(torch.from_numpy(np.array(resized).astype(np.float32) / 255.0))
            result = torch.stack(results)
        else:
            torch_mode = "nearest" if interpolation == "nearest" else "bilinear"
            result = resize_frames(clip, th, tw, torch_mode)

        info = f"BatchResize({mode}): {w}x{h} → {tw}x{th} ({interpolation})"
        return (result, int(result.shape[0]), info)


class S42CF_ChannelOps:
    """Extract, swap, merge, or manipulate color channels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip for channel operations."}),
                "operation": ([
                    "extract_red", "extract_green", "extract_blue",
                    "extract_luminance", "extract_alpha",
                    "swap_rb", "swap_rg", "swap_gb",
                    "to_grayscale", "invert",
                    "threshold",
                    "channel_multiply", "channel_add",
                ], {
                    "default": "to_grayscale",
                    "tooltip": "Channel operation to perform.\n"
                               "extract_* = output single channel as grayscale.\n"
                               "swap_* = swap two channels.\n"
                               "to_grayscale = luminance conversion.\n"
                               "invert = 1-pixel values.\n"
                               "threshold = binary threshold.\n"
                               "channel_multiply/add = math between channels."
                }),
                "threshold_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold level for 'threshold' operation. Pixels above = white, below = black."
                }),
                "multiply_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Multiplier for channel_multiply operation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, operation, threshold_value, multiply_factor):
        clip = ensure_rgb(clip)

        if operation == "extract_red":
            ch = clip[:, :, :, 0:1].repeat(1, 1, 1, 3)
            result = ch
        elif operation == "extract_green":
            ch = clip[:, :, :, 1:2].repeat(1, 1, 1, 3)
            result = ch
        elif operation == "extract_blue":
            ch = clip[:, :, :, 2:3].repeat(1, 1, 1, 3)
            result = ch
        elif operation == "extract_luminance":
            lum = 0.299 * clip[:, :, :, 0] + 0.587 * clip[:, :, :, 1] + 0.114 * clip[:, :, :, 2]
            result = lum.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif operation == "swap_rb":
            result = clip[:, :, :, [2, 1, 0]]
        elif operation == "swap_rg":
            result = clip[:, :, :, [1, 0, 2]]
        elif operation == "swap_gb":
            result = clip[:, :, :, [0, 2, 1]]
        elif operation == "to_grayscale":
            lum = 0.299 * clip[:, :, :, 0] + 0.587 * clip[:, :, :, 1] + 0.114 * clip[:, :, :, 2]
            result = lum.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif operation == "invert":
            result = 1.0 - clip
        elif operation == "threshold":
            lum = 0.299 * clip[:, :, :, 0] + 0.587 * clip[:, :, :, 1] + 0.114 * clip[:, :, :, 2]
            binary = (lum > threshold_value).float()
            result = binary.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif operation == "channel_multiply":
            result = (clip * multiply_factor).clamp(0, 1)
        elif operation == "channel_add":
            result = (clip + multiply_factor - 1.0).clamp(0, 1)
        else:
            result = clip

        info = f"ChannelOps({operation}): {result.shape[0]}f"
        return (result, int(result.shape[0]), info)


class S42CF_ImageToClip:
    """Convert a single image to an N-frame clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Single image (or first frame of batch) to convert to video clip."}),
                "duration_mode": (["frames", "seconds"], {
                    "default": "seconds",
                    "tooltip": "'frames' = specify exact frame count. 'seconds' = specify duration."
                }),
                "frame_count": ("INT", {"default": 48, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Number of frames (used in 'frames' mode)."}),
                "duration_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 300.0, "step": 0.1,
                    "tooltip": "Duration in seconds (used in 'seconds' mode)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "FPS for seconds-to-frames conversion."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, image, duration_mode, frame_count, duration_seconds, fps):
        image = ensure_rgb(image)
        frame = image[0:1]

        if duration_mode == "seconds":
            n = max(1, int(duration_seconds * fps))
        else:
            n = max(1, frame_count)

        result = frame.repeat(n, 1, 1, 1)
        info = f"ImageToClip: still → {n}f ({n / fps:.2f}s @ {fps}fps)"
        return (result, n, info)


class S42CF_ClipToGIF:
    """Export clip as optimized GIF file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to export as GIF."}),
                "fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "GIF playback FPS. GIF supports up to ~50fps. 12-15 typical."}),
                "max_colors": ("INT", {"default": 256, "min": 16, "max": 256, "step": 16,
                    "tooltip": "Maximum colors in GIF palette. 256 = best quality. 64 = smaller file."}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100,
                    "tooltip": "Number of loops. 0 = infinite loop."}),
                "dither": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Apply Floyd-Steinberg dithering for smoother gradients at lower color counts."
                }),
                "filename_prefix": ("STRING", {"default": "cutflow_gif", "tooltip": "Output filename prefix."}),
                "optimize": (["yes", "no"], {"default": "yes", "tooltip": "Optimize GIF file size."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("clip_passthrough", "filepath")
    FUNCTION = "process"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def process(self, clip, fps, max_colors, loop_count, dither, filename_prefix, optimize):
        if not PIL_AVAILABLE:
            return (clip, "ERROR: PIL not available for GIF export")

        clip = ensure_rgb(clip)
        n = clip.shape[0]
        duration_ms = int(1000 / fps)

        frames_pil = []
        for i in range(n):
            pil_frame = frame_to_pil(clip[i])
            if max_colors < 256 or dither == "yes":
                dither_mode = Image.Dither.FLOYDSTEINBERG if dither == "yes" else Image.Dither.NONE
                pil_frame = pil_frame.quantize(colors=max_colors, dither=dither_mode).convert("RGB")
            frames_pil.append(pil_frame)

        import folder_paths
        output_dir = folder_paths.get_output_directory()
        existing = [f for f in os.listdir(output_dir) if f.startswith(filename_prefix) and f.endswith(".gif")]
        counter = len(existing) + 1
        filename = f"{filename_prefix}_{counter:04d}.gif"
        filepath = os.path.join(output_dir, filename)

        frames_pil[0].save(
            filepath,
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration_ms,
            loop=loop_count,
            optimize=optimize == "yes",
        )

        logger.info(f"GIF saved: {filepath} ({n}f, {fps}fps)")
        return (clip, filepath)


NODE_CLASS_MAPPINGS = {
    "S42CF_AspectConvert": S42CF_AspectConvert,
    "S42CF_BatchResize": S42CF_BatchResize,
    "S42CF_ChannelOps": S42CF_ChannelOps,
    "S42CF_ImageToClip": S42CF_ImageToClip,
    "S42CF_ClipToGIF": S42CF_ClipToGIF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_AspectConvert": "📐 S42 CutFlow Aspect Convert",
    "S42CF_BatchResize": "↔ S42 CutFlow Batch Resize",
    "S42CF_ChannelOps": "🔴 S42 CutFlow Channel Ops",
    "S42CF_ImageToClip": "🖼 S42 CutFlow Image → Clip",
    "S42CF_ClipToGIF": "🎁 S42 CutFlow Clip → GIF",
}
