"""
S42 CutFlow — Filters & Color Effects
=======================================
Per-frame visual filters: sharpen, blur, denoise, vignette, grain,
chromatic aberration, lens distortion, style presets.

Python 3.12 | ComfyUI Portable | torch + numpy + PIL + optional OpenCV
"""

import torch
import numpy as np
import math
import logging
from typing import Optional

from .cf_utils import (
    ensure_rgb, frames_to_np, np_to_frames, frame_to_pil, pil_to_frame,
    PIL_AVAILABLE, CV2_AVAILABLE, SCIPY_AVAILABLE, process_frames_chunked
)

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Filters"

if CV2_AVAILABLE:
    import cv2
if PIL_AVAILABLE:
    from PIL import ImageFilter
if SCIPY_AVAILABLE:
    from scipy.ndimage import gaussian_filter, uniform_filter, median_filter


FILTER_MODES = [
    "sharpen", "unsharp_mask",
    "gaussian_blur", "box_blur", "motion_blur",
    "median_denoise", "bilateral_denoise",
]


class S42CF_FilterPack:
    """Multi-mode filter node with per-frame batch processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to filter."}),
                "mode": (FILTER_MODES, {
                    "default": "sharpen",
                    "tooltip": "Filter type:\n"
                               "'sharpen' = enhance edges and detail.\n"
                               "'unsharp_mask' = professional sharpening with radius/amount control.\n"
                               "'gaussian_blur' = smooth Gaussian blur.\n"
                               "'box_blur' = fast uniform blur.\n"
                               "'motion_blur' = directional blur simulating camera motion.\n"
                               "'median_denoise' = noise removal preserving edges.\n"
                               "'bilateral_denoise' = edge-preserving smoothing (best quality, slower)."
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Filter strength. Meaning varies by mode:\n"
                               "sharpen/unsharp: amount multiplier.\n"
                               "blur modes: kernel radius in pixels.\n"
                               "denoise modes: filter diameter/strength."
                }),
                "radius": ("INT", {
                    "default": 3, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Kernel radius (must be odd). Larger = stronger effect but slower. Used by blur, denoise, unsharp."
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Direction angle in degrees for motion_blur mode. 0=horizontal, 90=vertical."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def _apply_filter(self, frame_np, mode, intensity, radius, angle):
        """Apply filter to single uint8 [H,W,C] frame."""
        r = max(1, radius if radius % 2 == 1 else radius + 1)

        if mode == "sharpen":
            if CV2_AVAILABLE:
                blurred = cv2.GaussianBlur(frame_np, (r, r), 0)
                return cv2.addWeighted(frame_np, 1.0 + intensity, blurred, -intensity, 0)
            else:
                f32 = frame_np.astype(np.float32) / 255.0
                kernel = np.array([[-1, -1, -1], [-1, 9 + intensity, -1], [-1, -1, -1]]) / (1 + intensity)
                from scipy.ndimage import convolve
                for c in range(3):
                    f32[:, :, c] = convolve(f32[:, :, c], kernel)
                return (f32.clip(0, 1) * 255).astype(np.uint8)

        elif mode == "unsharp_mask":
            if CV2_AVAILABLE:
                blurred = cv2.GaussianBlur(frame_np, (r, r), 0)
                return cv2.addWeighted(frame_np, 1.0 + intensity, blurred, -intensity, 0)
            elif PIL_AVAILABLE:
                from PIL import Image
                img = Image.fromarray(frame_np)
                sharp = img.filter(ImageFilter.UnsharpMask(radius=r, percent=int(intensity * 100), threshold=2))
                return np.array(sharp)
            return frame_np

        elif mode == "gaussian_blur":
            sigma = intensity * 2
            if CV2_AVAILABLE:
                return cv2.GaussianBlur(frame_np, (r, r), sigma)
            elif SCIPY_AVAILABLE:
                return gaussian_filter(frame_np, sigma=[sigma, sigma, 0]).astype(np.uint8)
            return frame_np

        elif mode == "box_blur":
            if CV2_AVAILABLE:
                return cv2.blur(frame_np, (r, r))
            elif SCIPY_AVAILABLE:
                return uniform_filter(frame_np, size=[r, r, 1]).astype(np.uint8)
            return frame_np

        elif mode == "motion_blur":
            size = max(3, int(intensity * 5))
            if size % 2 == 0:
                size += 1
            kernel = np.zeros((size, size), dtype=np.float32)
            rad = math.radians(angle)
            cx, cy = size // 2, size // 2
            for i in range(size):
                x = int(cx + (i - cx) * math.cos(rad))
                y = int(cy + (i - cx) * math.sin(rad))
                if 0 <= x < size and 0 <= y < size:
                    kernel[y, x] = 1.0
            kernel /= max(kernel.sum(), 1)
            if CV2_AVAILABLE:
                return cv2.filter2D(frame_np, -1, kernel)
            return frame_np

        elif mode == "median_denoise":
            if CV2_AVAILABLE:
                return cv2.medianBlur(frame_np, r)
            elif SCIPY_AVAILABLE:
                result = np.stack([median_filter(frame_np[:, :, c], size=r) for c in range(3)], axis=-1)
                return result.astype(np.uint8)
            return frame_np

        elif mode == "bilateral_denoise":
            d = max(3, int(intensity * 3))
            sigma_color = intensity * 30
            sigma_space = intensity * 30
            if CV2_AVAILABLE:
                return cv2.bilateralFilter(frame_np, d, sigma_color, sigma_space)
            elif SCIPY_AVAILABLE:
                return gaussian_filter(frame_np, sigma=[intensity, intensity, 0]).astype(np.uint8)
            return frame_np

        return frame_np

    def process(self, clip, mode, intensity, radius, angle):
        clip = ensure_rgb(clip)
        frames_np = frames_to_np(clip)
        result = np.stack([self._apply_filter(f, mode, intensity, radius, angle) for f in frames_np])
        out = np_to_frames(result)
        info = f"FilterPack({mode}): intensity={intensity}, radius={radius} | {out.shape[0]}f"
        return (out, int(out.shape[0]), info)


class S42CF_Vignette:
    """Configurable vignette effect."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to apply vignette to."}),
                "intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Vignette darkness strength. 0 = none, 1.0 = strong, 2.0 = extreme."
                }),
                "softness": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How gradually the vignette fades in. 0.1 = sharp edge, 1.0 = very soft."
                }),
                "shape": (["circle", "oval", "rectangle"], {
                    "default": "circle",
                    "tooltip": "Vignette shape. 'oval' stretches to match frame aspect ratio."
                }),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Horizontal offset of vignette center. -1=left, 0=center, 1=right."}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Vertical offset of vignette center. -1=top, 0=center, 1=bottom."}),
                "vignette_color": ("STRING", {"default": "#000000",
                    "tooltip": "Vignette color (hex). Default #000000 = black. Click swatch to pick."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, intensity, softness, shape, offset_x, offset_y, vignette_color):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        Y, X = np.ogrid[:h, :w]
        cy = h / 2 + offset_y * h / 2
        cx = w / 2 + offset_x * w / 2

        if shape == "circle":
            r_max = min(h, w) / 2
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r_max
        elif shape == "oval":
            dist = np.sqrt(((X - cx) / (w / 2)) ** 2 + ((Y - cy) / (h / 2)) ** 2)
        else:
            dist_x = np.abs(X - cx) / (w / 2)
            dist_y = np.abs(Y - cy) / (h / 2)
            dist = np.maximum(dist_x, dist_y)

        inner = 1.0 - softness
        mask = np.clip((dist - inner) / max(softness, 0.01), 0.0, 1.0)
        mask = mask ** 1.5 * intensity
        mask = mask.clip(0, 1).astype(np.float32)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
        from .cf_utils import hex_to_rgb
        vr, vg, vb = hex_to_rgb(vignette_color)
        color_t = torch.tensor([vr, vg, vb], dtype=torch.float32)

        result = clip * (1 - mask_t) + color_t * mask_t
        result = result.clamp(0, 1)

        info = f"Vignette: intensity={intensity}, shape={shape}, softness={softness}"
        return (result, int(result.shape[0]), info)


class S42CF_FilmGrain:
    """Procedural film grain overlay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to add grain to."}),
                "intensity": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Grain intensity. 0.05 = subtle, 0.15 = standard film, 0.5 = heavy ISO noise."
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.5,
                    "tooltip": "Grain particle size. 1.0 = fine grain. 2-4 = coarser, more visible grain."
                }),
                "color_mode": (["mono", "color"], {
                    "default": "mono",
                    "tooltip": "'mono' = luminance-only grain (classic film). 'color' = per-channel noise (digital/chromatic)."
                }),
                "animated": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "'yes' = unique grain pattern per frame (natural). 'no' = static grain (same every frame)."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999, "tooltip": "Random seed for reproducible grain patterns."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, intensity, grain_size, color_mode, animated, seed):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        gh = max(1, int(h / grain_size))
        gw = max(1, int(w / grain_size))

        results = []
        for i in range(n):
            frame_seed = seed + i if animated == "yes" else seed
            rng = np.random.default_rng(frame_seed)

            if color_mode == "mono":
                noise_small = rng.standard_normal((gh, gw, 1)).astype(np.float32)
                noise_small = np.repeat(noise_small, 3, axis=-1)
            else:
                noise_small = rng.standard_normal((gh, gw, 3)).astype(np.float32)

            if grain_size > 1.0:
                noise = np.array(
                    PIL_AVAILABLE and __import__('PIL').Image.fromarray(
                        ((noise_small * 127 + 128).clip(0, 255)).astype(np.uint8)
                    ).resize((w, h), __import__('PIL').Image.Resampling.BILINEAR)
                ) if False else noise_small
                from PIL import Image as _Img
                noise_uint8 = ((noise_small * 127 + 128).clip(0, 255)).astype(np.uint8)
                noise_pil = _Img.fromarray(noise_uint8).resize((w, h), _Img.Resampling.BILINEAR)
                noise = (np.array(noise_pil).astype(np.float32) - 128) / 127.0
            else:
                noise = rng.standard_normal((h, w, 3 if color_mode == "color" else 1)).astype(np.float32)
                if color_mode == "mono":
                    noise = np.repeat(noise, 3, axis=-1)

            noise_t = torch.from_numpy(noise * intensity)
            frame = clip[i] + noise_t
            results.append(frame.clamp(0, 1))

        result = torch.stack(results)
        info = f"FilmGrain: intensity={intensity}, size={grain_size}, {color_mode}, animated={animated}"
        return (result, int(result.shape[0]), info)


class S42CF_ChromaticAberration:
    """RGB channel offset for chromatic aberration effect."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to apply chromatic aberration to."}),
                "strength": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Pixel offset amount. 1-3 = subtle lens fringe. 5-10 = visible. 15+ = extreme/stylistic."
                }),
                "mode": (["radial", "linear"], {
                    "default": "radial",
                    "tooltip": "'radial' = offset increases from center (realistic lens). 'linear' = uniform offset (artistic)."
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Direction angle for linear mode, or rotation of radial pattern. Degrees."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, strength, mode, angle):
        clip = ensure_rgb(clip)
        if strength < 0.1:
            return (clip, int(clip.shape[0]), "ChromaticAberration: off (strength<0.1)")

        n, h, w, c = clip.shape
        rad = math.radians(angle)
        dx_r = strength * math.cos(rad)
        dy_r = strength * math.sin(rad)
        dx_b = -dx_r
        dy_b = -dy_r

        results = []
        for i in range(n):
            frame = clip[i].cpu().numpy()
            r_ch = frame[:, :, 0]
            g_ch = frame[:, :, 1]
            b_ch = frame[:, :, 2]

            if mode == "radial":
                Y, X = np.mgrid[:h, :w].astype(np.float32)
                cx, cy = w / 2, h / 2
                dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max(cx, cy)
                map_x_r = X + dx_r * dist
                map_y_r = Y + dy_r * dist
                map_x_b = X + dx_b * dist
                map_y_b = Y + dy_b * dist
            else:
                Y, X = np.mgrid[:h, :w].astype(np.float32)
                map_x_r = X + dx_r
                map_y_r = Y + dy_r
                map_x_b = X + dx_b
                map_y_b = Y + dy_b

            if CV2_AVAILABLE:
                r_shifted = cv2.remap(r_ch, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                b_shifted = cv2.remap(b_ch, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            else:
                r_shifted = np.roll(r_ch, int(dx_r), axis=1)
                r_shifted = np.roll(r_shifted, int(dy_r), axis=0)
                b_shifted = np.roll(b_ch, int(dx_b), axis=1)
                b_shifted = np.roll(b_shifted, int(dy_b), axis=0)

            out = np.stack([r_shifted, g_ch, b_shifted], axis=-1)
            results.append(torch.from_numpy(out.clip(0, 1)))

        result = torch.stack(results)
        info = f"ChromaticAberration: strength={strength}, mode={mode}, angle={angle}"
        return (result, int(result.shape[0]), info)


class S42CF_LensDistortion:
    """Barrel/pincushion distortion and tilt-shift blur."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip."}),
                "mode": (["barrel", "pincushion", "tilt_shift"], {
                    "default": "barrel",
                    "tooltip": "'barrel' = edges curve outward (wide-angle lens).\n'pincushion' = edges curve inward (telephoto).\n'tilt_shift' = miniature/diorama blur effect."
                }),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Distortion amount. 0.1-0.3 = subtle. 0.5+ = strong. For tilt_shift: blur radius."
                }),
                "focus_position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Tilt-shift focus band position. 0=top, 0.5=center, 1=bottom. Only used in tilt_shift mode."
                }),
                "focus_width": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 0.8, "step": 0.05,
                    "tooltip": "Width of the in-focus band for tilt-shift. Smaller = narrower sharp zone."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, strength, focus_position, focus_width):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        if mode == "tilt_shift":
            focus_y = int(focus_position * h)
            half_band = int(focus_width * h / 2)
            blur_radius = max(1, int(strength * 20))
            if blur_radius % 2 == 0:
                blur_radius += 1

            Y = np.arange(h, dtype=np.float32)
            dist = np.abs(Y - focus_y) - half_band
            dist = np.clip(dist, 0, h) / max(h - half_band, 1)
            blur_map = dist ** 1.5

            results = []
            for i in range(n):
                frame_np = frames_to_np(clip[i:i+1])[0]
                if CV2_AVAILABLE:
                    blurred = cv2.GaussianBlur(frame_np, (blur_radius, blur_radius), 0)
                else:
                    blurred = frame_np
                mask = blur_map[:, np.newaxis, np.newaxis]
                out = (frame_np.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask)
                results.append(out.clip(0, 255).astype(np.uint8))
            result = np_to_frames(np.stack(results))
        elif CV2_AVAILABLE:
            results = []
            for i in range(n):
                frame_np = frames_to_np(clip[i:i+1])[0]
                cy, cx = h // 2, w // 2
                Y, X = np.mgrid[:h, :w].astype(np.float32)
                X_norm = (X - cx) / cx
                Y_norm = (Y - cy) / cy
                r = np.sqrt(X_norm ** 2 + Y_norm ** 2)
                if mode == "barrel":
                    r_distorted = r * (1 + strength * r ** 2)
                else:
                    r_distorted = r * (1 - strength * r ** 2)
                scale = np.where(r > 0, r_distorted / (r + 1e-8), 1.0)
                map_x = (cx + X_norm * scale * cx).astype(np.float32)
                map_y = (cy + Y_norm * scale * cy).astype(np.float32)
                out = cv2.remap(frame_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                results.append(out)
            result = np_to_frames(np.stack(results))
        else:
            result = clip
            logger.warning("LensDistortion barrel/pincushion requires opencv-python")

        info = f"LensDistortion({mode}): strength={strength}"
        return (result, int(result.shape[0]), info)


STYLE_PRESETS = [
    "cinematic_teal_orange", "vintage", "noir", "polaroid",
    "cyberpunk", "pastel", "bleach_bypass", "cross_process",
]


class S42CF_StyleTransfer:
    """LUT-free style presets using pure numpy color math."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to stylize."}),
                "style": (STYLE_PRESETS, {
                    "default": "cinematic_teal_orange",
                    "tooltip": "Color style preset:\n"
                               "'cinematic_teal_orange' = Hollywood blockbuster look.\n"
                               "'vintage' = faded warm tones, lifted blacks.\n"
                               "'noir' = high-contrast desaturated.\n"
                               "'polaroid' = warm shadows, cool highlights.\n"
                               "'cyberpunk' = neon-shifted, high saturation.\n"
                               "'pastel' = soft, desaturated pastels.\n"
                               "'bleach_bypass' = silver-retention film look.\n"
                               "'cross_process' = shifted color channels, retro."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and fully styled (1)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def _apply_style(self, frame, style):
        f = frame.astype(np.float64)
        r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        if style == "cinematic_teal_orange":
            shadows = (1 - lum)
            highlights = lum
            r2 = r + highlights * 0.12 + shadows * (-0.05)
            g2 = g + highlights * 0.02 + shadows * 0.04
            b2 = b + highlights * (-0.08) + shadows * 0.12
            sat = 1.15
        elif style == "vintage":
            r2 = r * 1.08 + 0.04
            g2 = g * 0.95 + 0.02
            b2 = b * 0.85 + 0.03
            r2 = r2 * 0.9 + 0.05  # lifted blacks
            g2 = g2 * 0.9 + 0.04
            b2 = b2 * 0.9 + 0.03
            sat = 0.8
        elif style == "noir":
            grey = lum
            r2 = grey * 1.05 + 0.02
            g2 = grey * 1.0
            b2 = grey * 0.95
            sat = 0.15
        elif style == "polaroid":
            r2 = r * 1.05 + 0.03
            g2 = g * 1.02
            b2 = b * 0.92 + 0.05
            sat = 0.9
        elif style == "cyberpunk":
            r2 = r * 0.9 + b * 0.15
            g2 = g * 1.1
            b2 = b * 1.15 + r * 0.05
            sat = 1.4
        elif style == "pastel":
            r2 = r * 0.7 + 0.2
            g2 = g * 0.7 + 0.2
            b2 = b * 0.7 + 0.2
            sat = 0.6
        elif style == "bleach_bypass":
            grey = lum
            r2 = (r + grey) / 2 * 1.2
            g2 = (g + grey) / 2 * 1.1
            b2 = (b + grey) / 2 * 1.0
            sat = 0.5
        elif style == "cross_process":
            r2 = np.clip(r * 1.2 - 0.05, 0, 1)
            g2 = np.clip(g ** 0.85, 0, 1)
            b2 = np.clip(b * 0.8 + g * 0.2, 0, 1)
            sat = 1.2
        else:
            return frame

        result = np.stack([r2, g2, b2], axis=-1)
        lum2 = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        lum2 = lum2[:, :, np.newaxis]
        result = lum2 + (result - lum2) * sat
        return result.clip(0, 1).astype(np.float32)

    def process(self, clip, style, strength):
        clip = ensure_rgb(clip)
        results = []
        for i in range(clip.shape[0]):
            orig = clip[i].cpu().numpy()
            styled = self._apply_style(orig, style)
            blended = orig * (1 - strength) + styled * strength
            results.append(torch.from_numpy(blended.clip(0, 1).astype(np.float32)))
        result = torch.stack(results)
        info = f"StyleTransfer({style}): strength={strength}"
        return (result, int(result.shape[0]), info)


class S42CF_ColorAdjust:
    """Fundamental brightness, contrast, saturation, and hue adjustment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to color-adjust."}),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Brightness offset. -1 = black, 0 = no change, +1 = white. Typical range: -0.2 to +0.2."
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Contrast multiplier. 0 = flat gray, 1.0 = no change, 1.5 = punchy, 2.0+ = extreme."
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Color saturation. 0 = grayscale, 1.0 = no change, 1.5 = vivid, 2.0+ = oversaturated."
                }),
                "hue_shift": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Hue rotation in degrees. 0 = no change. 30 = warm shift. -30 = cool shift. 180 = complementary."
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Gamma correction. <1.0 = brighten midtones (lift shadows). >1.0 = darken midtones. 1.0 = no change."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Color temperature shift. Negative = cooler/blue. Positive = warmer/orange. 0 = no change."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, brightness, contrast, saturation, hue_shift, gamma, temperature):
        clip = ensure_rgb(clip)
        result = clip.clone().float()

        # Brightness
        if abs(brightness) > 0.001:
            result = result + brightness

        # Contrast (around midpoint 0.5)
        if abs(contrast - 1.0) > 0.001:
            result = (result - 0.5) * contrast + 0.5

        # Gamma
        if abs(gamma - 1.0) > 0.001:
            result = result.clamp(0.001, 1.0).pow(1.0 / gamma)

        # Saturation
        if abs(saturation - 1.0) > 0.001:
            lum = 0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]
            lum = lum.unsqueeze(-1)
            result = lum + (result - lum) * saturation

        # Temperature (warm = +R -B, cool = -R +B)
        if abs(temperature) > 0.001:
            result[..., 0] = result[..., 0] + temperature * 0.1   # Red
            result[..., 2] = result[..., 2] - temperature * 0.1   # Blue

        # Hue shift (convert to HSV-like, rotate, convert back)
        if abs(hue_shift) > 0.5:
            r, g, b = result[..., 0], result[..., 1], result[..., 2]
            cmax = torch.max(result, dim=-1).values
            cmin = torch.min(result, dim=-1).values
            delta = cmax - cmin + 1e-8

            # Compute hue
            hue = torch.zeros_like(cmax)
            mask_r = (cmax == r) & (delta > 1e-7)
            mask_g = (cmax == g) & (delta > 1e-7) & ~mask_r
            mask_b = ~mask_r & ~mask_g & (delta > 1e-7)
            hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
            hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
            hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

            # Shift hue
            hue = (hue + hue_shift) % 360.0

            # HSV to RGB
            sat_v = delta / (cmax + 1e-8)
            val = cmax
            c = val * sat_v
            x = c * (1 - torch.abs((hue / 60.0) % 2 - 1))
            m = val - c

            h_sector = (hue / 60.0).long() % 6
            r2 = torch.zeros_like(hue)
            g2 = torch.zeros_like(hue)
            b2 = torch.zeros_like(hue)

            for sec, rv, gv, bv in [(0, 'c', 'x', '0'), (1, 'x', 'c', '0'),
                                      (2, '0', 'c', 'x'), (3, '0', 'x', 'c'),
                                      (4, 'x', '0', 'c'), (5, 'c', '0', 'x')]:
                mask = h_sector == sec
                vals = {'c': c, 'x': x, '0': torch.zeros_like(c)}
                r2[mask] = vals[rv][mask]
                g2[mask] = vals[gv][mask]
                b2[mask] = vals[bv][mask]

            result = torch.stack([r2 + m, g2 + m, b2 + m], dim=-1)

        result = result.clamp(0, 1)
        changes = []
        if abs(brightness) > 0.001: changes.append(f"bright={brightness:+.2f}")
        if abs(contrast - 1.0) > 0.001: changes.append(f"contrast={contrast:.2f}")
        if abs(saturation - 1.0) > 0.001: changes.append(f"sat={saturation:.2f}")
        if abs(hue_shift) > 0.5: changes.append(f"hue={hue_shift:+.0f}")
        if abs(gamma - 1.0) > 0.001: changes.append(f"gamma={gamma:.2f}")
        if abs(temperature) > 0.001: changes.append(f"temp={temperature:+.2f}")
        info = f"ColorAdjust: {', '.join(changes) if changes else 'no changes'}"
        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_FilterPack": S42CF_FilterPack,
    "S42CF_Vignette": S42CF_Vignette,
    "S42CF_FilmGrain": S42CF_FilmGrain,
    "S42CF_ChromaticAberration": S42CF_ChromaticAberration,
    "S42CF_LensDistortion": S42CF_LensDistortion,
    "S42CF_StyleTransfer": S42CF_StyleTransfer,
    "S42CF_ColorAdjust": S42CF_ColorAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_FilterPack": "🔧 S42 CutFlow Filter Pack",
    "S42CF_Vignette": "🔲 S42 CutFlow Vignette",
    "S42CF_FilmGrain": "🎞 S42 CutFlow Film Grain",
    "S42CF_ChromaticAberration": "🌈 S42 CutFlow Chromatic Aberration",
    "S42CF_LensDistortion": "🔍 S42 CutFlow Lens Distortion",
    "S42CF_StyleTransfer": "🎨 S42 CutFlow Style Transfer",
    "S42CF_ColorAdjust": "☀ S42 CutFlow Color Adjust",
}
