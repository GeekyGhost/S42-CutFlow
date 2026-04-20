"""
S42 CutFlow — Temporal Effects
================================
Time-domain effects: echo/ghosting, motion trails, time displacement,
glitch effects, and frame blending.

Python 3.12 | ComfyUI Portable | torch + numpy
"""

import torch
import numpy as np
import math
import logging

from .cf_utils import ensure_rgb, frames_to_np, np_to_frames

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Filters"

TEMPORAL_MODES = ["echo", "motion_trail", "time_displacement", "frame_average", "strobe"]


class S42CF_TemporalFX:
    """Temporal effects across multiple frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip for temporal effects."}),
                "mode": (TEMPORAL_MODES, {
                    "default": "echo",
                    "tooltip": "'echo' = semi-transparent copies of past frames layered on current.\n"
                               "'motion_trail' = accumulating trail from moving objects.\n"
                               "'time_displacement' = vertical pixel rows sample different time offsets.\n"
                               "'frame_average' = average N adjacent frames (smooth/ghostly).\n"
                               "'strobe' = alternating frame hold for strobe/flash effect."
                }),
                "window": ("INT", {
                    "default": 5, "min": 2, "max": 30, "step": 1,
                    "tooltip": "Number of frames in the temporal window. Larger = more ghosting/trail/blur."
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Effect strength. 0 = passthrough, 1 = maximum effect."
                }),
                "decay": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 0.99, "step": 0.05,
                    "tooltip": "How quickly past frames fade. Higher = longer persistence. Used by echo and motion_trail."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, window, intensity, decay):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        results = []

        if mode == "echo":
            for i in range(n):
                frame = clip[i].clone().float()
                for j in range(1, min(window, i + 1)):
                    weight = intensity * (decay ** j)
                    frame = frame + clip[i - j] * weight
                results.append(frame.clamp(0, 1))

        elif mode == "motion_trail":
            accumulator = clip[0].clone().float()
            for i in range(n):
                accumulator = accumulator * decay + clip[i] * (1 - decay)
                blended = clip[i] * (1 - intensity) + accumulator * intensity
                results.append(blended.clamp(0, 1))

        elif mode == "time_displacement":
            h = clip.shape[1]
            for i in range(n):
                frame = clip[i].clone()
                for row in range(h):
                    offset = int((row / h) * window * intensity)
                    src_idx = max(0, min(i - offset, n - 1))
                    frame[row] = clip[src_idx][row]
                results.append(frame)

        elif mode == "frame_average":
            half = window // 2
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                avg = clip[start:end].float().mean(dim=0)
                blended = clip[i] * (1 - intensity) + avg * intensity
                results.append(blended.clamp(0, 1))

        elif mode == "strobe":
            hold_len = max(1, window)
            for i in range(n):
                held_idx = (i // hold_len) * hold_len
                held_idx = min(held_idx, n - 1)
                blended = clip[i] * (1 - intensity) + clip[held_idx] * intensity
                results.append(blended.clamp(0, 1))

        result = torch.stack(results)
        info = f"TemporalFX({mode}): window={window}, intensity={intensity}, decay={decay}"
        return (result, int(result.shape[0]), info)


GLITCH_MODES = ["datamosh", "scan_lines", "pixel_sort", "rgb_split", "block_corrupt"]


class S42CF_GlitchFX:
    """Digital glitch effects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to glitch."}),
                "mode": (GLITCH_MODES, {
                    "default": "rgb_split",
                    "tooltip": "'datamosh' = frame bleed/smear between frames.\n"
                               "'scan_lines' = horizontal line artifacts.\n"
                               "'pixel_sort' = sort pixels by brightness in bands.\n"
                               "'rgb_split' = offset color channels randomly.\n"
                               "'block_corrupt' = random block displacement."
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Glitch intensity. 0 = subtle artifacts, 1 = heavy corruption."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999,
                    "tooltip": "Random seed for reproducible glitch patterns."}),
                "animated": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "'yes' = different glitch per frame. 'no' = static pattern."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, intensity, seed, animated):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape
        results = []

        for i in range(n):
            frame_seed = seed + i * 137 if animated == "yes" else seed
            rng = np.random.default_rng(frame_seed)
            frame = clip[i].cpu().numpy().copy()

            if mode == "rgb_split":
                amp = int(intensity * w * 0.08) + 1
                shift_r = rng.integers(-amp, amp)
                shift_b = rng.integers(-amp, amp)
                frame[:, :, 0] = np.roll(frame[:, :, 0], shift_r, axis=1)
                frame[:, :, 2] = np.roll(frame[:, :, 2], shift_b, axis=1)

            elif mode == "scan_lines":
                n_lines = max(1, int(intensity * 20))
                for _ in range(n_lines):
                    y = rng.integers(0, h)
                    thickness = rng.integers(1, max(2, int(intensity * 5)))
                    y2 = min(y + thickness, h)
                    shift = rng.integers(-int(w * 0.2), int(w * 0.2))
                    frame[y:y2] = np.roll(frame[y:y2], shift, axis=1)

            elif mode == "pixel_sort":
                n_bands = max(1, int(intensity * 10))
                for _ in range(n_bands):
                    y = rng.integers(0, h)
                    band_h = rng.integers(2, max(3, int(h * intensity * 0.1)))
                    y2 = min(y + band_h, h)
                    for row in range(y, y2):
                        lum = 0.299 * frame[row, :, 0] + 0.587 * frame[row, :, 1] + 0.114 * frame[row, :, 2]
                        indices = np.argsort(lum)
                        frame[row] = frame[row][indices]

            elif mode == "datamosh":
                if i > 0:
                    prev = clip[i - 1].cpu().numpy()
                    n_blocks = max(1, int(intensity * 15))
                    bh = max(8, h // 8)
                    bw = max(8, w // 8)
                    for _ in range(n_blocks):
                        by = rng.integers(0, max(1, h - bh))
                        bx = rng.integers(0, max(1, w - bw))
                        frame[by:by + bh, bx:bx + bw] = prev[by:by + bh, bx:bx + bw]

            elif mode == "block_corrupt":
                n_blocks = max(1, int(intensity * 12))
                bh = max(8, int(h * 0.05 * intensity))
                bw = max(8, int(w * 0.1 * intensity))
                for _ in range(n_blocks):
                    sy = rng.integers(0, max(1, h - bh))
                    sx = rng.integers(0, max(1, w - bw))
                    dy = rng.integers(0, max(1, h - bh))
                    dx = rng.integers(0, max(1, w - bw))
                    frame[dy:dy + bh, dx:dx + bw] = frame[sy:sy + bh, sx:sx + bw]

            results.append(torch.from_numpy(frame.clip(0, 1).astype(np.float32)))

        result = torch.stack(results)
        info = f"GlitchFX({mode}): intensity={intensity}, seed={seed}"
        return (result, int(result.shape[0]), info)


class S42CF_FrameBlend:
    """Blend N adjacent frames for motion blur or smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to apply frame blending to."}),
                "window": ("INT", {
                    "default": 3, "min": 2, "max": 15, "step": 1,
                    "tooltip": "Number of frames to blend together. 2-3 = subtle smoothing. 5+ = strong motion blur."
                }),
                "weights": (["uniform", "center_weighted", "exponential_decay"], {
                    "default": "center_weighted",
                    "tooltip": "'uniform' = equal weight for all frames in window.\n"
                               "'center_weighted' = current frame has highest weight.\n"
                               "'exponential_decay' = past frames decay exponentially."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Blend between original (0) and fully blended (1)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, window, weights, strength):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        half = window // 2

        if weights == "uniform":
            w = np.ones(window, dtype=np.float32) / window
        elif weights == "center_weighted":
            w = np.array([1.0 / (1.0 + abs(i - half)) for i in range(window)], dtype=np.float32)
            w /= w.sum()
        else:
            w = np.array([0.5 ** abs(i - half) for i in range(window)], dtype=np.float32)
            w /= w.sum()

        results = []
        for i in range(n):
            blended = torch.zeros_like(clip[0]).float()
            for j in range(window):
                src = max(0, min(i + j - half, n - 1))
                blended += clip[src] * w[j]
            mixed = clip[i] * (1 - strength) + blended * strength
            results.append(mixed.clamp(0, 1))

        result = torch.stack(results)
        info = f"FrameBlend: window={window}, weights={weights}, strength={strength}"
        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_TemporalFX": S42CF_TemporalFX,
    "S42CF_GlitchFX": S42CF_GlitchFX,
    "S42CF_FrameBlend": S42CF_FrameBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_TemporalFX": "⏳ S42 CutFlow Temporal FX",
    "S42CF_GlitchFX": "💥 S42 CutFlow Glitch FX",
    "S42CF_FrameBlend": "🔄 S42 CutFlow Frame Blend",
}
