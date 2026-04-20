"""
S42 CutFlow — Speed Ramp
=========================
Keyframe-based speed curves (CapCut-style). Constant, linear,
S-curve, and custom bezier speed ramping.

Python 3.12 | ComfyUI Portable | torch
"""

import torch
import numpy as np
import logging
from typing import Tuple

from .cf_utils import ensure_rgb, parse_keyframe_string, interpolate_keyframes, ALL_EASINGS

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Clip Ops"

SPEED_PRESETS = [
    "custom",
    "montage",        # fast-slow-fast
    "bullet_time",    # normal → very slow → normal
    "hero_moment",    # slow build → freeze → resume
    "whip_pan",       # very fast in middle
    "ramp_up",        # slow → fast
    "ramp_down",      # fast → slow
]


class S42CF_SpeedRamp:
    """CapCut-style speed ramping with keyframe curves."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to apply speed ramping to."}),
                "preset": (SPEED_PRESETS, {
                    "default": "custom",
                    "tooltip": "Speed curve preset. 'custom' uses the speed_curve keyframe string.\n"
                               "'montage' = fast-slow-fast for dynamic edits.\n"
                               "'bullet_time' = dramatic slow-motion in center.\n"
                               "'hero_moment' = slow build to freeze frame.\n"
                               "'whip_pan' = fast through middle.\n"
                               "'ramp_up'/'ramp_down' = gradual acceleration/deceleration."
                }),
                "speed_curve": ("STRING", {
                    "default": "0:1.0, 12:0.25:ease_in_out, 36:0.25, 48:1.0:ease_in_out",
                    "multiline": True,
                    "tooltip": "Custom speed curve as keyframe string.\nFormat: 'frame:speed_multiplier:easing'\n"
                               "Example: '0:1.0, 12:0.25:ease_in, 36:0.25, 48:1.0:ease_out'\n"
                               "Speed values: 0.1=very slow, 0.5=half speed, 1.0=normal, 2.0=double.\n"
                               "Only used when preset='custom'."
                }),
                "global_speed": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Global speed multiplier applied on top of the curve. 1.0 = no change."
                }),
                "interpolation": (["blend", "duplicate"], {
                    "default": "blend",
                    "tooltip": "'blend' = smooth linear interpolation between frames (recommended).\n'duplicate' = nearest frame, faster but may look jittery."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def _get_preset_curve(self, preset: str, n: int):
        mid = n // 2
        q1, q3 = n // 4, 3 * n // 4
        presets = {
            "montage":      [(0, 2.0, "ease_out"), (q1, 0.5, "ease_in_out"), (q3, 0.5, "ease_in_out"), (n-1, 2.0, "ease_in")],
            "bullet_time":  [(0, 1.0, "ease_in"), (q1, 0.15, "ease_in_out"), (q3, 0.15, "ease_in_out"), (n-1, 1.0, "ease_out")],
            "hero_moment":  [(0, 0.7, "ease_in"), (mid-4, 0.1, "ease_in_out"), (mid+4, 0.1, "ease_in_out"), (n-1, 1.0, "ease_out")],
            "whip_pan":     [(0, 0.8, "ease_in"), (q1, 3.0, "ease_in_out"), (q3, 3.0, "ease_in_out"), (n-1, 0.8, "ease_out")],
            "ramp_up":      [(0, 0.3, "ease_in_cubic"), (n-1, 2.5, "linear")],
            "ramp_down":    [(0, 2.5, "ease_out_cubic"), (n-1, 0.3, "linear")],
        }
        return presets.get(preset, [(0, 1.0, "linear"), (n-1, 1.0, "linear")])

    def process(self, clip, preset, speed_curve, global_speed, interpolation):
        clip = ensure_rgb(clip)
        n = clip.shape[0]

        if preset == "custom":
            kf = parse_keyframe_string(speed_curve)
            if not kf:
                kf = [(0, 1.0, "linear"), (n - 1, 1.0, "linear")]
        else:
            kf = self._get_preset_curve(preset, n)

        per_frame_speed = []
        for f in range(n):
            s = interpolate_keyframes(kf, f) * global_speed
            per_frame_speed.append(max(0.01, s))

        src_times = [0.0]
        for f in range(n - 1):
            dt = 1.0 / per_frame_speed[f]
            src_times.append(src_times[-1] + dt)

        total_output_time = src_times[-1]
        n_out = max(1, round(total_output_time))

        out_frames = []
        for i in range(n_out):
            t = i * total_output_time / max(n_out - 1, 1) if n_out > 1 else 0
            lo = 0
            for j in range(len(src_times) - 1):
                if src_times[j] <= t <= src_times[j + 1]:
                    lo = j
                    break
                if t > src_times[j]:
                    lo = j
            hi = min(lo + 1, n - 1)
            if lo == hi or src_times[hi] == src_times[lo]:
                out_frames.append(clip[lo])
            else:
                frac = (t - src_times[lo]) / (src_times[hi] - src_times[lo])
                frac = max(0.0, min(1.0, frac))
                if interpolation == "blend":
                    out_frames.append(clip[lo] * (1 - frac) + clip[hi] * frac)
                else:
                    out_frames.append(clip[lo] if frac < 0.5 else clip[hi])

        result = torch.stack(out_frames)
        avg_speed = sum(per_frame_speed) / len(per_frame_speed)
        info = f"SpeedRamp({preset}): {n}→{result.shape[0]}f | avg speed={avg_speed:.2f}x | global={global_speed}x"
        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_SpeedRamp": S42CF_SpeedRamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_SpeedRamp": "⏩ S42 CutFlow Speed Ramp",
}
