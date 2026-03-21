"""
S42 CutFlow — Video Transitions (Expanded)
=============================================
20 CapCut-style transitions between two video clip batches.
Ported from proven S42P Transition node with correct frame handling.

CRITICAL: Motion transitions (push/wipe/zoom/spin/iris/slide) use
STABLE reference frames (last of A, first of B) to prevent frame jitter.
Blending transitions (dissolve/fade/glitch) use corresponding overlap frames.

Python 3.12 | ComfyUI Portable | PIL + torch + numpy
"""

import torch
import numpy as np
import math
import logging
from typing import Tuple
from PIL import Image

from .cf_utils import ensure_rgb, match_resolution, apply_easing, ALL_EASINGS

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Expanded"

TRANSITION_TYPES = [
    "cut", "dissolve", "fade_black", "fade_white",
    "push_left", "push_right", "push_up", "push_down",
    "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "zoom_in", "zoom_out", "glitch",
    "spin_cw", "spin_ccw", "iris_in",
    "slide_up", "slide_down",
]

# Transitions that blend BOTH clips per-frame (use a_over[i] + b_over[i])
BLENDING_TRANSITIONS = {"dissolve", "fade_black", "fade_white", "glitch"}


# ── Per-transition frame generators ──────────────────────────────

def _t_dissolve(a, b, t):
    return (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)

def _t_fade_color(a, b, t, color):
    c = np.array(color, dtype=np.float32)
    if t < 0.5:
        p = t / 0.5
        return (a * (1.0 - p) + c * p).clip(0, 255).astype(np.uint8)
    else:
        p = (t - 0.5) / 0.5
        return (c * (1.0 - p) + b * p).clip(0, 255).astype(np.uint8)

def _t_push(a, b, t, direction):
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    if direction == "left":
        ox = int(w * t)
        if ox < w: out[:, :w-ox] = a[:, ox:]
        if ox > 0: out[:, w-ox:] = b[:, :ox]
    elif direction == "right":
        ox = int(w * t)
        if ox < w: out[:, ox:] = a[:, :w-ox]
        if ox > 0: out[:, :ox] = b[:, w-ox:]
    elif direction == "up":
        oy = int(h * t)
        if oy < h: out[:h-oy, :] = a[oy:, :]
        if oy > 0: out[h-oy:, :] = b[:oy, :]
    elif direction == "down":
        oy = int(h * t)
        if oy < h: out[oy:, :] = a[:h-oy, :]
        if oy > 0: out[:oy, :] = b[h-oy:, :]
    return out

def _t_wipe(a, b, t, direction):
    h, w = a.shape[:2]
    out = a.copy()
    if direction == "left":
        cut = int(w * t); out[:, :cut] = b[:, :cut]
    elif direction == "right":
        cut = int(w * (1.0 - t)); out[:, cut:] = b[:, cut:]
    elif direction == "up":
        cut = int(h * t); out[:cut, :] = b[:cut, :]
    elif direction == "down":
        cut = int(h * (1.0 - t)); out[cut:, :] = b[cut:, :]
    return out

def _t_zoom_in(a, b, t):
    h, w = a.shape[:2]
    scale = 0.1 + t * 0.9
    bh, bw = max(1, int(h * scale)), max(1, int(w * scale))
    b_s = np.array(Image.fromarray(b).resize((bw, bh), Image.LANCZOS))
    out = a.copy()
    py, px = (h - bh) // 2, (w - bw) // 2
    sy0, dy0 = max(0, -py), max(0, py)
    sx0, dx0 = max(0, -px), max(0, px)
    ph, pw = min(bh - sy0, h - dy0), min(bw - sx0, w - dx0)
    if ph > 0 and pw > 0:
        out[dy0:dy0+ph, dx0:dx0+pw] = b_s[sy0:sy0+ph, sx0:sx0+pw]
    return out

def _t_zoom_out(a, b, t):
    h, w = a.shape[:2]
    scale = 1.0 - t
    ah, aw = max(1, int(h * scale)), max(1, int(w * scale))
    a_s = np.array(Image.fromarray(a).resize((aw, ah), Image.LANCZOS))
    out = b.copy()
    py, px = (h - ah) // 2, (w - aw) // 2
    sy0, dy0 = max(0, -py), max(0, py)
    sx0, dx0 = max(0, -px), max(0, px)
    ph, pw = min(ah - sy0, h - dy0), min(aw - sx0, w - dx0)
    if ph > 0 and pw > 0:
        out[dy0:dy0+ph, dx0:dx0+pw] = a_s[sy0:sy0+ph, sx0:sx0+pw]
    return out

def _t_glitch(a, b, t, seed=42):
    rng = np.random.default_rng(seed + int(t * 1000))
    base = (a * (1.0 - t) + b * t).astype(np.float32)
    h, w = a.shape[:2]
    amp = int(t * w * 0.12) + 1
    out = base.copy()
    out[:, :, 0] = np.roll(base[:, :, 0], rng.integers(-amp, amp), axis=1)
    out[:, :, 2] = np.roll(base[:, :, 2], rng.integers(-amp, amp), axis=1)
    for _ in range(max(1, int(t * 8))):
        y = rng.integers(0, h)
        out[y, :] = rng.integers(0, 255, (w, 3))
    return out.clip(0, 255).astype(np.uint8)

def _t_spin(a, b, t, clockwise):
    direction = 1 if clockwise else -1
    if t < 0.5:
        angle, src = direction * t * 180.0, a
    else:
        angle, src = direction * (t - 0.5) * 180.0, b
    rot = Image.fromarray(src).rotate(-angle, resample=Image.BICUBIC, expand=False)
    return np.array(rot)

def _t_iris(a, b, t):
    h, w = a.shape[:2]
    out = a.copy()
    cy, cx = h // 2, w // 2
    r = t * math.hypot(cx, cy)
    Y, X = np.ogrid[:h, :w]
    mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= r
    out[mask] = b[mask]
    return out

def _t_slide(a, b, t, direction):
    h, w = a.shape[:2]
    out = a.copy()
    if direction == "up":
        sh = int(h * t)
        if sh > 0: out[h-sh:, :] = b[:sh, :]
    elif direction == "down":
        sh = int(h * t)
        if sh > 0: out[:sh, :] = b[h-sh:, :]
    return out


def _apply_transition_frame(a_np, b_np, t, transition, seed=42):
    """Dispatch to the correct transition function."""
    t = float(np.clip(t, 0.0, 1.0))
    dispatch = {
        "dissolve": lambda: _t_dissolve(a_np, b_np, t),
        "fade_black": lambda: _t_fade_color(a_np, b_np, t, (0, 0, 0)),
        "fade_white": lambda: _t_fade_color(a_np, b_np, t, (255, 255, 255)),
        "push_left": lambda: _t_push(a_np, b_np, t, "left"),
        "push_right": lambda: _t_push(a_np, b_np, t, "right"),
        "push_up": lambda: _t_push(a_np, b_np, t, "up"),
        "push_down": lambda: _t_push(a_np, b_np, t, "down"),
        "wipe_left": lambda: _t_wipe(a_np, b_np, t, "left"),
        "wipe_right": lambda: _t_wipe(a_np, b_np, t, "right"),
        "wipe_up": lambda: _t_wipe(a_np, b_np, t, "up"),
        "wipe_down": lambda: _t_wipe(a_np, b_np, t, "down"),
        "zoom_in": lambda: _t_zoom_in(a_np, b_np, t),
        "zoom_out": lambda: _t_zoom_out(a_np, b_np, t),
        "glitch": lambda: _t_glitch(a_np, b_np, t, seed),
        "spin_cw": lambda: _t_spin(a_np, b_np, t, True),
        "spin_ccw": lambda: _t_spin(a_np, b_np, t, False),
        "iris_in": lambda: _t_iris(a_np, b_np, t),
        "slide_up": lambda: _t_slide(a_np, b_np, t, "up"),
        "slide_down": lambda: _t_slide(a_np, b_np, t, "down"),
    }
    return dispatch.get(transition, lambda: _t_dissolve(a_np, b_np, t))()


# ── Node ─────────────────────────────────────────────────────────

class S42CF_Transition:
    """20 CapCut-style transitions between two video clips.
    
    FRAME HANDLING (ported from proven S42P Transition):
    - Blending transitions (dissolve/fade/glitch): use corresponding overlap frames
    - Motion transitions (push/wipe/zoom/spin/iris/slide): use STABLE reference
      frames (last frame of A, first frame of B) to prevent jitter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "First video clip. Plays first, transitions out."}),
                "clip_b": ("IMAGE", {"tooltip": "Second video clip. Transitions in, plays after."}),
                "transition_type": (TRANSITION_TYPES, {
                    "default": "dissolve",
                    "tooltip": "Transition style:\ncut = hard cut (no blending).\ndissolve = crossfade.\nfade_black/white = fade through color.\npush_* = push A off screen.\nwipe_* = reveal B with edge wipe.\nzoom_in/out = scale transition.\nglitch = RGB channel-shift glitch.\nspin_cw/ccw = rotation transition.\niris_in = circular iris wipe.\nslide_up/down = slide reveal."
                }),
                "transition_frames": ("INT", {
                    "default": 12, "min": 1, "max": 120, "step": 1,
                    "tooltip": "Transition duration in frames.\n12 = 0.5s at 24fps.\n24 = 1s at 24fps."}),
                "easing": (ALL_EASINGS, {
                    "default": "ease_in_out",
                    "tooltip": "Transition animation curve."}),
                "glitch_seed": ("INT", {
                    "default": 42, "min": 0, "max": 9999,
                    "tooltip": "Random seed for glitch transition. Only used with 'glitch' type."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("merged_frames", "total_frame_count", "info")
    FUNCTION = "apply_transition"
    CATEGORY = CATEGORY

    def apply_transition(self, clip_a, clip_b, transition_type, transition_frames, easing, glitch_seed):
        
        clip_a = ensure_rgb(clip_a)
        clip_b = ensure_rgb(clip_b)
        
        clip_a, clip_b = match_resolution(clip_a, clip_b)
        na, nb = clip_a.shape[0], clip_b.shape[0]

        # Cut = just concatenate
        if transition_type == "cut":
            merged = torch.cat([clip_a, clip_b], dim=0)
            info = f"Transition(cut): {na}+{nb}={merged.shape[0]} frames"
            return (merged, int(merged.shape[0]), info)

        tf = min(transition_frames, na, nb)
        a_keep = clip_a[:na - tf]
        a_over = clip_a[na - tf:]
        b_over = clip_b[:tf]
        b_keep = clip_b[tf:]

        # CRITICAL: motion transitions use stable reference frames
        use_frame_blend = transition_type in BLENDING_TRANSITIONS
        if not use_frame_blend:
            a_ref = (a_over[-1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            b_ref = (b_over[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        trans_frames = []
        for i in range(tf):
            raw_t = i / max(tf - 1, 1)
            t = apply_easing(raw_t, easing)

            if use_frame_blend:
                a_np = (a_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                b_np = (b_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                a_np = a_ref
                b_np = b_ref

            result_np = _apply_transition_frame(a_np, b_np, t, transition_type, glitch_seed)
            trans_frames.append(torch.from_numpy(result_np.astype(np.float32) / 255.0))

        trans_tensor = torch.stack(trans_frames)
        parts = [p for p in [a_keep, trans_tensor, b_keep] if p.shape[0] > 0]
        merged = torch.cat(parts, dim=0)

        blend_mode = "per-frame" if use_frame_blend else "stable-ref"
        info = (f"Transition({transition_type}): {na}+{nb}-{tf}={merged.shape[0]}f "
                f"({blend_mode}, {easing})")
        logger.info(f"[S42CF] {info}")
        return (merged, int(merged.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_Transition": S42CF_Transition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_Transition": "🔀 S42 CutFlow Transition",
}