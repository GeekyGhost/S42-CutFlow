"""
S42 CutFlow — Video Transitions (Expanded)
=============================================
25 CapCut-style transitions between two video clip batches.
Ported from proven S42P Transition node with correct frame handling.

Includes Audio Synchronized Transitions and Advanced Audio Concatenation.

CRITICAL: Motion transitions (push/wipe/slide) use STABLE reference 
frames (last of A, first of B) to prevent frame jitter in AI gens.
Blending transitions (dissolve/fade/spin/zoom) play continuously through the overlap.

Python 3.12 | ComfyUI Portable | PIL + torch + numpy
"""

import torch
import numpy as np
import math
import logging
from PIL import Image

from .cf_utils import ensure_rgb, match_resolution, apply_easing, ALL_EASINGS

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Expanded"

TRANSITION_TYPES = [
    "cut", "dissolve", "fade_black", "fade_white", "luma_wipe",
    "push_left", "push_right", "push_up", "push_down",
    "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "zoom_in", "zoom_out", "glitch", "pixelize",
    "spin_cw", "spin_ccw", "spin_zoom_cw", "spin_zoom_ccw",
    "iris_in", "iris_out",
    "slide_up", "slide_down",
]

# These transitions look best when the video continues to play during the effect
BLENDING_TRANSITIONS = {
    "dissolve", "fade_black", "fade_white", "glitch", "luma_wipe", "pixelize",
    "spin_cw", "spin_ccw", "spin_zoom_cw", "spin_zoom_ccw",
    "zoom_in", "zoom_out"
}


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

def _t_luma_wipe(a, b, t):
    # Professional transition where bright pixels fade before dark pixels
    luma = np.dot(a[..., :3], [0.2989, 0.5870, 0.1140])
    threshold = t * 255.0
    softness = 30.0
    mask = np.clip((threshold - luma + softness/2) / softness, 0.0, 1.0)
    mask = mask[..., np.newaxis]
    return (a * (1.0 - mask) + b * mask).clip(0, 255).astype(np.uint8)

def _t_pixelize(a, b, t):
    h, w = a.shape[:2]
    pixel_size = int(math.sin(t * math.pi) * 40) + 1
    
    # Crossfade the bases
    base = (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)
    
    if pixel_size <= 1:
        return base
        
    img = Image.fromarray(base)
    small = img.resize((max(1, w // pixel_size), max(1, h // pixel_size)), Image.NEAREST)
    return np.array(small.resize((w, h), Image.NEAREST))

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
    scale = 1.0 + t * 0.5
    bh, bw = max(1, int(h * scale)), max(1, int(w * scale))
    b_s = np.array(Image.fromarray(b).resize((bw, bh), Image.LANCZOS))
    out = a.copy()
    py, px = (bh - h) // 2, (bw - w) // 2
    b_cropped = b_s[py:py+h, px:px+w]
    # Crossfade during the zoom for fluidity
    return (out * (1.0 - t) + b_cropped * t).clip(0, 255).astype(np.uint8)

def _t_zoom_out(a, b, t):
    h, w = a.shape[:2]
    scale = 1.0 + (1.0 - t) * 0.5
    ah, aw = max(1, int(h * scale)), max(1, int(w * scale))
    a_s = np.array(Image.fromarray(a).resize((aw, ah), Image.LANCZOS))
    out = b.copy()
    py, px = (ah - h) // 2, (aw - w) // 2
    a_cropped = a_s[py:py+h, px:px+w]
    return (a_cropped * (1.0 - t) + out * t).clip(0, 255).astype(np.uint8)

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
    direction = -1 if clockwise else 1
    # A spins out, B spins in. Guarantees B lands at exactly 0 degrees.
    angle_a = t * 180.0 * direction
    angle_b = (t - 1.0) * 180.0 * direction
    
    rot_a = np.array(Image.fromarray(a).rotate(angle_a, resample=Image.BICUBIC, expand=False), dtype=np.float32)
    rot_b = np.array(Image.fromarray(b).rotate(angle_b, resample=Image.BICUBIC, expand=False), dtype=np.float32)
    
    # Alpha blend during rotation
    return (rot_a * (1.0 - t) + rot_b * t).clip(0, 255).astype(np.uint8)

def _t_spin_zoom(a, b, t, clockwise):
    direction = -1 if clockwise else 1
    h, w = a.shape[:2]
    
    # A zooms in slightly while spinning
    scale_a = 1.0 + (t * 0.5)
    angle_a = t * 180.0 * direction
    
    # B starts zoomed out, scales to normal, spins to 0
    scale_b = 0.5 + (t * 0.5)
    angle_b = (t - 1.0) * 180.0 * direction
    
    def transform(img_arr, scale, angle):
        img = Image.fromarray(img_arr).rotate(angle, resample=Image.BICUBIC, expand=False)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        
        out = Image.new('RGB', (w, h), (0, 0, 0))
        px, py = (w - nw) // 2, (h - nh) // 2
        out.paste(img, (px, py))
        return np.array(out, dtype=np.float32)
        
    arr_a = transform(a, scale_a, angle_a)
    arr_b = transform(b, scale_b, angle_b)
    
    return (arr_a * (1.0 - t) + arr_b * t).clip(0, 255).astype(np.uint8)

def _t_iris_in(a, b, t):
    h, w = a.shape[:2]
    out = a.copy()
    cy, cx = h // 2, w // 2
    r = t * math.hypot(cx, cy)
    Y, X = np.ogrid[:h, :w]
    mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= r
    out[mask] = b[mask]
    return out

def _t_iris_out(a, b, t):
    h, w = a.shape[:2]
    out = b.copy()
    cy, cx = h // 2, w // 2
    # Circle shrinks around A, revealing B outside
    r = (1.0 - t) * math.hypot(cx, cy)
    Y, X = np.ogrid[:h, :w]
    mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= r
    out[mask] = a[mask]
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
    t = float(np.clip(t, 0.0, 1.0))
    dispatch = {
        "dissolve": lambda: _t_dissolve(a_np, b_np, t),
        "fade_black": lambda: _t_fade_color(a_np, b_np, t, (0, 0, 0)),
        "fade_white": lambda: _t_fade_color(a_np, b_np, t, (255, 255, 255)),
        "luma_wipe": lambda: _t_luma_wipe(a_np, b_np, t),
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
        "pixelize": lambda: _t_pixelize(a_np, b_np, t),
        "spin_cw": lambda: _t_spin(a_np, b_np, t, True),
        "spin_ccw": lambda: _t_spin(a_np, b_np, t, False),
        "spin_zoom_cw": lambda: _t_spin_zoom(a_np, b_np, t, True),
        "spin_zoom_ccw": lambda: _t_spin_zoom(a_np, b_np, t, False),
        "iris_in": lambda: _t_iris_in(a_np, b_np, t),
        "iris_out": lambda: _t_iris_out(a_np, b_np, t),
        "slide_up": lambda: _t_slide(a_np, b_np, t, "up"),
        "slide_down": lambda: _t_slide(a_np, b_np, t, "down"),
    }
    return dispatch.get(transition, lambda: _t_dissolve(a_np, b_np, t))()


# ── Video Only Transition Node ─────────────────────────────────────────────────────────

class S42CF_Transition:
    """25 CapCut-style transitions between two video clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "First video clip. Plays first, transitions out."}),
                "clip_b": ("IMAGE", {"tooltip": "Second video clip. Transitions in, plays after."}),
                "transition_type": (TRANSITION_TYPES, {
                    "default": "dissolve",
                    "tooltip": "Transition style:\ncut = hard cut.\ndissolve = crossfade.\nluma_wipe = fade based on brightness.\nzoom_in/out = scale transition.\nspin/spin_zoom = fluid rotation blends.\niris_in/out = circular wipe.\npixelize = stylistic mosaic."
                }),
                "transition_frames": ("INT", {
                    "default": 12, "min": 1, "max": 120, "step": 1,
                    "tooltip": "Transition duration in frames. Controls the speed of the effect (more frames = slower)."}),
                "easing": (ALL_EASINGS, {
                    "default": "ease_in_out",
                    "tooltip": "Transition animation curve to control pacing."}),
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

        if transition_type == "cut":
            merged = torch.cat([clip_a, clip_b], dim=0)
            info = f"Transition(cut): {na}+{nb}={merged.shape[0]} frames"
            return (merged, int(merged.shape[0]), info)

        tf = min(transition_frames, na, nb)
        a_keep = clip_a[:na - tf]
        a_over = clip_a[na - tf:]
        b_over = clip_b[:tf]
        b_keep = clip_b[tf:]

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

        blend_mode = "fluid-playback" if use_frame_blend else "stable-ref"
        info = (f"Transition({transition_type}): {na}+{nb}-{tf}={merged.shape[0]}f "
                f"({blend_mode}, {easing})")
        logger.info(f"[S42CF] {info}")
        return (merged, int(merged.shape[0]), info)

# ── Synchronized Video & Audio Transition Node ─────────────────────────────────────────

class S42CF_TransitionSync:
    """
    Synchronized Transition for Video and Audio.
    Automatically applies the correct visual transition and precisely calculates 
    the overlap reduction to ensure audio remains perfectly synced.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "First video clip."}),
                "audio_a": ("AUDIO", {"tooltip": "Audio track for the first video clip."}),
                "clip_b": ("IMAGE", {"tooltip": "Second video clip."}),
                "audio_b": ("AUDIO", {"tooltip": "Audio track for the second video clip."}),
                "transition_type": (TRANSITION_TYPES, {"default": "dissolve", "tooltip": "Style of visual transition."}),
                "transition_frames": ("INT", {"default": 12, "min": 1, "max": 120, "step": 1, "tooltip": "Visual overlap duration in frames."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Frames per second of the video. Required to accurately calculate the audio overlap duration."}),
                "audio_crossfade": ("BOOLEAN", {"default": True, "tooltip": "If True, performs an equal-length crossfade on the audio to match the visual transition. If False, performs a hard audio cut."}),
                "easing": (ALL_EASINGS, {"default": "ease_in_out", "tooltip": "Transition animation curve."}),
                "glitch_seed": ("INT", {"default": 42, "min": 0, "max": 9999, "tooltip": "Random seed for glitch transition."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("merged_video", "merged_audio", "total_frame_count", "info")
    FUNCTION = "apply_sync_transition"
    CATEGORY = CATEGORY

    def apply_sync_transition(self, clip_a, audio_a, clip_b, audio_b, transition_type, transition_frames, fps, audio_crossfade, easing, glitch_seed):
        # 1. Handle Video Transition
        transition_node = S42CF_Transition()
        merged_video, total_frames, info_video = transition_node.apply_transition(
            clip_a, clip_b, transition_type, transition_frames, easing, glitch_seed
        )

        # 2. Handle Audio Sync
        overlap_sec = transition_frames / fps
        sample_rate = audio_a['sample_rate']
        a_waveform = audio_a['waveform'] 
        b_waveform = audio_b['waveform']
        
        overlap_samples = int(overlap_sec * sample_rate)
        
        if transition_type == "cut" or overlap_samples <= 0:
            merged_audio_waveform = torch.cat([a_waveform, b_waveform], dim=-1)
            info_audio = "Audio: Hard cut (no overlap)"
        else:
            a_cut = a_waveform[..., :-overlap_samples]
            a_end = a_waveform[..., -overlap_samples:]
            
            b_start = b_waveform[..., :overlap_samples]
            b_end = b_waveform[..., overlap_samples:]

            if audio_crossfade:
                t = torch.linspace(0, 1, overlap_samples, device=a_waveform.device).view(1, 1, -1)
                fade_out = a_end * (1 - t)
                fade_in = b_start * t
                audio_mid = fade_out + fade_in
                merged_audio_waveform = torch.cat([a_cut, audio_mid, b_end], dim=-1)
                info_audio = f"Audio: Crossfaded ({overlap_samples} samples)"
            else:
                merged_audio_waveform = torch.cat([a_cut, b_end], dim=-1)
                info_audio = f"Audio: Overlapped Hard Cut ({overlap_samples} samples dropped)"

        merged_audio = {"waveform": merged_audio_waveform, "sample_rate": sample_rate}
        
        info = f"{info_video} | {info_audio}"
        return (merged_video, merged_audio, total_frames, info)

# ── Advanced Audio Concatenation Node ──────────────────────────────────────────────

class S42CF_AudioConcatAdv:
    """
    Advanced Audio Concatenator with manual padding offsets and professional logarithmic crossfading.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {"tooltip": "First audio track."}),
                "audio_b": ("AUDIO", {"tooltip": "Second audio track."}),
                "pad_ms": ("INT", {"default": 0, "min": -5000, "max": 5000, "step": 1, "tooltip": "Positive (+): Adds silence between tracks. Negative (-): Trims the end of audio_a to pull audio_b in earlier."}),
                "crossfade_ms": ("INT", {"default": 50, "min": 0, "max": 2000, "step": 10, "tooltip": "Duration of the crossfade in milliseconds to prevent audio pops."}),
                "fade_curve": (["logarithmic", "linear", "exponential"], {"default": "logarithmic", "tooltip": "Curve type. Logarithmic provides an 'Equal Power' crossfade which prevents volume dips, standard in pro NLEs."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("merged_audio", "info")
    FUNCTION = "concat_pro"
    CATEGORY = CATEGORY

    def concat_pro(self, audio_a, audio_b, pad_ms, crossfade_ms, fade_curve):
        sr = audio_a['sample_rate']
        wav_a = audio_a['waveform'] 
        wav_b = audio_b['waveform']
        device = wav_a.device
        
        # 1. Handle Padding / Offset
        if pad_ms > 0:
            pad_samples = int((pad_ms / 1000.0) * sr)
            silence = torch.zeros((wav_a.shape[0], wav_a.shape[1], pad_samples), device=device)
            wav_a = torch.cat([wav_a, silence], dim=-1)
        elif pad_ms < 0:
            trim_samples = int((abs(pad_ms) / 1000.0) * sr)
            trim_samples = min(trim_samples, wav_a.shape[-1] - 1) 
            wav_a = wav_a[..., :-trim_samples]

        # 2. Handle Crossfade
        cf_samples = int((crossfade_ms / 1000.0) * sr)
        cf_samples = min(cf_samples, wav_a.shape[-1], wav_b.shape[-1])
        
        if cf_samples > 0:
            a_main = wav_a[..., :-cf_samples]
            a_fade = wav_a[..., -cf_samples:]
            b_fade = wav_b[..., :cf_samples]
            b_main = wav_b[..., cf_samples:]
            
            # Generate Curves
            t = torch.linspace(0, 1, cf_samples, device=device).view(1, 1, -1)
            if fade_curve == "logarithmic":
                curve_a = torch.sqrt(1 - t)
                curve_b = torch.sqrt(t)
            elif fade_curve == "exponential":
                curve_a = 1 - torch.pow(t, 2)
                curve_b = torch.pow(t, 2)
            else: # linear
                curve_a = 1 - t
                curve_b = t
                
            merged_cf = (a_fade * curve_a) + (b_fade * curve_b)
            final_wav = torch.cat([a_main, merged_cf, b_main], dim=-1)
        else:
            final_wav = torch.cat([wav_a, wav_b], dim=-1)

        info = f"Audio Concat Pro: {pad_ms}ms offset, {crossfade_ms}ms {fade_curve} crossfade."
        logger.info(f"[S42CF] {info}")

        return ({"waveform": final_wav, "sample_rate": sr}, info)


# ── Mappings ────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42CF_Transition": S42CF_Transition,
    "S42CF_TransitionSync": S42CF_TransitionSync,
    "S42CF_AudioConcatAdv": S42CF_AudioConcatAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_Transition": "🔀 S42 CutFlow Transition",
    "S42CF_TransitionSync": "🔀 S42 CutFlow Sync Transition",
    "S42CF_AudioConcatAdv": "🔊 S42 CutFlow Audio Concat Pro",
}