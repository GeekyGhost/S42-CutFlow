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
    luma = np.dot(a[..., :3], [0.2989, 0.5870, 0.1140])
    threshold = t * 255.0
    softness = 30.0
    mask = np.clip((threshold - luma + softness/2) / softness, 0.0, 1.0)
    mask = mask[..., np.newaxis]
    return (a * (1.0 - mask) + b * mask).clip(0, 255).astype(np.uint8)

def _t_pixelize(a, b, t):
    h, w = a.shape[:2]
    pixel_size = int(math.sin(t * math.pi) * 40) + 1
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
        if ox < w:
            out[:, :w-ox] = a[:, ox:]
        if ox > 0:
            out[:, w-ox:] = b[:, :ox]
    elif direction == "right":
        ox = int(w * t)
        if ox < w:
            out[:, ox:] = a[:, :w-ox]
        if ox > 0:
            out[:, :ox] = b[:, w-ox:]
    elif direction == "up":
        oy = int(h * t)
        if oy < h:
            out[:h-oy, :] = a[oy:, :]
        if oy > 0:
            out[h-oy:, :] = b[:oy, :]
    elif direction == "down":
        oy = int(h * t)
        if oy < h:
            out[oy:, :] = a[:h-oy, :]
        if oy > 0:
            out[:oy, :] = b[h-oy:, :]
    return out

def _t_wipe(a, b, t, direction):
    h, w = a.shape[:2]
    out = a.copy()
    if direction == "left":
        cut = int(w * t)
        out[:, :cut] = b[:, :cut]
    elif direction == "right":
        cut = int(w * (1.0 - t))
        out[:, cut:] = b[:, cut:]
    elif direction == "up":
        cut = int(h * t)
        out[:cut, :] = b[:cut, :]
    elif direction == "down":
        cut = int(h * (1.0 - t))
        out[cut:, :] = b[cut:, :]
    return out

def _t_zoom_in(a, b, t):
    h, w = a.shape[:2]
    scale = 1.0 + t * 0.5
    bh, bw = max(1, int(h * scale)), max(1, int(w * scale))
    b_s = np.array(Image.fromarray(b).resize((bw, bh), Image.LANCZOS))
    out = a.copy()
    py, px = (bh - h) // 2, (bw - w) // 2
    b_cropped = b_s[py:py+h, px:px+w]
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
    angle_a = t * 180.0 * direction
    angle_b = (t - 1.0) * 180.0 * direction
    rot_a = np.array(Image.fromarray(a).rotate(angle_a, resample=Image.BICUBIC, expand=False), dtype=np.float32)
    rot_b = np.array(Image.fromarray(b).rotate(angle_b, resample=Image.BICUBIC, expand=False), dtype=np.float32)
    return (rot_a * (1.0 - t) + rot_b * t).clip(0, 255).astype(np.uint8)

def _t_spin_zoom(a, b, t, clockwise):
    direction = -1 if clockwise else 1
    h, w = a.shape[:2]
    scale_a = 1.0 + (t * 0.5)
    angle_a = t * 180.0 * direction
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
        if sh > 0:
            out[h-sh:, :] = b[:sh, :]
    elif direction == "down":
        sh = int(h * t)
        if sh > 0:
            out[:sh, :] = b[h-sh:, :]
    return out


def _apply_transition_frame(a_np, b_np, t, transition, seed=42):
    t = float(np.clip(t, 0.0, 1.0))
    if transition == "dissolve":
        return _t_dissolve(a_np, b_np, t)
    if transition == "fade_black":
        return _t_fade_color(a_np, b_np, t, (0, 0, 0))
    if transition == "fade_white":
        return _t_fade_color(a_np, b_np, t, (255, 255, 255))
    if transition == "luma_wipe":
        return _t_luma_wipe(a_np, b_np, t)
    if transition == "push_left":
        return _t_push(a_np, b_np, t, "left")
    if transition == "push_right":
        return _t_push(a_np, b_np, t, "right")
    if transition == "push_up":
        return _t_push(a_np, b_np, t, "up")
    if transition == "push_down":
        return _t_push(a_np, b_np, t, "down")
    if transition == "wipe_left":
        return _t_wipe(a_np, b_np, t, "left")
    if transition == "wipe_right":
        return _t_wipe(a_np, b_np, t, "right")
    if transition == "wipe_up":
        return _t_wipe(a_np, b_np, t, "up")
    if transition == "wipe_down":
        return _t_wipe(a_np, b_np, t, "down")
    if transition == "zoom_in":
        return _t_zoom_in(a_np, b_np, t)
    if transition == "zoom_out":
        return _t_zoom_out(a_np, b_np, t)
    if transition == "glitch":
        return _t_glitch(a_np, b_np, t, seed)
    if transition == "pixelize":
        return _t_pixelize(a_np, b_np, t)
    if transition == "spin_cw":
        return _t_spin(a_np, b_np, t, True)
    if transition == "spin_ccw":
        return _t_spin(a_np, b_np, t, False)
    if transition == "spin_zoom_cw":
        return _t_spin_zoom(a_np, b_np, t, True)
    if transition == "spin_zoom_ccw":
        return _t_spin_zoom(a_np, b_np, t, False)
    if transition == "iris_in":
        return _t_iris_in(a_np, b_np, t)
    if transition == "iris_out":
        return _t_iris_out(a_np, b_np, t)
    if transition == "slide_up":
        return _t_slide(a_np, b_np, t, "up")
    if transition == "slide_down":
        return _t_slide(a_np, b_np, t, "down")
    return _t_dissolve(a_np, b_np, t)


class S42CF_Transition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {}),
                "clip_b": ("IMAGE", {}),
                "transition_type": (TRANSITION_TYPES, {"default": "dissolve"}),
                "transition_frames": ("INT", {"default": 12, "min": 1, "max": 120, "step": 1}),
                "easing": (ALL_EASINGS, {"default": "ease_in_out"}),
                "glitch_seed": ("INT", {"default": 42, "min": 0, "max": 9999}),
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
            return (merged, int(merged.shape[0]), f"Transition(cut): {na}+{nb}")

        tf = min(transition_frames, na, nb)
        a_over = clip_a[na - tf:]
        b_over = clip_b[:tf]

        use_frame_blend = transition_type in BLENDING_TRANSITIONS
        if not use_frame_blend:
            a_ref = (a_over[-1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            b_ref = (b_over[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        trans_frames = []
        for i in range(tf):
            t = apply_easing(i / max(tf - 1, 1), easing)
            if use_frame_blend:
                a_np = (a_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                b_np = (b_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                a_np, b_np = a_ref, b_ref

            result_np = _apply_transition_frame(a_np, b_np, t, transition_type, glitch_seed)
            trans_frames.append(torch.from_numpy(result_np.astype(np.float32) / 255.0))

        merged = torch.cat([clip_a[:na - tf], torch.stack(trans_frames), clip_b[tf:]], dim=0)
        return (merged, int(merged.shape[0]), f"Transition({transition_type})")


class S42CF_TransitionSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {}),
                "audio_a": ("AUDIO", {}),
                "clip_b": ("IMAGE", {}),
                "audio_b": ("AUDIO", {}),
                "transition_type": (TRANSITION_TYPES, {"default": "dissolve"}),
                "transition_frames": ("INT", {"default": 12}),
                "fps": ("FLOAT", {"default": 24.0}),
                "audio_crossfade": ("BOOLEAN", {"default": True}),
                "easing": (ALL_EASINGS, {"default": "ease_in_out"}),
                "glitch_seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("merged_video", "merged_audio", "total_frame_count", "info")
    FUNCTION = "apply_sync_transition"
    CATEGORY = CATEGORY

    def apply_sync_transition(self, clip_a, audio_a, clip_b, audio_b, transition_type, transition_frames, fps, audio_crossfade, easing, glitch_seed):
        transition_node = S42CF_Transition()
        merged_video, total_frames, info_video = transition_node.apply_transition(
            clip_a, clip_b, transition_type, transition_frames, easing, glitch_seed
        )

        overlap_sec = transition_frames / fps
        sr = audio_a['sample_rate']
        wav_a = audio_a['waveform'] 
        wav_b = audio_b['waveform']
        overlap_samples = int(overlap_sec * sr)
        
        if transition_type == "cut" or overlap_samples <= 0:
            merged_audio_waveform = torch.cat([wav_a, wav_b], dim=-1)
        else:
            a_cut = wav_a[..., :-overlap_samples]
            a_end = wav_a[..., -overlap_samples:]
            b_start = wav_b[..., :overlap_samples]
            b_end = wav_b[..., overlap_samples:]

            if audio_crossfade:
                t = torch.linspace(0, 1, overlap_samples, device=wav_a.device).view(1, 1, -1)
                audio_mid = (a_end * (1 - t)) + (b_start * t)
                merged_audio_waveform = torch.cat([a_cut, audio_mid, b_end], dim=-1)
            else:
                merged_audio_waveform = torch.cat([a_cut, b_end], dim=-1)

        return (merged_video, {"waveform": merged_audio_waveform, "sample_rate": sr}, total_frames, info_video)


class S42CF_AudioConcatAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
                "pad_ms": ("INT", {"default": 0}),
                "crossfade_ms": ("INT", {"default": 50}),
                "fade_curve": (["logarithmic", "linear", "exponential"], {"default": "logarithmic"}),
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
        
        if pad_ms > 0:
            pad_samples = int((pad_ms / 1000.0) * sr)
            silence = torch.zeros((wav_a.shape[0], wav_a.shape[1], pad_samples), device=device)
            wav_a = torch.cat([wav_a, silence], dim=-1)
        elif pad_ms < 0:
            trim_samples = min(int((abs(pad_ms) / 1000.0) * sr), wav_a.shape[-1] - 1)
            wav_a = wav_a[..., :-trim_samples]

        cf_samples = min(int((crossfade_ms / 1000.0) * sr), wav_a.shape[-1], wav_b.shape[-1])
        
        if cf_samples > 0:
            a_main, a_fade = wav_a[..., :-cf_samples], wav_a[..., -cf_samples:]
            b_fade, b_main = wav_b[..., :cf_samples], wav_b[..., cf_samples:]
            t = torch.linspace(0, 1, cf_samples, device=device).view(1, 1, -1)
            if fade_curve == "logarithmic":
                curve_a, curve_b = torch.sqrt(1 - t), torch.sqrt(t)
            elif fade_curve == "exponential":
                curve_a, curve_b = 1 - torch.pow(t, 2), torch.pow(t, 2)
            else:
                curve_a, curve_b = 1 - t, t
            merged_cf = (a_fade * curve_a) + (b_fade * curve_b)
            final_wav = torch.cat([a_main, merged_cf, b_main], dim=-1)
        else:
            final_wav = torch.cat([wav_a, wav_b], dim=-1)

        return ({"waveform": final_wav, "sample_rate": sr}, "Audio Concat Success")


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
