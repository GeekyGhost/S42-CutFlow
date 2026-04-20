"""
S42 CutFlow — Shared Utilities
================================
Common helpers for all CutFlow nodes. Frame manipulation,
easing, keyframe parsing, chunked processing, fallback detection.

All ops work on ComfyUI IMAGE tensors: [B, H, W, C] float32 0-1.
No GPU models — pure torch/numpy/scipy/PIL.

Python 3.12 | ComfyUI Portable
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import re
import logging
from typing import List, Tuple, Optional, Dict, Union

logger = logging.getLogger("S42CutFlow")

# ── Optional dependency detection ─────────────────────────────────

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available — text/GIF nodes disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not available — stabilize/bilateral will use fallback")

try:
    from scipy import signal as sp_signal
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

CHUNK_SIZE = 48


# ── Tensor helpers ────────────────────────────────────────────────

def ensure_rgb(frames: torch.Tensor) -> torch.Tensor:
    """Guarantee [B,H,W,3] — drop alpha or expand grayscale."""
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    return frames


def ensure_rgba(frames: torch.Tensor) -> torch.Tensor:
    """Guarantee [B,H,W,4]."""
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 3:
        alpha = torch.ones(*frames.shape[:3], 1, dtype=frames.dtype, device=frames.device)
        frames = torch.cat([frames, alpha], dim=-1)
    return frames


def match_resolution(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize b to match a's H×W if they differ."""
    if a.shape[1:3] == b.shape[1:3]:
        return a, b
    h, w = a.shape[1], a.shape[2]
    b_p = b.permute(0, 3, 1, 2)
    b_r = F.interpolate(b_p, size=(h, w), mode="bilinear", align_corners=False)
    return a, b_r.permute(0, 2, 3, 1)


def frames_to_np(frames: torch.Tensor) -> np.ndarray:
    """[B,H,W,C] float32 0-1 → [B,H,W,C] uint8 0-255"""
    return (frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def np_to_frames(arr: np.ndarray) -> torch.Tensor:
    """[B,H,W,C] uint8 0-255 → [B,H,W,C] float32 0-1"""
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def frame_to_pil(frame: torch.Tensor) -> "Image.Image":
    """Single frame [H,W,C] float32 → PIL RGB"""
    return Image.fromarray((frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def pil_to_frame(img: "Image.Image") -> torch.Tensor:
    """PIL RGB → [H,W,C] float32"""
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0)


def resize_frames(frames: torch.Tensor, h: int, w: int,
                   mode: str = "bilinear") -> torch.Tensor:
    """Resize batch [B,H,W,C] → [B,h,w,C]."""
    x = frames.permute(0, 3, 1, 2)
    align = False if mode != "nearest" else None
    x = F.interpolate(x, size=(h, w), mode=mode, align_corners=align)
    return x.permute(0, 2, 3, 1)


# ── Easing functions ──────────────────────────────────────────────

def ease_linear(t): return t
def ease_in_quad(t): return t * t
def ease_out_quad(t): return t * (2.0 - t)
def ease_in_out_quad(t):
    return 2.0 * t * t if t < 0.5 else -1.0 + (4.0 - 2.0 * t) * t
def ease_in_cubic(t): return t ** 3
def ease_out_cubic(t): return (t - 1.0) ** 3 + 1.0
def ease_in_out_cubic(t):
    return 4.0 * t ** 3 if t < 0.5 else 0.5 * (2.0 * t - 2.0) ** 3 + 1.0
def ease_in_elastic(t):
    if t in (0.0, 1.0): return t
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((t * 10.0 - 10.75) * (2.0 * math.pi) / 3.0)
def ease_out_elastic(t):
    if t in (0.0, 1.0): return t
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * (2.0 * math.pi) / 3.0) + 1.0
def ease_out_bounce(t):
    n, d = 7.5625, 2.75
    if t < 1.0 / d: return n * t * t
    elif t < 2.0 / d: t -= 1.5 / d; return n * t * t + 0.75
    elif t < 2.5 / d: t -= 2.25 / d; return n * t * t + 0.9375
    else: t -= 2.625 / d; return n * t * t + 0.984375

EASING_FUNCTIONS = {
    "linear": ease_linear, "ease_in": ease_in_quad, "ease_out": ease_out_quad,
    "ease_in_out": ease_in_out_quad, "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic, "ease_in_out_cubic": ease_in_out_cubic,
    "elastic_in": ease_in_elastic, "elastic_out": ease_out_elastic,
    "bounce": ease_out_bounce,
}
ALL_EASINGS = list(EASING_FUNCTIONS.keys())


def apply_easing(t: float, easing: str) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    fn = EASING_FUNCTIONS.get(easing, ease_in_out_quad)
    return fn(t)


# ── Keyframe parser ───────────────────────────────────────────────
# Format: "frame:value, frame:value, ..."
# Optional easing: "frame:value:easing, ..."

def parse_keyframe_string(s: str) -> List[Tuple[int, float, str]]:
    if not s or not s.strip():
        return []
    keyframes = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 2:
            continue
        try:
            frame = int(parts[0].strip())
            value = float(parts[1].strip())
            easing = parts[2].strip() if len(parts) >= 3 else "ease_in_out"
            if easing not in EASING_FUNCTIONS:
                easing = "ease_in_out"
            keyframes.append((frame, value, easing))
        except (ValueError, IndexError):
            continue
    return sorted(keyframes, key=lambda x: x[0])


def interpolate_keyframes(keyframes: List[Tuple[int, float, str]], frame: int) -> float:
    if not keyframes:
        return 0.0
    if frame <= keyframes[0][0]:
        return keyframes[0][1]
    if frame >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        f0, v0, easing = keyframes[i]
        f1, v1, _ = keyframes[i + 1]
        if f0 <= frame <= f1:
            if f1 == f0:
                return v1
            t = (frame - f0) / (f1 - f0)
            t_e = apply_easing(t, easing)
            return v0 + (v1 - v0) * t_e
    return keyframes[-1][1]


def build_value_curve(keyframes: List[Tuple[int, float, str]], n_frames: int) -> List[float]:
    return [interpolate_keyframes(keyframes, f) for f in range(n_frames)]


# ── FPS conversion ────────────────────────────────────────────────

def convert_fps(frames: torch.Tensor, src_fps: float, dst_fps: float,
                mode: str = "blend") -> torch.Tensor:
    if abs(src_fps - dst_fps) < 0.01:
        return frames
    n_src = frames.shape[0]
    duration = n_src / src_fps
    n_dst = max(1, round(duration * dst_fps))
    if mode == "duplicate":
        indices = [min(int(i * src_fps / dst_fps), n_src - 1) for i in range(n_dst)]
        return frames[indices]
    out = []
    for i in range(n_dst):
        src_pos = i * src_fps / dst_fps
        idx_lo = min(int(src_pos), n_src - 1)
        idx_hi = min(idx_lo + 1, n_src - 1)
        frac = src_pos - int(src_pos)
        blended = frames[idx_lo] * (1.0 - frac) + frames[idx_hi] * frac
        out.append(blended)
    return torch.stack(out)


# ── Chunked processing ────────────────────────────────────────────

def process_frames_chunked(frames: torch.Tensor, fn, chunk_size: int = CHUNK_SIZE, **kwargs) -> torch.Tensor:
    """Process frames in chunks to avoid OOM on long clips."""
    n = frames.shape[0]
    if n <= chunk_size:
        return fn(frames, **kwargs)
    results = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = fn(frames[start:end], **kwargs)
        results.append(chunk)
    return torch.cat(results, dim=0)


# ── Audio helpers ─────────────────────────────────────────────────

def get_audio_numpy(audio_dict: dict) -> Tuple[np.ndarray, int]:
    """Extract numpy waveform from ComfyUI AUDIO dict.
    Returns (waveform [channels, samples], sample_rate)."""
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    if isinstance(waveform, torch.Tensor):
        wav = waveform.squeeze(0).cpu().numpy()
    else:
        wav = np.array(waveform).squeeze(0)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    return wav, sr


def numpy_to_audio(wav: np.ndarray, sr: int) -> dict:
    """Numpy waveform → ComfyUI AUDIO dict."""
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    return {"waveform": t, "sample_rate": sr}


# ── SRT parser ────────────────────────────────────────────────────

def parse_srt(srt_text: str) -> List[Dict]:
    """Parse SRT subtitle string into list of {start_s, end_s, text}."""
    entries = []
    blocks = re.split(r'\n\s*\n', srt_text.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        time_line = lines[1]
        match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            time_line
        )
        if not match:
            continue
        g = [int(x) for x in match.groups()]
        start_s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end_s = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        text = '\n'.join(lines[2:]).strip()
        text = re.sub(r'<[^>]+>', '', text)
        entries.append({"start_s": start_s, "end_s": end_s, "text": text})
    return entries


# ── Blend modes ───────────────────────────────────────────────────

# ── Hex color parser ──────────────────────────────────────────────

def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Parse '#RRGGBB' or '#RGB' hex string to (r, g, b) floats 0-1.
    Returns (0, 0, 0) on parse failure."""
    try:
        s = hex_str.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def hex_to_rgb255(hex_str: str) -> Tuple[int, int, int]:
    """Parse '#RRGGBB' to (r, g, b) ints 0-255."""
    r, g, b = hex_to_rgb(hex_str)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── Blend modes ───────────────────────────────────────────────────

def blend_images(base: np.ndarray, layer: np.ndarray, opacity: float = 1.0,
                 mode: str = "normal") -> np.ndarray:
    """Blend two float32 [H,W,C] arrays."""
    b = base.astype(np.float64)
    l = layer.astype(np.float64)
    if mode == "normal":
        result = l
    elif mode == "screen":
        result = 1.0 - (1.0 - b) * (1.0 - l)
    elif mode == "multiply":
        result = b * l
    elif mode == "overlay":
        mask = b < 0.5
        result = np.where(mask, 2.0 * b * l, 1.0 - 2.0 * (1.0 - b) * (1.0 - l))
    elif mode == "add":
        result = b + l
    elif mode == "difference":
        result = np.abs(b - l)
    else:
        result = l
    blended = b * (1.0 - opacity) + result * opacity
    return blended.clip(0.0, 1.0).astype(np.float32)
