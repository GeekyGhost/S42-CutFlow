"""
S42 CutFlow — Audio Sync
==========================
Audio trimming, beat detection for synced cuts, and audio-video duration matching.

Python 3.12 | ComfyUI Portable | torch + numpy + scipy
"""

import torch
import numpy as np
import logging

from .cf_utils import (
    ensure_rgb, get_audio_numpy, numpy_to_audio, SCIPY_AVAILABLE
)

if SCIPY_AVAILABLE:
    from scipy.signal import find_peaks

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Audio Sync"


class S42CF_AudioTrim:
    """Trim audio to match clip length or custom in/out points."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "ComfyUI AUDIO input to trim."}),
                "mode": (["match_frames", "custom_time", "custom_frames"], {
                    "default": "match_frames",
                    "tooltip": "'match_frames' = trim audio to match connected clip frame count.\n'custom_time' = trim to in/out seconds.\n'custom_frames' = trim to in/out frame numbers at given fps."
                }),
                "in_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.1,
                    "tooltip": "Start time in seconds (custom_time mode)."}),
                "out_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 99999.0, "step": 0.1,
                    "tooltip": "End time in seconds. -1 = end of audio."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "FPS for frame-based calculations."}),
                "target_frames": ("INT", {"default": 48, "min": 1, "max": 99999,
                    "tooltip": "Target frame count for match_frames mode."}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Fade-in duration in seconds."}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Fade-out duration in seconds."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, audio, mode, in_seconds, out_seconds, fps, target_frames, fade_in, fade_out):
        wav, sr = get_audio_numpy(audio)
        total_samples = wav.shape[-1]

        if mode == "match_frames":
            duration = target_frames / fps
            start_sample = 0
            end_sample = min(int(duration * sr), total_samples)
        elif mode == "custom_time":
            start_sample = int(in_seconds * sr)
            end_sample = int(out_seconds * sr) if out_seconds >= 0 else total_samples
        else:
            start_sample = int(in_seconds * fps / fps * sr)
            end_sample = int(out_seconds * fps / fps * sr) if out_seconds >= 0 else total_samples

        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        trimmed = wav[:, start_sample:end_sample].copy()

        n_samples = trimmed.shape[-1]
        if fade_in > 0:
            fade_samples = min(int(fade_in * sr), n_samples)
            fade_curve = np.linspace(0, 1, fade_samples, dtype=np.float32)
            trimmed[:, :fade_samples] *= fade_curve

        if fade_out > 0:
            fade_samples = min(int(fade_out * sr), n_samples)
            fade_curve = np.linspace(1, 0, fade_samples, dtype=np.float32)
            trimmed[:, -fade_samples:] *= fade_curve

        duration_s = n_samples / sr
        info = f"AudioTrim({mode}): {duration_s:.2f}s, {n_samples} samples, fade_in={fade_in}s, fade_out={fade_out}s"
        return (numpy_to_audio(trimmed, sr), info)


class S42CF_BeatSnap:
    """Detect beats and output frame indices for synced cuts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to analyze for beats."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Video FPS to convert beat times to frame indices."}),
                "sensitivity": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Beat detection sensitivity. Lower = fewer beats detected (only strong beats). Higher = more beats."
                }),
                "min_interval": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Minimum time between detected beats in seconds. Prevents double-triggers."
                }),
                "output_format": (["frame_indices", "split_points", "keyframe_string"], {
                    "default": "frame_indices",
                    "tooltip": "'frame_indices' = comma-separated frame numbers.\n'split_points' = ready for MultiSplit node.\n'keyframe_string' = frame:1.0 keyframe format."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("beat_data", "beat_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, audio, fps, sensitivity, min_interval, output_format):
        wav, sr = get_audio_numpy(audio)
        mono = wav.mean(axis=0) if wav.shape[0] > 1 else wav[0]

        hop = int(sr * 0.01)
        window = int(sr * 0.02)
        energy = []
        for i in range(0, len(mono) - window, hop):
            energy.append(np.sum(mono[i:i + window] ** 2))
        energy = np.array(energy)
        energy = energy / (energy.max() + 1e-10)

        diff = np.diff(energy)
        diff = np.maximum(diff, 0)
        threshold = np.percentile(diff, (1 - sensitivity) * 100)
        min_dist = int(min_interval / 0.01)

        if SCIPY_AVAILABLE:
            peaks, _ = find_peaks(diff, height=threshold, distance=min_dist)
        else:
            peaks = []
            last = -min_dist
            for i in range(len(diff)):
                if diff[i] > threshold and i - last >= min_dist:
                    peaks.append(i)
                    last = i
            peaks = np.array(peaks)

        beat_times = peaks * 0.01
        beat_frames = [int(round(t * fps)) for t in beat_times]
        beat_frames = sorted(set(bf for bf in beat_frames if bf >= 0))

        if output_format == "frame_indices":
            data = ", ".join(str(f) for f in beat_frames)
        elif output_format == "split_points":
            data = ", ".join(str(f) for f in beat_frames)
        elif output_format == "keyframe_string":
            data = ", ".join(f"{f}:1.0" for f in beat_frames)
        else:
            data = ", ".join(str(f) for f in beat_frames)

        info = f"BeatSnap: {len(beat_frames)} beats detected, sensitivity={sensitivity}, min_interval={min_interval}s"
        return (data, len(beat_frames), info)


class S42CF_AudioVideoSync:
    """Match audio duration to video duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to adjust."}),
                "target_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.1,
                    "tooltip": "Target duration in seconds. 0 = use target_frames/fps."
                }),
                "target_frames": ("INT", {"default": 48, "min": 1, "max": 99999,
                    "tooltip": "Target frame count (used when target_duration=0)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "FPS for frame-to-time conversion."}),
                "method": (["trim_pad", "time_stretch"], {
                    "default": "trim_pad",
                    "tooltip": "'trim_pad' = cut or silence-pad to match duration (preserves pitch).\n'time_stretch' = resample to fit (may change pitch slightly)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, audio, target_duration, target_frames, fps, method):
        wav, sr = get_audio_numpy(audio)
        current_samples = wav.shape[-1]
        current_dur = current_samples / sr

        if target_duration > 0:
            target_dur = target_duration
        else:
            target_dur = target_frames / fps

        target_samples = int(target_dur * sr)

        if method == "trim_pad":
            if target_samples <= current_samples:
                result = wav[:, :target_samples]
            else:
                padding = np.zeros((wav.shape[0], target_samples - current_samples), dtype=wav.dtype)
                result = np.concatenate([wav, padding], axis=-1)
        elif method == "time_stretch":
            ratio = target_samples / max(current_samples, 1)
            x = np.arange(target_samples, dtype=np.float64)
            src_positions = x / ratio
            src_positions = np.clip(src_positions, 0, current_samples - 1)
            lo = src_positions.astype(int)
            hi = np.minimum(lo + 1, current_samples - 1)
            frac = (src_positions - lo).astype(np.float32)
            result = wav[:, lo] * (1 - frac) + wav[:, hi] * frac
        else:
            result = wav

        info = f"AudioVideoSync({method}): {current_dur:.2f}s → {target_dur:.2f}s ({target_samples} samples)"
        return (numpy_to_audio(result, sr), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_AudioTrim": S42CF_AudioTrim,
    "S42CF_BeatSnap": S42CF_BeatSnap,
    "S42CF_AudioVideoSync": S42CF_AudioVideoSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_AudioTrim": "🔊 S42 CutFlow Audio Trim",
    "S42CF_BeatSnap": "🥁 S42 CutFlow Beat Snap",
    "S42CF_AudioVideoSync": "🔗 S42 CutFlow Audio-Video Sync",
}
