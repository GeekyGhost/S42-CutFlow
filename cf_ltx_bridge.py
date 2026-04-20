"""
S42 CutFlow — LTX 2.3 Bridge Nodes (Expanded)
================================================
Utility nodes that prepare data FOR and process data FROM
LTX 2.3 video generation pipelines. These don't replace the
native LTX ComfyUI nodes — they bridge CutFlow's editing
capabilities with LTX's generation capabilities.

Nodes:
  GuideFramePrep    — Extract first/last/key frames for LTXVAddGuide
  LTXFrameCalc      — Calculate valid LTX frame counts (8n+1 rule)
  SubjectIsolate    — BG remove + prepare subject plate for replacement
  BackgroundPlate   — Extract/generate clean bg for compositing
  VideoSegmentPrep  — Split video into LTX-compatible length segments
  AudioCondPrep     — Prepare audio for LTX audio conditioning
  LTXPostProcess    — Post-process LTX output (crop guide artifacts, clean)
  SceneDescriber    — Generate text descriptions from video frames for prompting

Python 3.12 | ComfyUI Portable | torch + numpy + PIL
"""

import torch
import numpy as np
import math
import logging
from typing import Optional, Tuple

from .cf_utils import (
    ensure_rgb, resize_frames, frame_to_pil, pil_to_frame,
    frames_to_np, np_to_frames, get_audio_numpy, numpy_to_audio,
    PIL_AVAILABLE, CV2_AVAILABLE, SCIPY_AVAILABLE
)

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/LTX Bridge"


class S42CF_GuideFramePrep:
    """Extract and prepare guide frames for LTX 2.3 image-to-video conditioning.
    Outputs frames ready to connect to LTXVAddGuide + LTXVPreprocess nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {
                    "tooltip": "Source video or image batch. Can be AI-generated frames, photos, or existing video."}),
                "mode": (["first_last", "first_only", "last_only", "keyframes", "evenly_spaced"], {
                    "default": "first_last",
                    "tooltip": "Which frames to extract as LTX guide frames:\n"
                               "'first_last' = first + last frame (standard FLF2V workflow).\n"
                               "'first_only' = just the first frame (simpler I2V).\n"
                               "'last_only' = just the last frame.\n"
                               "'keyframes' = extract at specified indices.\n"
                               "'evenly_spaced' = N frames evenly distributed."
                }),
                "keyframe_indices": ("STRING", {
                    "default": "0, -1",
                    "tooltip": "Comma-separated frame indices for 'keyframes' mode. Negative indices count from end. E.g. '0, 40, -1'."}),
                "num_guides": ("INT", {
                    "default": 3, "min": 2, "max": 10, "step": 1,
                    "tooltip": "Number of guide frames for 'evenly_spaced' mode."}),
                "target_width": ("INT", {
                    "default": 768, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Output width. LTX 2.3 supports up to 1920. Common: 768, 1024, 1280, 1920."}),
                "target_height": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Output height. Must match your LTX empty latent dimensions."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "guide_indices", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, source, mode, keyframe_indices, num_guides, target_width, target_height):
        source = ensure_rgb(source)
        n = source.shape[0]

        if mode == "first_last":
            indices = [0, n - 1]
        elif mode == "first_only":
            indices = [0]
        elif mode == "last_only":
            indices = [n - 1]
        elif mode == "keyframes":
            raw = [int(x.strip()) for x in keyframe_indices.split(",") if x.strip().lstrip('-').isdigit()]
            indices = [i if i >= 0 else n + i for i in raw]
            indices = [max(0, min(i, n - 1)) for i in indices]
        elif mode == "evenly_spaced":
            indices = [int(i * (n - 1) / max(num_guides - 1, 1)) for i in range(num_guides)]
        else:
            indices = [0]

        indices = sorted(set(indices))
        frames = source[indices]

        if frames.shape[1] != target_height or frames.shape[2] != target_width:
            frames = resize_frames(frames, target_height, target_width)

        first_frame = frames[0:1]
        last_frame = frames[-1:] if len(indices) > 1 else frames[0:1]
        idx_str = ", ".join(str(i) for i in indices)

        info = (f"GuideFramePrep({mode}): extracted {len(indices)} guides at [{idx_str}] "
                f"from {n}f source → {target_width}x{target_height}")

        return (first_frame, last_frame, idx_str, info)


class S42CF_LTXFrameCalc:
    """Calculate valid LTX 2.3 frame counts.
    LTX requires frame counts = 8n + 1 (e.g. 9, 17, 25, 33, ... 121)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "desired_duration": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "Desired video duration in seconds."}),
                "fps": ("INT", {
                    "default": 24, "min": 8, "max": 60, "step": 1,
                    "tooltip": "Target frame rate. LTX 2.3 default is 24fps."}),
                "round_mode": (["round_up", "round_down", "nearest"], {
                    "default": "nearest",
                    "tooltip": "How to round to valid 8n+1 frame count.\n'round_up' = never shorter than desired.\n'round_down' = never longer.\n'nearest' = closest match."}),
                "max_frames": ("INT", {
                    "default": 257, "min": 9, "max": 513, "step": 8,
                    "tooltip": "Maximum frame count. 121=~5s, 257=~10.7s, 513=~21.4s at 24fps. Higher needs more VRAM."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame_count", "fps", "duration_seconds", "fps_float", "info")
    FUNCTION = "calculate"
    CATEGORY = CATEGORY

    def calculate(self, desired_duration, fps, round_mode, max_frames):
        raw_frames = desired_duration * fps

        if round_mode == "round_up":
            n = math.ceil((raw_frames - 1) / 8) * 8 + 1
        elif round_mode == "round_down":
            n = math.floor((raw_frames - 1) / 8) * 8 + 1
        else:
            n = round((raw_frames - 1) / 8) * 8 + 1

        n = max(9, min(n, max_frames))
        if (n - 1) % 8 != 0:
            n = round((n - 1) / 8) * 8 + 1

        actual_duration = n / fps
        fps_float = float(fps)

        info = (f"LTXFrameCalc: {desired_duration:.1f}s × {fps}fps = {raw_frames:.0f}f "
                f"→ {n}f (8×{(n-1)//8}+1) = {actual_duration:.2f}s\n"
                f"VRAM estimate: ~{n * 0.04:.1f}GB at 768x512")

        return (n, fps, round(actual_duration, 3), fps_float, info)


class S42CF_SubjectIsolate:
    """Isolate subject from video for replacement workflows.
    Outputs subject (with alpha), background plate, and mask.
    Connect subject to LTX guide frame for subject-preserved regeneration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Source video with subject to isolate."}),
                "subject_mask": ("MASK", {
                    "tooltip": "Subject mask from Background Remover. White = subject, Black = background."}),
                "output_mode": (["subject_on_black", "subject_on_white", "subject_rgba",
                                  "background_only", "background_inpaint_simple"], {
                    "default": "subject_on_black",
                    "tooltip": "'subject_on_black' = subject composited on black bg (for LTX guide).\n"
                               "'subject_on_white' = subject on white bg.\n"
                               "'subject_rgba' = RGBA with alpha channel.\n"
                               "'background_only' = everything except the subject.\n"
                               "'background_inpaint_simple' = bg with subject area filled by edge extension."
                }),
                "expand_mask": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Expand the subject mask by N pixels for cleaner edges."}),
                "feather": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Feather/blur the mask edges for smooth compositing."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("subject_plate", "background_plate", "refined_mask", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, video, subject_mask, output_mode, expand_mask, feather):
        video = ensure_rgb(video)
        n, h, w, c = video.shape

        if subject_mask.ndim == 2:
            masks = subject_mask.unsqueeze(0).repeat(n, 1, 1)
        elif subject_mask.shape[0] == 1:
            masks = subject_mask.repeat(n, 1, 1)
        else:
            masks = subject_mask[:n]

        if masks.shape[1] != h or masks.shape[2] != w:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)

        if expand_mask > 0 and CV2_AVAILABLE:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_mask * 2 + 1, expand_mask * 2 + 1))
            expanded = []
            for i in range(n):
                m = (masks[i].cpu().numpy() * 255).astype(np.uint8)
                m = cv2.dilate(m, k, iterations=1)
                expanded.append(torch.from_numpy(m.astype(np.float32) / 255.0))
            masks = torch.stack(expanded)

        if feather > 0 and CV2_AVAILABLE:
            import cv2
            ks = int(feather * 2) | 1
            feathered = []
            for i in range(n):
                m = (masks[i].cpu().numpy() * 255).astype(np.uint8)
                m = cv2.GaussianBlur(m, (ks, ks), feather / 3.0)
                feathered.append(torch.from_numpy(m.astype(np.float32) / 255.0))
            masks = torch.stack(feathered)

        mask_3d = masks.unsqueeze(-1)
        inv_mask_3d = 1.0 - mask_3d

        if output_mode == "subject_on_black":
            subject_plate = video * mask_3d
        elif output_mode == "subject_on_white":
            subject_plate = video * mask_3d + inv_mask_3d
        elif output_mode == "subject_rgba":
            subject_plate = torch.cat([video * mask_3d, masks.unsqueeze(-1)], dim=-1)
        else:
            subject_plate = video * mask_3d

        background_plate = video * inv_mask_3d

        if output_mode == "background_inpaint_simple" and CV2_AVAILABLE:
            import cv2
            bg_list = []
            for i in range(n):
                frame_uint8 = (video[i].cpu().numpy() * 255).astype(np.uint8)
                inpaint_mask = (masks[i].cpu().numpy() * 255).astype(np.uint8)
                inpainted = cv2.inpaint(frame_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA)
                bg_list.append(torch.from_numpy(inpainted.astype(np.float32) / 255.0))
            background_plate = torch.stack(bg_list)

        info = (f"SubjectIsolate({output_mode}): {n}f, expand={expand_mask}, "
                f"feather={feather}, mask range=[{masks.min():.2f}, {masks.max():.2f}]")

        return (subject_plate, background_plate, masks, info)


class S42CF_BackgroundPlate:
    """Generate a clean background plate for compositing or LTX regeneration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["solid_color", "gradient_h", "gradient_v", "from_video_median", "from_image"], {
                    "default": "solid_color",
                    "tooltip": "'solid_color' = flat color fill.\n"
                               "'gradient_h' = horizontal gradient.\n"
                               "'gradient_v' = vertical gradient.\n"
                               "'from_video_median' = compute median background from video.\n"
                               "'from_image' = use a provided image, scaled to canvas."
                }),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8, "tooltip": "Background width."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Background height."}),
                "frames": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Number of frames to generate. Match your video length for compositing."}),
                "color1_hex": ("STRING", {"default": "#1a1a2e", "tooltip": "Primary color (hex). Used for solid and gradient start."}),
                "color2_hex": ("STRING", {"default": "#16213e", "tooltip": "Secondary color (hex). Used for gradient end."}),
            },
            "optional": {
                "source_video": ("IMAGE", {"tooltip": "Source video for 'from_video_median' mode. Computes temporal median per-pixel."}),
                "source_image": ("IMAGE", {"tooltip": "Source image for 'from_image' mode. First frame used, scaled to canvas."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("background", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def _hex(self, s):
        s = s.strip().lstrip("#")
        if len(s) == 3: s = "".join(c * 2 for c in s)
        try:
            return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
        except Exception:
            return (0.0, 0.0, 0.0)

    def process(self, mode, width, height, frames, color1_hex, color2_hex,
                source_video=None, source_image=None):
        c1 = self._hex(color1_hex)
        c2 = self._hex(color2_hex)

        if mode == "solid_color":
            bg = np.full((height, width, 3), c1, dtype=np.float32)
        elif mode == "gradient_h":
            t = np.linspace(0, 1, width, dtype=np.float32)[np.newaxis, :, np.newaxis]
            c1a, c2a = np.array(c1), np.array(c2)
            bg = (c1a * (1 - t) + c2a * t).astype(np.float32)
            bg = np.broadcast_to(bg, (height, width, 3)).copy()
        elif mode == "gradient_v":
            t = np.linspace(0, 1, height, dtype=np.float32)[:, np.newaxis, np.newaxis]
            c1a, c2a = np.array(c1), np.array(c2)
            bg = (c1a * (1 - t) + c2a * t).astype(np.float32)
            bg = np.broadcast_to(bg, (height, width, 3)).copy()
        elif mode == "from_video_median" and source_video is not None:
            vid = ensure_rgb(source_video)
            median = torch.median(vid, dim=0).values
            bg = resize_frames(median.unsqueeze(0), height, width)[0].cpu().numpy()
        elif mode == "from_image" and source_image is not None:
            img = ensure_rgb(source_image)
            bg = resize_frames(img[0:1], height, width)[0].cpu().numpy()
        else:
            bg = np.full((height, width, 3), c1, dtype=np.float32)

        result = torch.from_numpy(bg).unsqueeze(0).repeat(frames, 1, 1, 1)
        info = f"BackgroundPlate({mode}): {width}x{height}, {frames}f"
        return (result, info)


class S42CF_VideoSegmentPrep:
    """Split video into LTX-compatible segments with overlap for seamless generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Long video to split into LTX-compatible segments."}),
                "segment_frames": ("INT", {
                    "default": 121, "min": 9, "max": 513, "step": 8,
                    "tooltip": "Frames per segment. Must be 8n+1. 121=5s, 257=10.7s at 24fps."}),
                "overlap_frames": ("INT", {
                    "default": 9, "min": 1, "max": 33, "step": 8,
                    "tooltip": "Overlap between segments for smooth transitions. Should be 8n+1."}),
                "segment_index": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "Which segment to output (0-based)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("segment", "first_frame", "last_frame", "segment_count", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, video, segment_frames, overlap_frames, segment_index):
        video = ensure_rgb(video)
        n = video.shape[0]
        step = max(1, segment_frames - overlap_frames)

        segments = []
        start = 0
        while start < n:
            end = min(start + segment_frames, n)
            segments.append((start, end))
            if end >= n:
                break
            start += step

        idx = min(segment_index, len(segments) - 1)
        s, e = segments[idx]
        segment = video[s:e]
        first_frame = segment[0:1]
        last_frame = segment[-1:]

        seg_info = " | ".join(f"seg{i}:[{s_}-{e_}]" for i, (s_, e_) in enumerate(segments))
        info = (f"VideoSegmentPrep: {n}f → {len(segments)} segments of {segment_frames}f "
                f"(overlap={overlap_frames})\nOutput seg{idx}: [{s}-{e}] = {segment.shape[0]}f\n{seg_info}")

        return (segment, first_frame, last_frame, len(segments), int(segment.shape[0]), info)


class S42CF_AudioCondPrep:
    """Prepare audio for LTX 2.3 audio conditioning.
    Handles trimming, silence detection, and volume normalization."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input for LTX conditioning."}),
                "target_frames": ("INT", {
                    "default": 121, "min": 9, "max": 513,
                    "tooltip": "Target video frame count. Audio will be trimmed/padded to match."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60,
                    "tooltip": "Video FPS for audio duration calculation."}),
                "normalize": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Normalize audio to -3dBFS peak. Recommended for consistent LTX results."}),
                "fade_in_ms": ("INT", {"default": 50, "min": 0, "max": 2000, "step": 10,
                    "tooltip": "Fade-in duration in milliseconds. Prevents click artifacts."}),
                "fade_out_ms": ("INT", {"default": 100, "min": 0, "max": 2000, "step": 10,
                    "tooltip": "Fade-out duration in milliseconds."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration_seconds", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, audio, target_frames, fps, normalize, fade_in_ms, fade_out_ms):
        wav, sr = get_audio_numpy(audio)
        target_dur = target_frames / fps
        target_samples = int(target_dur * sr)

        if wav.shape[-1] > target_samples:
            wav = wav[:, :target_samples]
        elif wav.shape[-1] < target_samples:
            pad = np.zeros((wav.shape[0], target_samples - wav.shape[-1]), dtype=wav.dtype)
            wav = np.concatenate([wav, pad], axis=-1)

        if normalize == "yes":
            peak = np.abs(wav).max()
            if peak > 0:
                target_peak = 10 ** (-3 / 20)
                wav = wav * (target_peak / peak)

        if fade_in_ms > 0:
            fade_samples = min(int(fade_in_ms * sr / 1000), wav.shape[-1])
            fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
            wav[:, :fade_samples] *= fade

        if fade_out_ms > 0:
            fade_samples = min(int(fade_out_ms * sr / 1000), wav.shape[-1])
            fade = np.linspace(1, 0, fade_samples, dtype=np.float32)
            wav[:, -fade_samples:] *= fade

        duration = wav.shape[-1] / sr
        info = (f"AudioCondPrep: {duration:.2f}s ({wav.shape[-1]} samples @ {sr}Hz) "
                f"for {target_frames}f @ {fps}fps | normalize={'yes' if normalize == 'yes' else 'no'}")

        return (numpy_to_audio(wav, sr), round(duration, 3), info)


class S42CF_LTXPostProcess:
    """Post-process LTX 2.3 output — crop guide artifacts, clean, prepare for editing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Video output from LTX VAE decode (or RTX upscale)."}),
                "crop_last_frames": ("INT", {
                    "default": 2, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Remove N frames from end. LTXVAddGuide can cause artifacts in final frames. Default 2 matches LTXVCropGuides behavior."}),
                "crop_first_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Remove N frames from start. Usually 0 unless using special conditioning."}),
                "auto_brightness": (["none", "subtle", "moderate"], {
                    "default": "none",
                    "tooltip": "'none' = no change. 'subtle' = slight exposure correction. 'moderate' = auto-levels."}),
                "denoise_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Light temporal denoising. 0 = off. 0.3 = subtle. Helps with LTX generation noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("video", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, video, crop_last_frames, crop_first_frames, auto_brightness, denoise_strength):
        video = ensure_rgb(video)
        n_orig = video.shape[0]

        start = crop_first_frames
        end = n_orig - crop_last_frames if crop_last_frames > 0 else n_orig
        end = max(start + 1, end)
        video = video[start:end]

        if auto_brightness == "subtle":
            mean_b = video.mean()
            if mean_b < 0.35:
                video = (video + 0.05).clamp(0, 1)
            elif mean_b > 0.7:
                video = (video - 0.03).clamp(0, 1)
        elif auto_brightness == "moderate":
            for i in range(video.shape[0]):
                frame = video[i]
                lo, hi = frame.min(), frame.max()
                if hi - lo > 0.1:
                    video[i] = ((frame - lo) / (hi - lo)).clamp(0, 1)

        if denoise_strength > 0 and video.shape[0] > 2:
            smoothed = video.clone()
            for i in range(1, video.shape[0] - 1):
                smoothed[i] = video[i] * (1 - denoise_strength) + \
                              (video[i - 1] + video[i + 1]) / 2 * denoise_strength
            video = smoothed

        info = (f"LTXPostProcess: {n_orig}f → {video.shape[0]}f "
                f"(cropped first={crop_first_frames}, last={crop_last_frames}) "
                f"brightness={auto_brightness}, denoise={denoise_strength}")

        return (video, int(video.shape[0]), info)


class S42CF_SceneDescriber:
    """Extract visual info from frames to help write LTX prompts.
    Analyzes color palette, motion, composition for prompt engineering."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Video or image to analyze for prompt engineering."}),
                "detail_level": (["basic", "detailed"], {
                    "default": "basic",
                    "tooltip": "'basic' = color + brightness + motion summary.\n'detailed' = adds composition analysis and suggested prompt elements."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_description", "prompt_suggestions")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, video, detail_level):
        video = ensure_rgb(video)
        n, h, w, c = video.shape
        first = video[0].cpu().numpy()
        last = video[-1].cpu().numpy() if n > 1 else first

        mean_rgb = first.mean(axis=(0, 1))
        brightness = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]
        contrast = first.std()

        r_dom = mean_rgb[0] > mean_rgb[1] and mean_rgb[0] > mean_rgb[2]
        g_dom = mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]
        b_dom = mean_rgb[2] > mean_rgb[0] and mean_rgb[2] > mean_rgb[1]

        if brightness > 0.7:
            light_desc = "bright, high-key lighting"
        elif brightness > 0.4:
            light_desc = "medium exposure, balanced lighting"
        else:
            light_desc = "dark, low-key lighting"

        if r_dom:
            color_desc = "warm tones (red/orange dominant)"
        elif g_dom:
            color_desc = "natural tones (green dominant)"
        elif b_dom:
            color_desc = "cool tones (blue dominant)"
        else:
            color_desc = "neutral/balanced colors"

        motion_desc = "static image"
        if n > 1:
            diff = torch.abs(video[1:] - video[:-1]).mean().item()
            if diff > 0.05:
                motion_desc = "high motion/action"
            elif diff > 0.02:
                motion_desc = "moderate motion"
            elif diff > 0.005:
                motion_desc = "subtle motion/slow movement"
            else:
                motion_desc = "near-static/minimal motion"

        aspect = w / h
        if aspect > 1.5:
            comp_desc = "widescreen cinematic framing"
        elif aspect < 0.7:
            comp_desc = "vertical/portrait framing"
        else:
            comp_desc = "standard framing"

        scene = (f"Scene: {w}x{h} ({comp_desc}), {n} frames\n"
                 f"Lighting: {light_desc} (brightness={brightness:.2f}, contrast={contrast:.2f})\n"
                 f"Color: {color_desc} (R={mean_rgb[0]:.2f} G={mean_rgb[1]:.2f} B={mean_rgb[2]:.2f})\n"
                 f"Motion: {motion_desc}")

        prompt_parts = []
        if brightness > 0.7:
            prompt_parts.append("bright lighting")
        elif brightness < 0.3:
            prompt_parts.append("dramatic low-key lighting")
        prompt_parts.append("professional lighting")

        if "high motion" in motion_desc:
            prompt_parts.append("dynamic movement, action shot")
        elif "subtle" in motion_desc:
            prompt_parts.append("gentle movement, smooth animation")

        prompt_parts.append("4k, highly detailed")

        if detail_level == "detailed":
            if contrast > 0.25:
                prompt_parts.append("high contrast")
            prompt_parts.append("cinematic" if aspect > 1.3 else "")

        suggestions = "Suggested prompt elements: " + ", ".join(p for p in prompt_parts if p)

        return (scene, suggestions)


NODE_CLASS_MAPPINGS = {
    "S42CF_GuideFramePrep": S42CF_GuideFramePrep,
    "S42CF_LTXFrameCalc": S42CF_LTXFrameCalc,
    "S42CF_SubjectIsolate": S42CF_SubjectIsolate,
    "S42CF_BackgroundPlate": S42CF_BackgroundPlate,
    "S42CF_VideoSegmentPrep": S42CF_VideoSegmentPrep,
    "S42CF_AudioCondPrep": S42CF_AudioCondPrep,
    "S42CF_LTXPostProcess": S42CF_LTXPostProcess,
    "S42CF_SceneDescriber": S42CF_SceneDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_GuideFramePrep": "🎯 S42 CutFlow Guide Frame Prep",
    "S42CF_LTXFrameCalc": "🔢 S42 CutFlow LTX Frame Calculator",
    "S42CF_SubjectIsolate": "👤 S42 CutFlow Subject Isolate",
    "S42CF_BackgroundPlate": "🏞 S42 CutFlow Background Plate",
    "S42CF_VideoSegmentPrep": "📐 S42 CutFlow Video Segment Prep",
    "S42CF_AudioCondPrep": "🔊 S42 CutFlow Audio Cond Prep",
    "S42CF_LTXPostProcess": "✨ S42 CutFlow LTX Post-Process",
    "S42CF_SceneDescriber": "📝 S42 CutFlow Scene Describer",
}
