"""
S42 CutFlow — Clip Operations
==============================
Core clip manipulation: trim, split, loop, freeze, extract, insert, reverse.
All work on IMAGE batches [B, H, W, C] float32 0-1.

Python 3.12 | ComfyUI Portable | torch + numpy
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple

from .cf_utils import ensure_rgb, match_resolution, apply_easing, convert_fps, CHUNK_SIZE

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Clip Ops"


class S42CF_SmartTrim:
    """Trim a clip to in/out points with frame or timecode precision."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip as IMAGE batch [B,H,W,C]. Connect from any video loader or upstream node."}),
                "trim_unit": (["frames", "seconds"], {
                    "default": "frames",
                    "tooltip": "Unit for in/out points.\n'frames' = exact frame indices (0-based).\n'seconds' = time-based (uses fps parameter)."
                }),
                "in_point": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0, "step": 1.0,
                    "tooltip": "Start point. In 'frames' mode: frame index (0 = first). In 'seconds' mode: time from start."
                }),
                "out_point": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 99999.0, "step": 1.0,
                    "tooltip": "End point (inclusive). Set to -1 to trim to end of clip."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Frames per second. Used when trim_unit='seconds' to convert time to frame indices."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, trim_unit, in_point, out_point, fps):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        if trim_unit == "seconds":
            in_f = int(round(in_point * fps))
            out_f = int(round(out_point * fps)) if out_point >= 0 else n - 1
        else:
            in_f = int(in_point)
            out_f = int(out_point) if out_point >= 0 else n - 1
        in_f = max(0, min(in_f, n - 1))
        out_f = max(in_f, min(out_f, n - 1))
        result = clip[in_f:out_f + 1]
        info = f"SmartTrim: frames {in_f}–{out_f} of {n} → {result.shape[0]} frames"
        return (result, int(result.shape[0]), info)


class S42CF_MultiSplit:
    """Split a clip at multiple points into segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to split into segments."}),
                "split_points": ("STRING", {
                    "default": "24, 48",
                    "tooltip": "Comma-separated frame indices where to split. E.g. '24, 48' splits into 3 segments: [0-23], [24-47], [48-end]."
                }),
                "segment_index": ("INT", {
                    "default": 0, "min": 0, "max": 99,
                    "tooltip": "Which segment to output (0-based). Segment 0 is before first split point."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("segment", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, split_points, segment_index):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        points = sorted(set(int(x.strip()) for x in split_points.split(",") if x.strip().isdigit()))
        points = [p for p in points if 0 < p < n]
        boundaries = [0] + points + [n]
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i + 1]))
        idx = min(segment_index, len(segments) - 1)
        start, end = segments[idx]
        result = clip[start:end]
        seg_info = " | ".join(f"seg{i}:[{s}-{e}]({e-s}f)" for i, (s, e) in enumerate(segments))
        info = f"MultiSplit: {len(segments)} segments. Output seg{idx} [{start}-{end}] = {result.shape[0]}f\n{seg_info}"
        return (result, int(result.shape[0]), info)


class S42CF_LoopBounce:
    """Loop a clip N times with optional ping-pong and crossfade."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to loop."}),
                "loops": ("INT", {
                    "default": 2, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Total number of times the clip plays (1 = no loop, just passthrough)."
                }),
                "mode": (["loop", "ping_pong"], {
                    "default": "loop",
                    "tooltip": "'loop' = repeat forward. 'ping_pong' = forward then reverse alternating."
                }),
                "crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 60, "step": 1,
                    "tooltip": "Number of frames to crossfade at each loop seam. 0 = hard cut."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, loops, mode, crossfade_frames):
        clip = ensure_rgb(clip)
        if loops <= 1:
            return (clip, int(clip.shape[0]), "LoopBounce: passthrough (loops=1)")

        parts = []
        for i in range(loops):
            if mode == "ping_pong" and i % 2 == 1:
                parts.append(clip.flip(0))
            else:
                parts.append(clip.clone())

        if crossfade_frames > 0:
            merged = [parts[0]]
            for i in range(1, len(parts)):
                cf = min(crossfade_frames, merged[-1].shape[0], parts[i].shape[0])
                if cf > 0:
                    tail = merged[-1][-cf:]
                    head = parts[i][:cf]
                    blended = []
                    for j in range(cf):
                        t = j / max(cf - 1, 1)
                        blended.append(tail[j] * (1 - t) + head[j] * t)
                    merged[-1] = merged[-1][:-cf]
                    merged.append(torch.stack(blended))
                    merged.append(parts[i][cf:])
                else:
                    merged.append(parts[i])
            result = torch.cat([p for p in merged if p.shape[0] > 0], dim=0)
        else:
            result = torch.cat(parts, dim=0)

        info = f"LoopBounce: {mode} x{loops}, crossfade={crossfade_frames}f → {result.shape[0]} frames"
        return (result, int(result.shape[0]), info)


class S42CF_FrameHold:
    """Hold a specific frame for N frames (freeze frame effect)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Source clip to extract freeze frame from."}),
                "frame_source": (["first", "last", "middle", "custom"], {
                    "default": "first",
                    "tooltip": "Which frame to freeze. 'custom' uses the frame_index parameter."
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": "Frame index to freeze (only used when frame_source='custom')."
                }),
                "hold_frames": ("INT", {
                    "default": 24, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Number of frames to hold. At 24fps: 24=1sec, 48=2sec."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, frame_source, frame_index, hold_frames):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        if frame_source == "first":
            idx = 0
        elif frame_source == "last":
            idx = n - 1
        elif frame_source == "middle":
            idx = n // 2
        else:
            idx = min(frame_index, n - 1)
        frame = clip[idx:idx + 1]
        result = frame.repeat(hold_frames, 1, 1, 1)
        info = f"FrameHold: frame {idx} held for {hold_frames} frames"
        return (result, hold_frames, info)


class S42CF_FrameExtract:
    """Extract specific frames by index, range, or interval."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Source clip."}),
                "mode": (["indices", "interval", "range"], {
                    "default": "interval",
                    "tooltip": "'indices' = specific frame numbers from index_list.\n'interval' = every Nth frame.\n'range' = continuous range."
                }),
                "index_list": ("STRING", {
                    "default": "0, 12, 24",
                    "tooltip": "Comma-separated frame indices (used in 'indices' mode). E.g. '0, 12, 24, 36'."
                }),
                "interval": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": "Extract every Nth frame (used in 'interval' mode). E.g. 4 = frames 0,4,8,12..."
                }),
                "range_start": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "Start frame for 'range' mode."}),
                "range_end": ("INT", {"default": -1, "min": -1, "max": 99999, "tooltip": "End frame for 'range' mode. -1 = end of clip."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, index_list, interval, range_start, range_end):
        clip = ensure_rgb(clip)
        n = clip.shape[0]
        if mode == "indices":
            indices = []
            for x in index_list.split(","):
                x = x.strip()
                if x.isdigit():
                    idx = int(x)
                    if 0 <= idx < n:
                        indices.append(idx)
            if not indices:
                indices = [0]
            result = clip[indices]
        elif mode == "interval":
            indices = list(range(0, n, max(1, interval)))
            result = clip[indices]
        else:
            rs = max(0, min(range_start, n - 1))
            re_ = n if range_end < 0 else min(range_end + 1, n)
            result = clip[rs:re_]
        info = f"FrameExtract({mode}): {n} → {result.shape[0]} frames"
        return (result, int(result.shape[0]), info)


class S42CF_FrameInsert:
    """Insert frames at a specific index within a clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Main clip to insert into."}),
                "insert_frames": ("IMAGE", {"tooltip": "Frames to insert. Will be resolution-matched to clip."}),
                "insert_at": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": "Frame index where insertion begins. 0 = prepend. Use frame_count of clip to append."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, insert_frames, insert_at):
        clip = ensure_rgb(clip)
        insert_frames = ensure_rgb(insert_frames)
        _, insert_frames = match_resolution(clip, insert_frames)
        n = clip.shape[0]
        idx = max(0, min(insert_at, n))
        parts = []
        if idx > 0:
            parts.append(clip[:idx])
        parts.append(insert_frames)
        if idx < n:
            parts.append(clip[idx:])
        result = torch.cat(parts, dim=0)
        info = f"FrameInsert: inserted {insert_frames.shape[0]}f at index {idx}. {n} → {result.shape[0]} frames"
        return (result, int(result.shape[0]), info)


class S42CF_ClipReverse:
    """Reverse a clip with optional speed modification."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to reverse."}),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Playback speed multiplier after reversing.\n1.0 = normal speed reverse.\n0.5 = slow-motion reverse (frame blending).\n2.0 = fast reverse (frame skipping)."
                }),
                "interpolation": (["duplicate", "blend"], {
                    "default": "blend",
                    "tooltip": "Frame interpolation for non-1x speeds.\n'blend' = smooth linear interpolation between frames.\n'duplicate' = nearest-frame, faster but jittery."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, speed, interpolation):
        clip = ensure_rgb(clip)
        reversed_clip = clip.flip(0)
        if abs(speed - 1.0) > 0.01:
            n = reversed_clip.shape[0]
            new_n = max(1, round(n / speed))
            if interpolation == "blend":
                out = []
                for i in range(new_n):
                    src = i * speed
                    lo = min(int(src), n - 1)
                    hi = min(lo + 1, n - 1)
                    frac = src - int(src)
                    out.append(reversed_clip[lo] * (1 - frac) + reversed_clip[hi] * frac)
                result = torch.stack(out)
            else:
                indices = [min(int(i * speed), n - 1) for i in range(new_n)]
                result = reversed_clip[indices]
        else:
            result = reversed_clip
        info = f"ClipReverse: {clip.shape[0]}f reversed at {speed}x → {result.shape[0]}f ({interpolation})"
        return (result, int(result.shape[0]), info)


# ── Registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42CF_SmartTrim": S42CF_SmartTrim,
    "S42CF_MultiSplit": S42CF_MultiSplit,
    "S42CF_LoopBounce": S42CF_LoopBounce,
    "S42CF_FrameHold": S42CF_FrameHold,
    "S42CF_FrameExtract": S42CF_FrameExtract,
    "S42CF_FrameInsert": S42CF_FrameInsert,
    "S42CF_ClipReverse": S42CF_ClipReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_SmartTrim": "✂ S42 CutFlow Smart Trim",
    "S42CF_MultiSplit": "✂ S42 CutFlow Multi Split",
    "S42CF_LoopBounce": "🔁 S42 CutFlow Loop Bounce",
    "S42CF_FrameHold": "⏸ S42 CutFlow Frame Hold",
    "S42CF_FrameExtract": "📋 S42 CutFlow Frame Extract",
    "S42CF_FrameInsert": "📥 S42 CutFlow Frame Insert",
    "S42CF_ClipReverse": "⏪ S42 CutFlow Clip Reverse",
}
