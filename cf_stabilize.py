"""
S42 CutFlow — Video Stabilization
====================================
Software stabilization using OpenCV feature tracking.
Requires opencv-python. Graceful error if missing.

Python 3.12 | ComfyUI Portable | OpenCV + numpy + torch
"""

import torch
import numpy as np
import logging

from .cf_utils import ensure_rgb, frames_to_np, np_to_frames, CV2_AVAILABLE

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Composition"

if CV2_AVAILABLE:
    import cv2


class S42CF_Stabilize:
    """Software video stabilization via optical flow tracking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Shaky video clip to stabilize."}),
                "smoothing": ("INT", {
                    "default": 15, "min": 3, "max": 60, "step": 1,
                    "tooltip": "Smoothing window radius in frames. Larger = smoother but more cropping. 15 = moderate stabilization."
                }),
                "crop_mode": (["auto_crop", "fill_border", "none"], {
                    "default": "auto_crop",
                    "tooltip": "'auto_crop' = zoom in to hide black borders (recommended).\n'fill_border' = fill borders by extending edge pixels.\n'none' = leave black borders visible."
                }),
                "max_shift": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01,
                    "tooltip": "Maximum allowed shift as fraction of frame. 0.1 = up to 10% movement correction. Prevents over-correction."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, smoothing, crop_mode, max_shift):
        if not CV2_AVAILABLE:
            logger.error("Stabilize requires opencv-python. Install: pip install opencv-python")
            return (clip, int(clip.shape[0]), "ERROR: opencv-python not installed")

        clip = ensure_rgb(clip)
        frames_np = frames_to_np(clip)
        n, h, w, c = frames_np.shape
        max_px_x = int(w * max_shift)
        max_px_y = int(h * max_shift)

        prev_gray = cv2.cvtColor(frames_np[0], cv2.COLOR_RGB2GRAY)
        transforms = [(0.0, 0.0, 0.0)]

        for i in range(1, n):
            curr_gray = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

            if pts is not None and len(pts) > 0:
                pts_new, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
                good_old = pts[status.flatten() == 1]
                good_new = pts_new[status.flatten() == 1]

                if len(good_old) >= 4:
                    mat, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                    if mat is not None:
                        dx = np.clip(mat[0, 2], -max_px_x, max_px_x)
                        dy = np.clip(mat[1, 2], -max_px_y, max_px_y)
                        da = np.arctan2(mat[1, 0], mat[0, 0])
                        transforms.append((dx, dy, da))
                    else:
                        transforms.append((0.0, 0.0, 0.0))
                else:
                    transforms.append((0.0, 0.0, 0.0))
            else:
                transforms.append((0.0, 0.0, 0.0))

            prev_gray = curr_gray

        cum_x = np.cumsum([t[0] for t in transforms])
        cum_y = np.cumsum([t[1] for t in transforms])
        cum_a = np.cumsum([t[2] for t in transforms])

        kernel = np.ones(smoothing * 2 + 1) / (smoothing * 2 + 1)
        smooth_x = np.convolve(cum_x, kernel, mode='same')
        smooth_y = np.convolve(cum_y, kernel, mode='same')
        smooth_a = np.convolve(cum_a, kernel, mode='same')

        diff_x = smooth_x - cum_x
        diff_y = smooth_y - cum_y
        diff_a = smooth_a - cum_a

        stabilized = []
        max_border = 0
        for i in range(n):
            dx, dy, da = diff_x[i], diff_y[i], diff_a[i]
            cos_a, sin_a = np.cos(da), np.sin(da)
            mat = np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float64)
            result = cv2.warpAffine(frames_np[i], mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            stabilized.append(result)
            max_border = max(max_border, abs(dx), abs(dy))

        stabilized = np.stack(stabilized)

        if crop_mode == "auto_crop":
            crop = int(max_border) + 5
            crop = min(crop, min(h, w) // 4)
            if crop > 0:
                cropped = stabilized[:, crop:h - crop, crop:w - crop]
                resized = []
                for frame in cropped:
                    resized.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4))
                stabilized = np.stack(resized)

        result = np_to_frames(stabilized)
        info = f"Stabilize: smoothing={smoothing}, max_shift={max_shift:.0%}, crop={crop_mode}, max_border={max_border:.1f}px"
        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_Stabilize": S42CF_Stabilize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_Stabilize": "📐 S42 CutFlow Stabilize",
}
