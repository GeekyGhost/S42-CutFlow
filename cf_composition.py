"""
S42 CutFlow — Composition & Motion
=====================================
Picture-in-picture, split screen, Ken Burns, crop/pad,
mask wipes, and shape mask generators.

Python 3.12 | ComfyUI Portable | torch + numpy + PIL
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import logging

from .cf_utils import (
    ensure_rgb, match_resolution, resize_frames, frame_to_pil, pil_to_frame,
    parse_keyframe_string, interpolate_keyframes, apply_easing, ALL_EASINGS,
    PIL_AVAILABLE
)

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Composition"


class S42CF_PictureInPicture:
    """Overlay one clip onto another at configurable position/scale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {"tooltip": "Main background clip."}),
                "overlay": ("IMAGE", {"tooltip": "Clip to overlay as picture-in-picture."}),
                "position_x": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Horizontal center of PiP. 0=left, 0.5=center, 1=right."}),
                "position_y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical center of PiP. 0=top, 0.5=center, 1=bottom."}),
                "scale": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale of overlay relative to background. 0.3 = 30% of background size."}),
                "border_width": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Border width around PiP window in pixels."}),
                "border_color": ("STRING", {"default": "#FFFFFF",
                    "tooltip": "Border color (hex). Click the swatch to open color picker."}),
                "corner_radius": ("INT", {"default": 0, "min": 0, "max": 50, "step": 2,
                    "tooltip": "Rounded corner radius for PiP window. 0 = sharp corners."}),
                "shadow": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Drop shadow size. 0 = no shadow."}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "PiP overlay opacity."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, background, overlay, position_x, position_y, scale,
                border_width, border_color,
                corner_radius, shadow, opacity):
        bg = ensure_rgb(background)
        ov = ensure_rgb(overlay)
        n_bg, h, w, c = bg.shape
        n_ov = ov.shape[0]
        n = min(n_bg, n_ov)

        pip_h = max(1, int(h * scale))
        pip_w = max(1, int(w * scale))
        ov_resized = resize_frames(ov[:n], pip_h, pip_w)

        cx = int(position_x * w)
        cy = int(position_y * h)
        x0 = max(0, cx - pip_w // 2)
        y0 = max(0, cy - pip_h // 2)
        x1 = min(w, x0 + pip_w)
        y1 = min(h, y0 + pip_h)
        aw = x1 - x0
        ah = y1 - y0

        from .cf_utils import hex_to_rgb
        br, bg_c, bb = hex_to_rgb(border_color)

        if corner_radius > 0:
            Y, X = np.ogrid[:ah, :aw]
            r = corner_radius
            mask = np.ones((ah, aw), dtype=np.float32)
            for (corner_y, corner_x) in [(r, r), (r, aw - r), (ah - r, r), (ah - r, aw - r)]:
                dist = np.sqrt((X - corner_x) ** 2 + (Y - corner_y) ** 2)
                region = (np.abs(X - corner_x) + r > aw if corner_x > aw // 2 else X < r) & \
                         (np.abs(Y - corner_y) + r > ah if corner_y > ah // 2 else Y < r)
                mask = np.where(region & (dist > r), 0.0, mask)
            alpha_mask = torch.from_numpy(mask).unsqueeze(-1) * opacity
        else:
            alpha_mask = torch.ones(ah, aw, 1) * opacity

        results = []
        for i in range(n):
            frame = bg[i].clone()
            ov_frame = ov_resized[i][:ah, :aw]
            if border_width > 0:
                b = border_width
                bx0 = max(0, x0 - b)
                by0 = max(0, y0 - b)
                bx1 = min(w, x1 + b)
                by1 = min(h, y1 + b)
                border_color_t = torch.tensor([br, bg_c, bb])
                frame[by0:by1, bx0:bx1] = border_color_t
            frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - alpha_mask) + ov_frame * alpha_mask
            results.append(frame)

        result = torch.stack(results)
        info = f"PictureInPicture: scale={scale}, pos=({position_x:.2f},{position_y:.2f}), {n}f"
        return (result, n, info)


class S42CF_SplitScreen:
    """Multi-way split screen layouts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "First clip (top/left)."}),
                "clip_b": ("IMAGE", {"tooltip": "Second clip (bottom/right)."}),
                "layout": (["horizontal_2", "vertical_2", "grid_4", "diagonal"], {
                    "default": "vertical_2",
                    "tooltip": "'horizontal_2' = top/bottom.\n'vertical_2' = left/right.\n'grid_4' = 2x2 grid (needs clip_c and clip_d).\n'diagonal' = diagonal split."
                }),
                "border_width": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Border/gap between panels in pixels."}),
                "border_color": ("STRING", {"default": "#FFFFFF",
                    "tooltip": "Border color (hex). Click the swatch to open color picker."}),
                "split_position": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Where to split. 0.5 = even split. 0.3 = 30/70 split."}),
            },
            "optional": {
                "clip_c": ("IMAGE", {"tooltip": "Third clip (for grid_4 layout)."}),
                "clip_d": ("IMAGE", {"tooltip": "Fourth clip (for grid_4 layout)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip_a, clip_b, layout, border_width, border_color,
                split_position, clip_c=None, clip_d=None):
        a = ensure_rgb(clip_a)
        b = ensure_rgb(clip_b)
        _, a = match_resolution(a, a)
        _, b = match_resolution(a, b)
        n = min(a.shape[0], b.shape[0])
        h, w = a.shape[1], a.shape[2]
        from .cf_utils import hex_to_rgb
        br, bg_v, bb = hex_to_rgb(border_color)
        border_color_t = torch.tensor([br, bg_v, bb])

        results = []
        for i in range(n):
            canvas = torch.zeros(h, w, 3)

            if layout == "vertical_2":
                split_x = int(w * split_position)
                a_panel = resize_frames(a[i:i+1], h, split_x)[0]
                b_panel = resize_frames(b[i:i+1], h, w - split_x - border_width)[0]
                canvas[:, :split_x] = a_panel
                if border_width > 0:
                    canvas[:, split_x:split_x + border_width] = border_color_t
                canvas[:, split_x + border_width:] = b_panel[:, :w - split_x - border_width]

            elif layout == "horizontal_2":
                split_y = int(h * split_position)
                a_panel = resize_frames(a[i:i+1], split_y, w)[0]
                b_panel = resize_frames(b[i:i+1], h - split_y - border_width, w)[0]
                canvas[:split_y] = a_panel
                if border_width > 0:
                    canvas[split_y:split_y + border_width] = border_color_t
                canvas[split_y + border_width:] = b_panel[:h - split_y - border_width]

            elif layout == "grid_4":
                hw, hh = w // 2, h // 2
                bw = border_width
                pw, ph = hw - bw, hh - bw
                clips = [a, b]
                if clip_c is not None:
                    c = ensure_rgb(clip_c)
                    _, c = match_resolution(a, c)
                    clips.append(c)
                else:
                    clips.append(a)
                if clip_d is not None:
                    d = ensure_rgb(clip_d)
                    _, d = match_resolution(a, d)
                    clips.append(d)
                else:
                    clips.append(b)

                idx = min(i, *[cl.shape[0] - 1 for cl in clips])
                panels = [resize_frames(cl[idx:idx+1], ph, pw)[0] for cl in clips]
                canvas[:ph, :pw] = panels[0]
                canvas[:ph, hw + bw:hw + bw + pw] = panels[1][:, :pw]
                canvas[hh + bw:hh + bw + ph, :pw] = panels[2][:ph]
                canvas[hh + bw:hh + bw + ph, hw + bw:hw + bw + pw] = panels[3][:ph, :pw]
                if bw > 0:
                    canvas[:, hw:hw + bw] = border_color_t
                    canvas[hh:hh + bw, :] = border_color_t

            elif layout == "diagonal":
                for y in range(h):
                    split_x = int(w * (y / h))
                    canvas[y, :split_x] = a[min(i, a.shape[0]-1), y, :split_x]
                    canvas[y, split_x:] = b[min(i, b.shape[0]-1), y, split_x:]

            results.append(canvas.clamp(0, 1))

        result = torch.stack(results)
        info = f"SplitScreen({layout}): split={split_position}, {n}f"
        return (result, n, info)


class S42CF_KenBurns:
    """Pan and zoom effect (Ken Burns) on clips or images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Source clip or image. For stills, use ImageToClip first or this will animate a single frame."}),
                "start_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Start crop left edge (0-1 of width)."}),
                "start_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Start crop top edge."}),
                "start_zoom": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "Start zoom level. 1.0 = full view, 2.0 = 2x zoom (50% crop)."}),
                "end_x": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "End crop left edge."}),
                "end_y": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "End crop top edge."}),
                "end_zoom": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 3.0, "step": 0.05,
                    "tooltip": "End zoom level."}),
                "easing": (ALL_EASINGS, {"default": "ease_in_out",
                    "tooltip": "Animation easing curve."}),
                "output_frames": ("INT", {"default": 0, "min": 0, "max": 9999,
                    "tooltip": "Output frame count. 0 = same as input clip length."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, start_x, start_y, start_zoom, end_x, end_y, end_zoom,
                easing, output_frames):
        clip = ensure_rgb(clip)
        n_in, h, w, c = clip.shape
        n_out = output_frames if output_frames > 0 else n_in

        results = []
        for i in range(n_out):
            t = i / max(n_out - 1, 1)
            t_eased = apply_easing(t, easing)

            zoom = start_zoom + (end_zoom - start_zoom) * t_eased
            cx = start_x + (end_x - start_x) * t_eased
            cy = start_y + (end_y - start_y) * t_eased

            crop_w = w / zoom
            crop_h = h / zoom
            x0 = int(cx * (w - crop_w))
            y0 = int(cy * (h - crop_h))
            x0 = max(0, min(x0, int(w - crop_w)))
            y0 = max(0, min(y0, int(h - crop_h)))
            x1 = min(w, x0 + int(crop_w))
            y1 = min(h, y0 + int(crop_h))

            src_idx = min(int(i * n_in / n_out), n_in - 1)
            cropped = clip[src_idx, y0:y1, x0:x1]
            resized = F.interpolate(
                cropped.unsqueeze(0).permute(0, 3, 1, 2),
                size=(h, w), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).squeeze(0)
            results.append(resized)

        result = torch.stack(results)
        info = f"KenBurns: zoom {start_zoom:.1f}→{end_zoom:.1f}, pan ({start_x:.2f},{start_y:.2f})→({end_x:.2f},{end_y:.2f}), {n_out}f"
        return (result, n_out, info)


class S42CF_CropPad:
    """Crop or pad frames to target dimensions/aspect ratio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to crop or pad."}),
                "mode": (["crop", "pad_letterbox", "pad_pillarbox", "aspect_ratio"], {
                    "default": "aspect_ratio",
                    "tooltip": "'crop' = cut to exact dimensions.\n'pad_letterbox' = add black bars top/bottom.\n'pad_pillarbox' = add black bars left/right.\n'aspect_ratio' = auto-detect best fit for target ratio."
                }),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target width in pixels. 0 = derive from ratio."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target height in pixels. 0 = derive from ratio."}),
                "ratio_preset": (["keep", "16:9", "9:16", "4:3", "1:1", "21:9", "2.35:1"], {
                    "default": "16:9",
                    "tooltip": "Aspect ratio preset. 'keep' = don't change ratio.\nUsed when target_width or target_height is 0."
                }),
                "pad_color": ("STRING", {"default": "#000000",
                    "tooltip": "Padding/letterbox color (hex). Click swatch to pick."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, target_width, target_height, ratio_preset,
                pad_color):
        clip = ensure_rgb(clip)
        n, h, w, c = clip.shape

        ratios = {"16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "1:1": 1.0, "21:9": 21/9, "2.35:1": 2.35}

        if target_width > 0 and target_height > 0:
            tw, th = target_width, target_height
        elif ratio_preset in ratios:
            ratio = ratios[ratio_preset]
            if w / h > ratio:
                th = h
                tw = int(h * ratio)
            else:
                tw = w
                th = int(w / ratio)
            tw = (tw // 8) * 8
            th = (th // 8) * 8
        else:
            tw, th = w, h

        if mode == "crop":
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - tw // 2)
            y0 = max(0, cy - th // 2)
            x1, y1 = min(w, x0 + tw), min(h, y0 + th)
            result = clip[:, y0:y1, x0:x1]

        elif mode in ("pad_letterbox", "pad_pillarbox", "aspect_ratio"):
            from .cf_utils import hex_to_rgb as _hex2
            pr, pg, pb = _hex2(pad_color)
            pad_color_t = torch.tensor([pr, pg, pb])
            result = pad_color_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, th, tw, 3).clone()
            scale = min(tw / w, th / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = resize_frames(clip, new_h, new_w)
            y_off = (th - new_h) // 2
            x_off = (tw - new_w) // 2
            result[:, y_off:y_off + new_h, x_off:x_off + new_w] = resized
        else:
            result = clip

        info = f"CropPad({mode}): {w}x{h} → {result.shape[2]}x{result.shape[1]}"
        return (result, int(result.shape[0]), info)


class S42CF_MaskWipe:
    """Animated mask-based transitions/reveals."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {"tooltip": "Clip to transition FROM (visible at start)."}),
                "clip_b": ("IMAGE", {"tooltip": "Clip to transition TO (visible at end)."}),
                "wipe_type": (["gradient_left", "gradient_right", "gradient_up", "gradient_down",
                               "radial_in", "radial_out", "diamond", "clock"], {
                    "default": "gradient_right",
                    "tooltip": "Wipe pattern type. Gradient = linear wipe. Radial = circle from center. Diamond = diamond shape. Clock = rotating clock wipe."
                }),
                "transition_frames": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1,
                    "tooltip": "Number of frames for the wipe transition."}),
                "softness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Edge softness of the wipe. 0 = hard edge, 0.5 = very soft."}),
                "easing": (ALL_EASINGS, {"default": "ease_in_out", "tooltip": "Wipe animation easing."}),
            },
            "optional": {
                "custom_mask": ("MASK", {"tooltip": "Custom grayscale mask for wipe pattern. White reveals clip_b."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip_a, clip_b, wipe_type, transition_frames, softness, easing,
                custom_mask=None):
        a = ensure_rgb(clip_a)
        b = ensure_rgb(clip_b)
        a, b = match_resolution(a, b)
        na, nb = a.shape[0], b.shape[0]
        tf = min(transition_frames, na, nb)
        h, w = a.shape[1], a.shape[2]

        Y, X = np.mgrid[:h, :w].astype(np.float32)
        Y_norm, X_norm = Y / h, X / w

        a_keep = a[:na - tf]
        b_keep = b[tf:]
        a_over = a[na - tf:]
        b_over = b[:tf]

        # CRITICAL: Wipes are motion transitions — use stable reference frames
        # (last frame of A, first frame of B) to prevent frame jitter.
        # Same fix as S42P Transition node.
        a_ref = a_over[-1]
        b_ref = b_over[0]

        trans_frames = []
        for i in range(tf):
            raw_t = i / max(tf - 1, 1)
            t = apply_easing(raw_t, easing)

            if wipe_type == "gradient_right":
                mask = X_norm
            elif wipe_type == "gradient_left":
                mask = 1.0 - X_norm
            elif wipe_type == "gradient_down":
                mask = Y_norm
            elif wipe_type == "gradient_up":
                mask = 1.0 - Y_norm
            elif wipe_type == "radial_in":
                cx, cy = w / 2, h / 2
                mask = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / math.hypot(cx, cy)
            elif wipe_type == "radial_out":
                cx, cy = w / 2, h / 2
                mask = 1.0 - np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / math.hypot(cx, cy)
            elif wipe_type == "diamond":
                mask = (np.abs(X_norm - 0.5) + np.abs(Y_norm - 0.5))
            elif wipe_type == "clock":
                angle = np.arctan2(Y_norm - 0.5, X_norm - 0.5)
                mask = (angle + math.pi) / (2 * math.pi)
            else:
                mask = X_norm

            threshold = t
            if softness > 0:
                alpha = np.clip((mask - threshold + softness) / (2 * softness + 1e-6), 0, 1)
            else:
                alpha = (mask >= threshold).astype(np.float32)

            alpha_t = torch.from_numpy(alpha).unsqueeze(-1)
            blended = a_ref * (1 - alpha_t) + b_ref * alpha_t
            trans_frames.append(blended)

        parts = [p for p in [a_keep, torch.stack(trans_frames), b_keep] if p.shape[0] > 0]
        result = torch.cat(parts, dim=0)
        info = f"MaskWipe({wipe_type}): {tf}f transition, softness={softness}"
        return (result, int(result.shape[0]), info)


class S42CF_ShapeMask:
    """Generate animated shape masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Mask width."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Mask height."}),
                "frames": ("INT", {"default": 24, "min": 1, "max": 9999, "tooltip": "Number of mask frames to generate."}),
                "shape": (["rectangle", "ellipse", "star", "diamond"], {
                    "default": "ellipse",
                    "tooltip": "Mask shape type."
                }),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Shape center X (0-1)."}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Shape center Y (0-1)."}),
                "size": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Shape size relative to frame. 0.5 = half frame."}),
                "feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Edge feathering/softness."}),
                "rotation_kf": ("STRING", {
                    "default": "",
                    "tooltip": "Keyframe string for rotation animation. E.g. '0:0, 24:360'. Degrees."
                }),
                "size_kf": ("STRING", {
                    "default": "",
                    "tooltip": "Keyframe string for size animation. E.g. '0:0.2, 12:0.8, 24:0.2'."
                }),
                "invert": (["no", "yes"], {"default": "no", "tooltip": "Invert the mask (white↔black)."}),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "STRING")
    RETURN_NAMES = ("mask", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, width, height, frames, shape, center_x, center_y, size,
                feather, rotation_kf, size_kf, invert):
        rot_kfs = parse_keyframe_string(rotation_kf) if rotation_kf.strip() else []
        size_kfs = parse_keyframe_string(size_kf) if size_kf.strip() else []

        masks = []
        Y, X = np.ogrid[:height, :width]
        Y = Y.astype(np.float32)
        X = X.astype(np.float32)

        for f in range(frames):
            cur_size = interpolate_keyframes(size_kfs, f) if size_kfs else size
            cx = center_x * width
            cy = center_y * height
            rw = cur_size * width / 2
            rh = cur_size * height / 2

            if shape == "ellipse":
                dist = np.sqrt(((X - cx) / max(rw, 1)) ** 2 + ((Y - cy) / max(rh, 1)) ** 2)
                mask = np.clip(1.0 - (dist - 1.0) / max(feather * 10, 0.01), 0, 1)
            elif shape == "rectangle":
                dx = np.abs(X - cx) / max(rw, 1)
                dy = np.abs(Y - cy) / max(rh, 1)
                dist = np.maximum(dx, dy)
                mask = np.clip(1.0 - (dist - 1.0) / max(feather * 10, 0.01), 0, 1)
            elif shape == "diamond":
                dx = np.abs(X - cx) / max(rw, 1)
                dy = np.abs(Y - cy) / max(rh, 1)
                dist = dx + dy
                mask = np.clip(1.0 - (dist - 1.0) / max(feather * 10, 0.01), 0, 1)
            elif shape == "star":
                angles = np.arctan2(Y - cy, X - cx)
                r_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                points = 5
                star_r = rw * (0.5 + 0.5 * np.cos(points * angles))
                dist = r_dist / np.maximum(star_r, 1)
                mask = np.clip(1.0 - (dist - 1.0) / max(feather * 10, 0.01), 0, 1)
            else:
                mask = np.ones((height, width), dtype=np.float32)

            mask = mask.clip(0, 1).astype(np.float32)
            if invert == "yes":
                mask = 1.0 - mask
            masks.append(torch.from_numpy(mask))

        result = torch.stack(masks)
        info = f"ShapeMask({shape}): {width}x{height}, {frames}f, size={size}"
        return (result, frames, info)


NODE_CLASS_MAPPINGS = {
    "S42CF_PictureInPicture": S42CF_PictureInPicture,
    "S42CF_SplitScreen": S42CF_SplitScreen,
    "S42CF_KenBurns": S42CF_KenBurns,
    "S42CF_CropPad": S42CF_CropPad,
    "S42CF_MaskWipe": S42CF_MaskWipe,
    "S42CF_ShapeMask": S42CF_ShapeMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_PictureInPicture": "🖼 S42 CutFlow Picture-in-Picture",
    "S42CF_SplitScreen": "▦ S42 CutFlow Split Screen",
    "S42CF_KenBurns": "🎬 S42 CutFlow Ken Burns",
    "S42CF_CropPad": "✂ S42 CutFlow Crop & Pad",
    "S42CF_MaskWipe": "🎭 S42 CutFlow Mask Wipe",
    "S42CF_ShapeMask": "⬡ S42 CutFlow Shape Mask",
}
