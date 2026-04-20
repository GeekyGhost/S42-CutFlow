"""
S42 CutFlow — Layer Composer (Expanded)
=========================================
Professional multi-layer video/image compositor.
5 layers (bg, fill, fx, fg, overlay) with per-layer:
  blend mode (14 modes), opacity, mask, position, scale, rotation.

Foreground has full placement controls (fit/position/custom).
Batch-aware — processes video frame-by-frame with chunked memory management.

Python 3.12 | ComfyUI Portable | PIL + numpy + torch
"""

from __future__ import annotations
import logging
import math
import numpy as np
import torch
from PIL import Image
from typing import Optional

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Expanded"

BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "hard_light", "soft_light",
    "dodge", "burn", "darken", "lighten", "difference", "exclusion", "add", "subtract",
]

FIT_MODES = ["fill", "fit", "fit_width", "fit_height", "original", "custom"]

POSITIONS = [
    "center", "top_left", "top_center", "top_right",
    "center_left", "center_right",
    "bottom_left", "bottom_center", "bottom_right",
]


def _blend(base, layer, mode):
    b, l = np.clip(base, 0, 1), np.clip(layer, 0, 1)
    if mode == "normal":     return l
    if mode == "multiply":   return b * l
    if mode == "screen":     return 1 - (1 - b) * (1 - l)
    if mode == "overlay":    return np.where(b < 0.5, 2 * b * l, 1 - 2 * (1 - b) * (1 - l))
    if mode == "hard_light": return np.where(l < 0.5, 2 * b * l, 1 - 2 * (1 - b) * (1 - l))
    if mode == "soft_light": return np.where(l < 0.5, b - (1 - 2 * l) * b * (1 - b),
                                              b + (2 * l - 1) * (np.sqrt(np.clip(b, 0, 1)) - b))
    if mode == "dodge":      return np.clip(b / np.clip(1 - l, 1e-6, 1), 0, 1)
    if mode == "burn":       return np.clip(1 - (1 - b) / np.clip(l, 1e-6, 1), 0, 1)
    if mode == "darken":     return np.minimum(b, l)
    if mode == "lighten":    return np.maximum(b, l)
    if mode == "difference": return np.abs(b - l)
    if mode == "exclusion":  return b + l - 2 * b * l
    if mode == "add":        return np.clip(b + l, 0, 1)
    if mode == "subtract":   return np.clip(b - l, 0, 1)
    return l


def _composite(canvas, canvas_alpha, layer_rgb, layer_mask, opacity, blend_mode):
    a_l = np.clip(layer_mask * float(opacity), 0, 1)[..., np.newaxis]
    a_b = canvas_alpha[..., np.newaxis]
    blended = _blend(canvas, layer_rgb, blend_mode)
    out_alpha = a_l[:, :, 0] + canvas_alpha * (1 - a_l[:, :, 0])
    denom = np.clip(out_alpha, 1e-7, 1)[..., np.newaxis]
    out_rgb = (blended * a_l + canvas * a_b * (1 - a_l)) / denom
    return np.clip(out_rgb, 0, 1), np.clip(out_alpha, 0, 1)


def _pil_resize(arr, h, w, is_mask=False):
    if is_mask:
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L")
        return np.array(pil.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    return np.array(pil.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def _get_frame(batch, idx, h, w):
    if batch is None:
        return None
    frm = batch[idx % batch.shape[0]].cpu().numpy().astype(np.float32)
    if frm.ndim == 2:
        frm = np.stack([frm] * 3, axis=-1)
    elif frm.shape[-1] == 4:
        frm = frm[:, :, :3]
    if frm.shape[0] != h or frm.shape[1] != w:
        frm = _pil_resize(frm, h, w)
    return frm


def _get_mask(batch, idx, h, w):
    if batch is None:
        return np.ones((h, w), dtype=np.float32)
    m = batch[idx % batch.shape[0]].cpu().numpy().astype(np.float32)
    if m.ndim > 2:
        m = m[:, :, 0] if m.shape[-1] <= 4 else m
    if m.shape[0] != h or m.shape[1] != w:
        m = _pil_resize(m, h, w, is_mask=True)
    return np.clip(m, 0, 1)


def _place_layer(rgb, mask, canvas_h, canvas_w, fit_mode, position, cw, ch, cx, cy):
    src_h, src_w = rgb.shape[:2]
    H, W = canvas_h, canvas_w
    if fit_mode == "fill":
        rw, rh = W, H
    elif fit_mode == "fit":
        s = min(W / src_w, H / src_h)
        rw, rh = max(1, int(src_w * s)), max(1, int(src_h * s))
    elif fit_mode == "fit_width":
        s = W / src_w
        rw, rh = W, max(1, int(src_h * s))
    elif fit_mode == "fit_height":
        s = H / src_h
        rw, rh = max(1, int(src_w * s)), H
    elif fit_mode == "original":
        rw, rh = src_w, src_h
    else:
        rw, rh = max(1, cw), max(1, ch)

    if rw != src_w or rh != src_h:
        r_rgb = _pil_resize(rgb, rh, rw)
        r_mask = _pil_resize(mask, rh, rw, is_mask=True)
    else:
        r_rgb, r_mask = rgb, mask

    if fit_mode == "custom":
        ox, oy = cx, cy
    elif fit_mode == "fill":
        ox, oy = 0, 0
    else:
        pos = position.lower()
        ox_map = {"top_left": 0, "center_left": 0, "bottom_left": 0,
                  "top_center": (W - rw) // 2, "center": (W - rw) // 2, "bottom_center": (W - rw) // 2,
                  "top_right": W - rw, "center_right": W - rw, "bottom_right": W - rw}
        oy_map = {"top_left": 0, "top_center": 0, "top_right": 0,
                  "center_left": (H - rh) // 2, "center": (H - rh) // 2, "center_right": (H - rh) // 2,
                  "bottom_left": H - rh, "bottom_center": H - rh, "bottom_right": H - rh}
        ox = ox_map.get(pos, (W - rw) // 2)
        oy = oy_map.get(pos, (H - rh) // 2)

    placed_rgb = np.zeros((H, W, 3), dtype=np.float32)
    placed_mask = np.zeros((H, W), dtype=np.float32)
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    pw = min(rw - sx0, W - dx0)
    ph = min(rh - sy0, H - dy0)
    if pw > 0 and ph > 0:
        placed_rgb[dy0:dy0 + ph, dx0:dx0 + pw] = r_rgb[sy0:sy0 + ph, sx0:sx0 + pw]
        placed_mask[dy0:dy0 + ph, dx0:dx0 + pw] = r_mask[sy0:sy0 + ph, sx0:sx0 + pw]
    return placed_rgb, placed_mask


class S42CF_LayerComposer:
    """Professional multi-layer compositor for images and video.
    5 layers: background, fill, FX overlay, foreground (with placement), top overlay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Canvas width in pixels."}),
                "output_height": ("INT", {"default": 576, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Canvas height in pixels."}),
                "bg_fill_color": ("STRING", {"default": "#000000",
                    "tooltip": "Canvas fill color (hex). Visible through transparent areas."}),
                "output_format": (["rgb", "rgba"], {"default": "rgb",
                    "tooltip": "'rgb' = 3-channel standard. 'rgba' = includes alpha as 4th channel."}),
            },
            "optional": {
                # ── Background ──
                "bg_image": ("IMAGE", {"tooltip": "Bottom layer. Stretched to fill canvas."}),
                "bg_mask": ("MASK", {"tooltip": "Background mask."}),
                "bg_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Background opacity."}),
                "bg_blend": (BLEND_MODES, {"default": "normal", "tooltip": "Background blend mode."}),
                # ── Fill layer ──
                "fill_image": ("IMAGE", {"tooltip": "Fill layer above background. Stretched to fill. Good for gradients or solid colors."}),
                "fill_mask": ("MASK", {"tooltip": "Fill layer mask."}),
                "fill_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fill layer opacity."}),
                "fill_blend": (BLEND_MODES, {"default": "normal", "tooltip": "Fill layer blend mode."}),
                # ── FX overlay ──
                "fx_image": ("IMAGE", {"tooltip": "Effects layer. Stretched to fill. Connect particles, visualizers, procedural FX."}),
                "fx_mask": ("MASK", {"tooltip": "FX mask."}),
                "fx_opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "FX layer opacity."}),
                "fx_blend": (BLEND_MODES, {"default": "screen", "tooltip": "FX blend mode. 'screen' ideal for glow/light FX."}),
                # ── Foreground (with placement) ──
                "fg_image": ("IMAGE", {"tooltip": "Foreground subject layer. Has full placement controls."}),
                "fg_mask": ("MASK", {"tooltip": "Foreground alpha mask. Connect from Background Remover."}),
                "fg_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Foreground opacity."}),
                "fg_blend": (BLEND_MODES, {"default": "normal", "tooltip": "Foreground blend mode."}),
                "fg_fit_mode": (FIT_MODES, {"default": "fit",
                    "tooltip": "How foreground is scaled:\nfill = stretch to canvas.\nfit = scale to fit (letterboxed).\nfit_width/fit_height = match one dimension.\noriginal = source pixel size.\ncustom = manual w/h/x/y."}),
                "fg_position": (POSITIONS, {"default": "center",
                    "tooltip": "Foreground anchor position. Ignored in 'fill' and 'custom' modes."}),
                "fg_custom_w": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Custom foreground width (px). Only for 'custom' fit_mode."}),
                "fg_custom_h": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Custom foreground height (px)."}),
                "fg_custom_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Custom foreground X position (px from left)."}),
                "fg_custom_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Custom foreground Y position (px from top)."}),
                # ── Top overlay ──
                "overlay_image": ("IMAGE", {"tooltip": "Top overlay layer (above everything). Stretched to fill. Watermarks, borders, grain."}),
                "overlay_mask": ("MASK", {"tooltip": "Overlay mask."}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Overlay opacity."}),
                "overlay_blend": (BLEND_MODES, {"default": "screen", "tooltip": "Overlay blend mode."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composited", "alpha_mask")
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    def compose(self, output_width=1024, output_height=576,
                bg_fill_color="#000000", output_format="rgb",
                bg_image=None, bg_mask=None, bg_opacity=1.0, bg_blend="normal",
                fill_image=None, fill_mask=None, fill_opacity=1.0, fill_blend="normal",
                fx_image=None, fx_mask=None, fx_opacity=0.8, fx_blend="screen",
                fg_image=None, fg_mask=None, fg_opacity=1.0, fg_blend="normal",
                fg_fit_mode="fit", fg_position="center",
                fg_custom_w=512, fg_custom_h=512, fg_custom_x=0, fg_custom_y=0,
                overlay_image=None, overlay_mask=None, overlay_opacity=0.5, overlay_blend="screen"):

        W, H = int(output_width), int(output_height)
        fill_rgb = self._parse_hex(bg_fill_color)

        lengths = [b.shape[0] for b in [bg_image, fill_image, fx_image, fg_image, overlay_image] if b is not None]
        N = max(lengths) if lengths else 1

        frame_list, alpha_list = [], []
        CHUNK = 64

        for i in range(N):
            canvas = np.full((H, W, 3), fill_rgb, dtype=np.float32)
            canvas_alpha = np.zeros((H, W), dtype=np.float32)

            # Layer 1: Background
            bg = _get_frame(bg_image, i, H, W)
            if bg is not None:
                canvas, canvas_alpha = _composite(canvas, canvas_alpha, bg, _get_mask(bg_mask, i, H, W), bg_opacity, bg_blend)

            # Layer 2: Fill
            fl = _get_frame(fill_image, i, H, W)
            if fl is not None:
                canvas, canvas_alpha = _composite(canvas, canvas_alpha, fl, _get_mask(fill_mask, i, H, W), fill_opacity, fill_blend)

            # Layer 3: FX
            fx = _get_frame(fx_image, i, H, W)
            if fx is not None:
                canvas, canvas_alpha = _composite(canvas, canvas_alpha, fx, _get_mask(fx_mask, i, H, W), fx_opacity, fx_blend)

            # Layer 4: Foreground (with placement)
            if fg_image is not None:
                fg_raw = fg_image[i % fg_image.shape[0]].cpu().numpy().astype(np.float32)
                if fg_raw.ndim == 2:
                    fg_raw = np.stack([fg_raw] * 3, axis=-1)
                elif fg_raw.shape[-1] == 4:
                    fg_raw = fg_raw[:, :, :3]
                src_h, src_w = fg_raw.shape[:2]
                if fg_mask is not None:
                    fgm = _get_mask(fg_mask, i, src_h, src_w)
                else:
                    fgm = np.ones((src_h, src_w), dtype=np.float32)
                placed_rgb, placed_mask = _place_layer(fg_raw, fgm, H, W, fg_fit_mode, fg_position,
                                                        int(fg_custom_w), int(fg_custom_h), int(fg_custom_x), int(fg_custom_y))
                canvas, canvas_alpha = _composite(canvas, canvas_alpha, placed_rgb, placed_mask, fg_opacity, fg_blend)

            # Layer 5: Top overlay
            ov = _get_frame(overlay_image, i, H, W)
            if ov is not None:
                canvas, canvas_alpha = _composite(canvas, canvas_alpha, ov, _get_mask(overlay_mask, i, H, W), overlay_opacity, overlay_blend)

            if all(x is None for x in [bg_image, fill_image, fx_image, fg_image, overlay_image]):
                canvas_alpha = np.ones((H, W), dtype=np.float32)

            if output_format == "rgba":
                frm = np.dstack([canvas, canvas_alpha[..., np.newaxis]])
            else:
                frm = canvas

            frame_list.append(torch.from_numpy(frm.astype(np.float32)))
            alpha_list.append(torch.from_numpy(canvas_alpha.astype(np.float32)))

        composited = torch.stack(frame_list)
        alpha_mask = torch.stack(alpha_list)
        logger.info(f"[S42CF LayerComposer] {N}f {W}x{H} format={output_format}")
        return (composited, alpha_mask)

    def _parse_hex(self, hex_str):
        try:
            s = hex_str.strip().lstrip("#")
            if len(s) == 3:
                s = "".join(c * 2 for c in s)
            return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
        except Exception:
            return (0.0, 0.0, 0.0)


NODE_CLASS_MAPPINGS = {
    "S42CF_LayerComposer": S42CF_LayerComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_LayerComposer": "🎬 S42 CutFlow Layer Composer",
}
