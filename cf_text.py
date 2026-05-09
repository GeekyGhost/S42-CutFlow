"""
S42 CutFlow — Text & Overlays
================================
Text rendering, subtitles, watermarks, lower thirds.
All use PIL for text rendering — no external font dependencies.

Python 3.12 | ComfyUI Portable | PIL + torch
"""

import torch
import numpy as np
import math
import logging
import os
from typing import Optional

from .cf_utils import (
    ensure_rgb, frame_to_pil, pil_to_frame, PIL_AVAILABLE, parse_srt,
    parse_keyframe_string, interpolate_keyframes, apply_easing
)

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Text"

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int, bold: bool = False):
    """Get a font, falling back to PIL default if custom fonts unavailable."""
    try:
        font_names = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                      "LiberationSans-Regular.ttf", "FreeSans.ttf"]
        if bold:
            font_names = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                          "LiberationSans-Bold.ttf"] + font_names
        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default(size=max(10, size))
    except Exception:
        return ImageFont.load_default()


def _composite_rgba(base_tensor: torch.Tensor, overlay_rgba: torch.Tensor) -> torch.Tensor:
    """
    Properly composites an RGBA overlay onto a base tensor (RGB or RGBA) using straight alpha.
    base_tensor: shape (H, W, C) where C is 3 or 4
    overlay_rgba: shape (H, W, 4)
    """
    o_rgb = overlay_rgba[..., :3]
    o_a = overlay_rgba[..., 3:4]

    if base_tensor.shape[-1] == 4:
        b_rgb = base_tensor[..., :3]
        b_a = base_tensor[..., 3:4]

        # Calculate new alpha: out_a = src_a + dst_a * (1 - src_a)
        out_a = o_a + b_a * (1.0 - o_a)

        # Prevent division by zero
        out_a_safe = torch.where(out_a > 0, out_a, torch.ones_like(out_a))

        # Calculate new RGB: out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1 - src_a)) / out_a
        out_rgb = (o_rgb * o_a + b_rgb * b_a * (1.0 - o_a)) / out_a_safe
        
        # Zero out RGB where alpha is explicitly 0 to keep it perfectly clean
        out_rgb = torch.where(out_a > 0, out_rgb, torch.zeros_like(out_rgb))

        return torch.cat([out_rgb, out_a], dim=-1).clamp(0, 1)
    else:
        # If the base has no alpha, do a standard blend
        out_rgb = base_tensor * (1.0 - o_a) + o_rgb * o_a
        return out_rgb.clamp(0, 1)


class S42CF_TextOverlay:
    """Rich text rendering with animation support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to overlay text onto."}),
                "text": ("STRING", {
                    "default": "Hello World",
                    "multiline": True,
                    "tooltip": "Text to render. Supports multi-line with newlines."
                }),
                "font_size": ("INT", {
                    "default": 48, "min": 8, "max": 400, "step": 2,
                    "tooltip": "Font size in pixels."
                }),
                "bold": (["no", "yes"], {"default": "no", "tooltip": "Use bold font variant."}),
                "text_color": ("STRING", {"default": "#FFFFFF",
                    "tooltip": "Text color (hex). Click swatch to pick."}),
                "position_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Horizontal position. 0=left edge, 0.5=center, 1=right edge."
                }),
                "position_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical position. 0=top, 0.5=center, 1=bottom."
                }),
                "alignment": (["center", "left", "right"], {
                    "default": "center",
                    "tooltip": "Text alignment relative to position point."
                }),
                "stroke_width": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Text outline/stroke width in pixels. 0 = no stroke."
                }),
                "stroke_color": ("STRING", {"default": "#000000",
                    "tooltip": "Text outline/stroke color (hex). Click swatch to pick."}),
                "shadow_offset": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Drop shadow offset in pixels. 0 = no shadow."
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Text opacity. 0 = invisible, 1 = fully opaque."
                }),
                "bg_padding": ("INT", {
                    "default": 0, "min": 0, "max": 50, "step": 2,
                    "tooltip": "Background box padding around text. 0 = no background box."
                }),
                "bg_opacity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Background box opacity. Only used when bg_padding > 0."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, text, font_size, bold, text_color,
                position_x, position_y, alignment, stroke_width,
                stroke_color,
                shadow_offset, opacity, bg_padding, bg_opacity):
        if not PIL_AVAILABLE:
            return (clip, int(clip.shape[0]), "TextOverlay: PIL not available")

        # Removed ensure_rgb() to preserve incoming Alpha
        n, h, w, c = clip.shape
        font = _get_font(font_size, bold == "yes")
        from .cf_utils import hex_to_rgb255
        color = hex_to_rgb255(text_color)
        s_color = hex_to_rgb255(stroke_color)

        text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)

        bbox = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if alignment == "center":
            tx = int(position_x * w - tw / 2)
        elif alignment == "right":
            tx = int(position_x * w - tw)
        else:
            tx = int(position_x * w)
        ty = int(position_y * h - th / 2)

        if bg_padding > 0:
            bg_rect = [tx - bg_padding, ty - bg_padding, tx + tw + bg_padding, ty + th + bg_padding]
            bg_alpha = int(bg_opacity * 255)
            draw.rectangle(bg_rect, fill=(0, 0, 0, bg_alpha))

        if shadow_offset > 0:
            draw.multiline_text((tx + shadow_offset, ty + shadow_offset), text,
                                font=font, fill=(0, 0, 0, int(opacity * 180)),
                                align=alignment)

        alpha_val = int(opacity * 255)
        fill = (color[0], color[1], color[2], alpha_val)
        s_fill = (s_color[0], s_color[1], s_color[2], alpha_val) if stroke_width > 0 else None
        draw.multiline_text((tx, ty), text, font=font, fill=fill,
                            stroke_width=stroke_width, stroke_fill=s_fill, align=alignment)

        overlay_np = np.array(text_img).astype(np.float32) / 255.0
        overlay_tensor = torch.from_numpy(overlay_np)

        results = []
        for i in range(n):
            composited = _composite_rgba(clip[i], overlay_tensor)
            results.append(composited)

        result = torch.stack(results)
        info = f"TextOverlay: '{text[:30]}...' size={font_size} pos=({position_x:.2f},{position_y:.2f})"
        return (result, int(result.shape[0]), info)


class S42CF_SubtitleBurn:
    """Burn SRT subtitles onto video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to burn subtitles onto."}),
                "srt_text": ("STRING", {
                    "default": "1\n00:00:00,000 --> 00:00:02,000\nHello World\n\n2\n00:00:02,500 --> 00:00:05,000\nSubtitle Example",
                    "multiline": True,
                    "tooltip": "SRT format subtitle text. Standard format:\n1\n00:00:00,000 --> 00:00:02,000\nText here"
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Frame rate to convert SRT timestamps to frame indices."}),
                "font_size": ("INT", {"default": 36, "min": 12, "max": 200, "step": 2, "tooltip": "Subtitle font size."}),
                "style": (["default", "cinematic", "youtube", "outline_only"], {
                    "default": "default",
                    "tooltip": "'default' = white text, black outline.\n'cinematic' = white text, dark shadow, larger.\n'youtube' = white on semi-transparent black box.\n'outline_only' = thick black outline, no fill background."
                }),
                "position_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Vertical position. 0.9 = near bottom (standard). 0.1 = near top."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, srt_text, fps, font_size, style, position_y):
        if not PIL_AVAILABLE:
            return (clip, int(clip.shape[0]), "SubtitleBurn: PIL not available")

        n, h, w, c = clip.shape
        subs = parse_srt(srt_text)
        font = _get_font(font_size, style == "cinematic")

        style_cfg = {
            "default": {"fill": (255, 255, 255, 255), "stroke": 2, "stroke_fill": (0, 0, 0, 255), "bg": False},
            "cinematic": {"fill": (255, 255, 255, 255), "stroke": 0, "stroke_fill": None, "bg": False, "shadow": 3},
            "youtube": {"fill": (255, 255, 255, 255), "stroke": 0, "stroke_fill": None, "bg": True},
            "outline_only": {"fill": (255, 255, 255, 0), "stroke": 4, "stroke_fill": (0, 0, 0, 255), "bg": False},
        }.get(style, {"fill": (255, 255, 255, 255), "stroke": 2, "stroke_fill": (0, 0, 0, 255), "bg": False})

        results = []
        for i in range(n):
            t = i / fps
            active_sub = None
            for sub in subs:
                if sub["start_s"] <= t <= sub["end_s"]:
                    active_sub = sub
                    break

            if active_sub is None:
                results.append(clip[i])
                continue

            overlay_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img)
            text = active_sub["text"]

            bbox = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=style_cfg.get("stroke", 0))
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = (w - tw) // 2
            ty = int(position_y * h - th / 2)

            if style_cfg.get("bg"):
                pad = 8
                draw.rectangle([tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                               fill=(0, 0, 0, 180))

            if style_cfg.get("shadow"):
                s = style_cfg["shadow"]
                draw.multiline_text((tx + s, ty + s), text, font=font, fill=(0, 0, 0, 180), align="center")

            draw.multiline_text((tx, ty), text, font=font,
                                fill=style_cfg["fill"],
                                stroke_width=style_cfg.get("stroke", 0),
                                stroke_fill=style_cfg.get("stroke_fill"),
                                align="center")

            overlay_np = np.array(overlay_img).astype(np.float32) / 255.0
            overlay_tensor = torch.from_numpy(overlay_np)
            composited = _composite_rgba(clip[i], overlay_tensor)
            results.append(composited)

        result = torch.stack(results)
        info = f"SubtitleBurn: {len(subs)} subtitles, style={style}"
        return (result, int(result.shape[0]), info)


class S42CF_Watermark:
    """Image or text watermark overlay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to watermark."}),
                "mode": (["text", "image"], {
                    "default": "text",
                    "tooltip": "'text' = render text as watermark. 'image' = use connected image as watermark."
                }),
                "text": ("STRING", {"default": "© 2026", "tooltip": "Watermark text (used in 'text' mode)."}),
                "position": (["bottom_right", "bottom_left", "top_right", "top_left", "center", "tile"], {
                    "default": "bottom_right",
                    "tooltip": "Watermark placement. 'tile' = repeat across entire frame."
                }),
                "opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Watermark opacity. Lower = more subtle."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Scale multiplier for watermark size."}),
                "font_size": ("INT", {"default": 24, "min": 8, "max": 200, "step": 2,
                    "tooltip": "Font size for text watermark."}),
                "margin": ("INT", {"default": 20, "min": 0, "max": 200, "step": 5,
                    "tooltip": "Margin from edges in pixels."}),
            },
            "optional": {
                "watermark_image": ("IMAGE", {"tooltip": "Watermark image (used in 'image' mode). First frame used."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, mode, text, position, opacity, scale, font_size, margin,
                watermark_image=None):
        if not PIL_AVAILABLE:
            return (clip, int(clip.shape[0]), "Watermark: PIL not available")

        n, h, w, c = clip.shape

        if mode == "image" and watermark_image is not None:
            wm_tensor = watermark_image[0]
            # Convert tensor to PIL safely preserving Alpha if C=4
            wm_np = (wm_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if wm_tensor.shape[-1] == 4:
                wm_pil = Image.fromarray(wm_np, "RGBA")
            else:
                wm_pil = Image.fromarray(wm_np, "RGB").convert("RGBA")

            new_w = int(wm_pil.width * scale)
            new_h = int(wm_pil.height * scale)
            wm_pil = wm_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            font = _get_font(int(font_size * scale))
            temp = Image.new("RGBA", (1, 1))
            draw = ImageDraw.Draw(temp)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10
            wm_pil = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            draw = ImageDraw.Draw(wm_pil)
            draw.text((5, 5), text, font=font, fill=(255, 255, 255, 255))
            new_w, new_h = tw, th

        wm_np = np.array(wm_pil).astype(np.float32) / 255.0

        if position == "tile":
            full_wm = np.zeros((h, w, 4), dtype=np.float32)
            for ty in range(0, h, new_h + margin):
                for tx in range(0, w, new_w + margin):
                    y2 = min(ty + new_h, h)
                    x2 = min(tx + new_w, w)
                    sh = y2 - ty
                    sw = x2 - tx
                    full_wm[ty:y2, tx:x2] = wm_np[:sh, :sw]
            full_wm_tensor = torch.from_numpy(full_wm)
            full_wm_tensor[..., 3:4] *= opacity # scale opacity
        else:
            pos_map = {
                "bottom_right": (w - new_w - margin, h - new_h - margin),
                "bottom_left": (margin, h - new_h - margin),
                "top_right": (w - new_w - margin, margin),
                "top_left": (margin, margin),
                "center": ((w - new_w) // 2, (h - new_h) // 2),
            }
            px, py = pos_map.get(position, (margin, margin))
            px, py = max(0, px), max(0, py)

            full_wm = np.zeros((h, w, 4), dtype=np.float32)
            y2 = min(py + new_h, h)
            x2 = min(px + new_w, w)
            sh, sw = y2 - py, x2 - px
            full_wm[py:y2, px:x2] = wm_np[:sh, :sw]
            
            full_wm_tensor = torch.from_numpy(full_wm)
            full_wm_tensor[..., 3:4] *= opacity # scale opacity

        results = []
        for i in range(n):
            composited = _composite_rgba(clip[i], full_wm_tensor)
            results.append(composited)

        result = torch.stack(results)
        info = f"Watermark({mode}): position={position}, opacity={opacity}"
        return (result, int(result.shape[0]), info)


class S42CF_LowerThird:
    """Animated lower-third title cards."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {"tooltip": "Video clip to add lower third to."}),
                "name": ("STRING", {"default": "John Smith", "tooltip": "Primary name/title text."}),
                "title": ("STRING", {"default": "Lead Designer", "tooltip": "Secondary subtitle text."}),
                "style": (["modern", "minimal", "broadcast", "cinematic"], {
                    "default": "modern",
                    "tooltip": "'modern' = colored accent bar + clean text.\n'minimal' = text only, small.\n'broadcast' = full background bar.\n'cinematic' = large text with subtle bg."
                }),
                "accent_color": ("STRING", {"default": "#FF9900",
                    "tooltip": "Accent color for lower third (hex). Click swatch to pick."}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "Frame where lower third appears."}),
                "duration_frames": ("INT", {"default": 72, "min": 1, "max": 9999, "tooltip": "How many frames the lower third is visible."}),
                "animate_in": ("INT", {"default": 12, "min": 0, "max": 60, "tooltip": "Slide-in animation frames."}),
                "animate_out": ("INT", {"default": 12, "min": 0, "max": 60, "tooltip": "Slide-out animation frames."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("clip", "frame_count", "info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, clip, name, title, style, accent_color,
                start_frame, duration_frames, animate_in, animate_out):
        if not PIL_AVAILABLE:
            return (clip, int(clip.shape[0]), "LowerThird: PIL not available")

        n, h, w, c = clip.shape
        end_frame = start_frame + duration_frames

        name_font = _get_font(36, True)
        title_font = _get_font(24, False)
        from .cf_utils import hex_to_rgb255
        accent = hex_to_rgb255(accent_color)

        results = []
        for i in range(n):
            if i < start_frame or i >= end_frame:
                results.append(clip[i])
                continue

            local_t = i - start_frame
            if local_t < animate_in:
                slide = local_t / max(animate_in, 1)
                slide = apply_easing(slide, "ease_out_cubic")
            elif local_t >= duration_frames - animate_out:
                remaining = duration_frames - local_t
                slide = remaining / max(animate_out, 1)
                slide = apply_easing(slide, "ease_in_cubic")
            else:
                slide = 1.0

            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            bar_h = 80
            bar_y = int(h * 0.78)
            offset_x = int((1 - slide) * (-w * 0.4))

            if style == "modern":
                draw.rectangle([offset_x, bar_y, offset_x + 6, bar_y + bar_h],
                               fill=(*accent, 255))
                draw.rectangle([offset_x + 10, bar_y, offset_x + 400, bar_y + bar_h],
                               fill=(0, 0, 0, int(180 * slide)))
                draw.text((offset_x + 20, bar_y + 10), name, font=name_font,
                          fill=(255, 255, 255, int(255 * slide)))
                draw.text((offset_x + 20, bar_y + 48), title, font=title_font,
                          fill=(*accent, int(255 * slide)))
            elif style == "broadcast":
                draw.rectangle([offset_x, bar_y, offset_x + 450, bar_y + bar_h],
                               fill=(*accent, int(220 * slide)))
                draw.text((offset_x + 16, bar_y + 10), name, font=name_font,
                          fill=(255, 255, 255, int(255 * slide)))
                draw.text((offset_x + 16, bar_y + 48), title, font=title_font,
                          fill=(255, 255, 255, int(200 * slide)))
            elif style == "minimal":
                draw.text((offset_x + 40, bar_y + 10), name, font=name_font,
                          fill=(255, 255, 255, int(255 * slide)),
                          stroke_width=2, stroke_fill=(0, 0, 0, int(200 * slide)))
                draw.text((offset_x + 40, bar_y + 48), title, font=title_font,
                          fill=(200, 200, 200, int(220 * slide)),
                          stroke_width=1, stroke_fill=(0, 0, 0, int(180 * slide)))
            else:
                draw.rectangle([0, bar_y - 10, w, bar_y + bar_h + 10],
                               fill=(0, 0, 0, int(120 * slide)))
                draw.text(((w - 300) // 2 + offset_x, bar_y + 5), name, font=name_font,
                          fill=(255, 255, 255, int(255 * slide)))
                draw.text(((w - 300) // 2 + offset_x, bar_y + 45), title, font=title_font,
                          fill=(*accent, int(255 * slide)))

            overlay_np = np.array(overlay).astype(np.float32) / 255.0
            overlay_tensor = torch.from_numpy(overlay_np)
            
            composited = _composite_rgba(clip[i], overlay_tensor)
            results.append(composited)

        result = torch.stack(results)
        info = f"LowerThird({style}): '{name}' / '{title}' frames {start_frame}-{end_frame}"
        return (result, int(result.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "S42CF_TextOverlay": S42CF_TextOverlay,
    "S42CF_SubtitleBurn": S42CF_SubtitleBurn,
    "S42CF_Watermark": S42CF_Watermark,
    "S42CF_LowerThird": S42CF_LowerThird,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42CF_TextOverlay": "📝 S42 CutFlow Text Overlay",
    "S42CF_SubtitleBurn": "💬 S42 CutFlow Subtitle Burn",
    "S42CF_Watermark": "🔖 S42 CutFlow Watermark",
    "S42CF_LowerThird": "📺 S42 CutFlow Lower Third",
}
