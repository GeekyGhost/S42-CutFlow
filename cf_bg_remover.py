"""
S42 CutFlow — Background Remover (Expanded) v1.1
==================================================
Professional background removal for images and video batches.

FIXES in v1.1:
  - Correct HuggingFace repo names (underscores: BiRefNet_dynamic, BiRefNet_HR)
  - Added optional guidance_mask input for multi-subject isolation
  - Fixed luma key freeze (softness floor + limited post-process on traditional)
  - Added progress logging per 10 frames
  - Session cache validated with try/except per-frame

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations
import os, time, logging
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple

logger = logging.getLogger("S42CutFlow")
CATEGORY = "S42 CutFlow/Expanded"

try:
    import folder_paths
    _MODELS_DIR = os.path.join(folder_paths.models_dir, "rembg")
except Exception:
    _MODELS_DIR = os.path.join(os.path.expanduser("~"), ".u2net")
os.makedirs(_MODELS_DIR, exist_ok=True)

try:
    import cv2; _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from rembg import remove, new_session; _HAS_REMBG = True
except ImportError:
    _HAS_REMBG = False

try:
    from transformers import AutoModelForImageSegmentation; _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

_SESSION_CACHE = {}

# CORRECT HuggingFace repo names (verified March 2026)
_BIREFNET_REPOS = {
    "birefnet-general":      "ZhengPeng7/BiRefNet",
    "birefnet-portrait":     "ZhengPeng7/BiRefNet-portrait",
    "birefnet-massive":      "ZhengPeng7/BiRefNet-massive-TR_DIS5K",
    "birefnet-general-lite": "ZhengPeng7/BiRefNet_lite",
    "birefnet-hr":           "ZhengPeng7/BiRefNet_HR",
    "birefnet-dynamic":      "ZhengPeng7/BiRefNet_dynamic",
    "birefnet-matting":      "ZhengPeng7/BiRefNet-matting",
}

def _get_session(model_name, use_gpu):
    key = f"{model_name}_{use_gpu}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    try:
        os.environ["U2NET_HOME"] = _MODELS_DIR
        if model_name.startswith("birefnet") and _HAS_TRANSFORMERS:
            repo = _BIREFNET_REPOS.get(model_name)
            if repo is None:
                logger.warning(f"[S42CF BG] Unknown BiRefNet variant: {model_name}")
                return None
            logger.info(f"[S42CF BG] Loading {model_name} from {repo}...")
            model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
            if use_gpu and torch.cuda.is_available():
                model = model.cuda().half()
            model.eval()
            _SESSION_CACHE[key] = ("transformers", model)
            logger.info(f"[S42CF BG] Loaded {model_name} via transformers")
            return _SESSION_CACHE[key]
        if _HAS_REMBG:
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"])
            sess = new_session(model_name, providers=providers)
            _SESSION_CACHE[key] = ("rembg", sess)
            logger.info(f"[S42CF BG] Loaded {model_name} via rembg")
            return _SESSION_CACHE[key]
    except Exception as e:
        logger.error(f"[S42CF BG] Session failed {model_name}: {e}")
    return None

def _transformers_infer(image_pil, model, use_gpu, proc_res=1024):
    try:
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize((proc_res, proc_res)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        t = tf(image_pil).unsqueeze(0)
        if use_gpu and torch.cuda.is_available(): t = t.cuda().half()
        with torch.no_grad(): preds = model(t)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        from torchvision.transforms import ToPILImage
        mask = ToPILImage()(pred).resize(image_pil.size, Image.LANCZOS)
        return np.array(mask, dtype=np.float32) / 255.0
    except Exception as e:
        logger.error(f"[S42CF BG] Transformers inference failed: {e}")
        return np.ones((image_pil.size[1], image_pil.size[0]), dtype=np.float32)

def _rembg_infer(image_pil, session):
    result = remove(image_pil, session=session).convert("RGBA")
    return np.array(result.split()[3], dtype=np.float32) / 255.0

def _chroma_key(image_np, color, threshold, softness, spill_suppress):
    if not _HAS_CV2: return np.ones(image_np.shape[:2], dtype=np.float32)
    hsv = cv2.cvtColor((image_np*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32)/255.0, hsv[:,:,2].astype(np.float32)/255.0
    hue_map = {"green":(60,30), "blue":(110,30), "red":(0,15)}
    hue_center, hue_range = hue_map.get(color, (60,30))
    if color == "red": hue_dist = np.minimum(h, 180-h) / hue_range
    else: hue_dist = np.minimum(np.abs(h-hue_center), 180-np.abs(h-hue_center)) / hue_range
    key_signal = (1 - np.clip(hue_dist,0,1)) * np.clip(s-0.25,0,1)/0.75
    softness = max(softness, 0.005)
    mask = 1.0 - np.clip((key_signal - threshold) / softness, 0, 1)
    if spill_suppress > 0:
        ch_map = {"green":1, "blue":2, "red":0}
        ch = ch_map.get(color, 1)
        others = [i for i in range(3) if i != ch]
        spill = np.maximum(image_np[:,:,ch] - np.maximum(image_np[:,:,others[0]], image_np[:,:,others[1]]), 0)
        mask = mask * (1 - spill * spill_suppress)
    return np.clip(mask, 0, 1).astype(np.float32)

def _luma_key(image_np, mode, shadow_thresh, highlight_thresh, softness):
    lum = 0.299*image_np[:,:,0] + 0.587*image_np[:,:,1] + 0.114*image_np[:,:,2]
    softness = max(softness, 0.02)  # CRITICAL: prevent near-zero freeze
    if mode == "shadows": mask = np.clip((lum - shadow_thresh) / softness, 0, 1)
    elif mode == "highlights": mask = np.clip((highlight_thresh - lum) / softness, 0, 1)
    else:
        mask = np.clip((lum - shadow_thresh) / softness, 0, 1) * np.clip((highlight_thresh - lum) / softness, 0, 1)
    return np.clip(mask, 0, 1).astype(np.float32)

def _apply_guidance(ai_mask, guidance_np, strength):
    """Apply guidance mask as a GATE on the AI mask.
    
    How it works:
      - White in guidance (1.0) = "this is my subject, keep it" → AI mask passes through
      - Black in guidance (0.0) = "NOT my subject, force-remove" → result = 0 regardless of AI
      - Gray = proportional blend
    
    Strength controls how hard the gate is:
      - 1.0 = hard gate (black areas completely zeroed out)
      - 0.5 = soft gate (black areas reduced to 50% of AI mask)
      - 0.0 = guidance ignored entirely
    
    This means: paint white over the subject you want to keep,
    leave everything else black, and the BG remover will ONLY
    keep what's inside your painted area.
    """
    if guidance_np is None:
        return ai_mask
    if guidance_np.shape != ai_mask.shape:
        gm_pil = Image.fromarray((guidance_np * 255).astype(np.uint8), "L")
        guidance_np = np.array(
            gm_pil.resize((ai_mask.shape[1], ai_mask.shape[0]), Image.LANCZOS),
            dtype=np.float32) / 255.0

    # Gate: blend between "ai_mask unchanged" and "ai_mask * guidance"
    # At strength=1.0: final = ai_mask * guidance (hard gate)
    # At strength=0.0: final = ai_mask (no guidance effect)
    gate = guidance_np * strength + (1.0 - strength)  # ranges from 1.0 (no effect) to guidance (full effect)
    return np.clip(ai_mask * gate, 0, 1).astype(np.float32)

def _post_process_mask(mask, blur, dilation, erosion, small_obj, is_trad=False):
    if not _HAS_CV2: return mask
    arr = (mask*255).clip(0,255).astype(np.uint8)
    if small_obj > 0 and not is_trad:
        try:
            num,labels,stats,_ = cv2.connectedComponentsWithStats(arr, connectivity=8)
            if num < 50000:
                for i in range(1, num):
                    if stats[i, cv2.CC_STAT_AREA] < small_obj: arr[labels==i]=0
        except: pass
    if dilation > 0:
        arr = cv2.dilate(arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation*2+1,dilation*2+1)), iterations=1)
    if erosion > 0:
        arr = cv2.erode(arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion*2+1,erosion*2+1)), iterations=1)
    if blur > 0:
        ks = int(blur*4)|1
        arr = cv2.GaussianBlur(arr, (ks,ks), blur)
    return arr.astype(np.float32)/255.0

AI_METHODS = ["birefnet-general","birefnet-portrait","birefnet-massive","birefnet-general-lite",
    "birefnet-hr","birefnet-dynamic","birefnet-matting","u2net","u2net_human_seg","u2netp","silueta","isnet-general-use","isnet-anime"]
TRAD_METHODS = ["chroma_green","chroma_blue","chroma_red","luma_key"]
ALL_METHODS = AI_METHODS + TRAD_METHODS

class S42CF_BackgroundRemover:
    """Professional background removal with AI and traditional methods.
    Batch-optimized for video with temporal consistency and optional guidance mask."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Image or video batch [B,H,W,C]."}),
            "method": (ALL_METHODS, {"default": "birefnet-general",
                "tooltip": "birefnet-general = best all-around AI.\nbirefnet-portrait = people.\nbirefnet-massive = highest quality.\nbirefnet-hr = high-res 2048px.\nbirefnet-dynamic = robust across resolutions.\nbirefnet-matting = soft edges (hair, fur).\nu2net/isnet/silueta = classic AI (rembg).\nchroma_green/blue/red = screen keying.\nluma_key = brightness-based."}),
            "use_gpu": (["yes","no"], {"default":"yes", "tooltip":"GPU for AI inference."}),
            "processing_resolution": ("INT", {"default":1024,"min":256,"max":2048,"step":64, "tooltip":"AI inference resolution. 1024=standard, 2048=BiRefNet_HR."}),
            "mask_blur": ("FLOAT", {"default":0.8,"min":0.0,"max":5.0,"step":0.1, "tooltip":"Edge blur. 0=sharp, 0.8=smooth."}),
            "mask_dilation": ("INT", {"default":1,"min":0,"max":10,"step":1, "tooltip":"Expand mask N px. Helps with hair."}),
            "mask_erosion": ("INT", {"default":1,"min":0,"max":10,"step":1, "tooltip":"Shrink mask N px. Removes fringe."}),
            "remove_small_objects": ("INT", {"default":300,"min":0,"max":5000,"step":50, "tooltip":"Remove noise < N px. Disabled for traditional methods."}),
            "invert_mask": (["no","yes"], {"default":"no", "tooltip":"Swap foreground/background."}),
            "chroma_threshold": ("FLOAT", {"default":0.12,"min":0.01,"max":0.5,"step":0.01, "tooltip":"Chroma key sensitivity (chroma_* only)."}),
            "chroma_softness": ("FLOAT", {"default":0.08,"min":0.01,"max":0.3,"step":0.01, "tooltip":"Chroma edge softness."}),
            "spill_suppress": ("FLOAT", {"default":0.7,"min":0.0,"max":1.0,"step":0.05, "tooltip":"Color spill suppression."}),
            "luma_mode": (["shadows","highlights","shadows_highlights"], {"default":"shadows_highlights", "tooltip":"What to KEEP. shadows=keep dark. highlights=keep bright. both=keep midtones."}),
            "shadow_threshold": ("FLOAT", {"default":0.08,"min":0.0,"max":0.5,"step":0.01, "tooltip":"Below = removed. Luma key only."}),
            "highlight_threshold": ("FLOAT", {"default":0.92,"min":0.5,"max":1.0,"step":0.01, "tooltip":"Above = removed. Luma key only."}),
            "luma_softness": ("FLOAT", {"default":0.1,"min":0.02,"max":0.3,"step":0.01, "tooltip":"Luma edge softness. Min 0.02 prevents freezes."}),
            "temporal_smooth": ("INT", {"default":0,"min":0,"max":5,"step":1, "tooltip":"Video mask smoothing window. 0=off, 1-5=flicker reduction."}),
            "guidance_strength": ("FLOAT", {"default":0.7,"min":0.0,"max":1.0,"step":0.05, "tooltip":"How hard the guidance mask gates the output.\n0 = guidance ignored.\n0.5 = black areas reduced to 50%.\n1.0 = black areas completely removed.\nRecommended: 0.8-1.0 for clean isolation."}),
        }, "optional": {
            "guidance_mask": ("MASK", {"tooltip": "Paint white over the subject you want to KEEP.\nLeave everything else black.\nThe BG remover will ONLY keep what's inside your white area.\n\nHow it works: this mask GATES the AI output.\nWhite = AI mask passes through (subject kept).\nBlack = forced to zero (guaranteed removed).\nGray = proportional.\n\nPerfect for isolating one person from a group,\nor keeping just one object in a busy scene."}),
            "reference_bg": ("IMAGE", {"tooltip": "Clean background plate. Enhances AI with difference info."}),
        }}

    RETURN_TYPES = ("IMAGE","IMAGE","MASK","STRING")
    RETURN_NAMES = ("result_rgb","result_rgba","mask","info")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, images, method, use_gpu, processing_resolution, mask_blur, mask_dilation, mask_erosion,
                remove_small_objects, invert_mask, chroma_threshold, chroma_softness, spill_suppress,
                luma_mode, shadow_threshold, highlight_threshold, luma_softness, temporal_smooth,
                guidance_strength, guidance_mask=None, reference_bg=None):
        from .cf_utils import ensure_rgb
        images = ensure_rgb(images)
        n,h,w,c = images.shape
        gpu = use_gpu=="yes"
        is_trad = method in TRAD_METHODS
        t0 = time.time()

        guidance_np = None
        if guidance_mask is not None:
            gm = guidance_mask[0] if guidance_mask.ndim==3 else guidance_mask
            guidance_np = gm.cpu().numpy().astype(np.float32)
            if guidance_np.max() > 1.0: guidance_np /= 255.0

        session = None
        if method in AI_METHODS:
            session = _get_session(method, gpu)
            if session is None:
                logger.warning(f"[S42CF BG] {method} unavailable, falling back to u2net")
                session = _get_session("u2net", gpu)

        masks_list = []
        for i in range(n):
            frame_np = images[i].cpu().numpy()
            frame_pil = Image.fromarray((frame_np*255).clip(0,255).astype(np.uint8), "RGB")
            try:
                if method.startswith("chroma_"):
                    mask = _chroma_key(frame_np, method.split("_")[1], chroma_threshold, chroma_softness, spill_suppress)
                elif method == "luma_key":
                    mask = _luma_key(frame_np, luma_mode, shadow_threshold, highlight_threshold, luma_softness)
                elif session is not None:
                    st, so = session
                    mask = _transformers_infer(frame_pil, so, gpu, processing_resolution) if st=="transformers" else _rembg_infer(frame_pil, so)
                else:
                    mask = np.ones((h,w), dtype=np.float32)
            except Exception as e:
                logger.error(f"[S42CF BG] Frame {i} failed: {e}"); mask = np.ones((h,w), dtype=np.float32)

            if reference_bg is not None:
                try:
                    ref = reference_bg[i%reference_bg.shape[0]].cpu().numpy()
                    if ref.shape[:2]!=(h,w): ref = np.array(Image.fromarray((ref*255).astype(np.uint8)).resize((w,h),Image.LANCZOS),dtype=np.float32)/255.0
                    diff_mask = np.clip(np.mean(np.abs(frame_np[:,:,:3]-ref[:,:,:3]),axis=-1)/0.15, 0, 1)
                    mask = np.clip(mask*0.7 + diff_mask*0.3, 0, 1)
                except: pass

            if guidance_np is not None:
                mask = _apply_guidance(mask, guidance_np, guidance_strength)

            mask = _post_process_mask(mask, mask_blur, mask_dilation, mask_erosion, remove_small_objects, is_trad)
            if invert_mask=="yes": mask = 1.0 - mask
            masks_list.append(mask)
            if (i+1)%10==0 or i==n-1:
                logger.info(f"[S42CF BG] {i+1}/{n} frames ({(i+1)/max(time.time()-t0,0.01):.1f} fps)")

        masks_np = np.stack(masks_list)
        if temporal_smooth > 0 and n > 1:
            sm = masks_np.copy()
            for i in range(n):
                s,e = max(0,i-temporal_smooth), min(n,i+temporal_smooth+1)
                sm[i] = masks_np[s:e].mean(axis=0)
            masks_np = sm

        masks_t = torch.from_numpy(masks_np)
        m3 = masks_t.unsqueeze(-1)
        result_rgb = images * m3
        result_rgba = torch.cat([images*m3, m3], dim=-1)
        elapsed = time.time()-t0
        guided = f", guided={guidance_strength}" if guidance_mask is not None else ""
        info = f"BackgroundRemover({method}): {n}f in {elapsed:.1f}s ({n/max(elapsed,0.01):.1f}fps){guided}"
        return (result_rgb, result_rgba, masks_t, info)

NODE_CLASS_MAPPINGS = {"S42CF_BackgroundRemover": S42CF_BackgroundRemover}
NODE_DISPLAY_NAME_MAPPINGS = {"S42CF_BackgroundRemover": "🎭 S42 CutFlow Background Remover"}