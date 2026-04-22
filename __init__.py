"""
S42 CutFlow — Comprehensive Video Editing Suite for ComfyUI
==============================================================
42 core + expanded nodes for professional video editing.
Core: CapCut-class editing, speed ramping, transitions. 
Expanded: BG removal, compositing, LTX 2.3 bridge, Advanced Audio Mixing, 
Audio-Reactive Time Warping, and Semantic Latent Operations.

Repository: https://github.com/GeekyGhost/S42-CutFlow
License: MIT

Python 3.12 | ComfyUI Portable
"""

# ── Core nodes (42) ──────────────────────────────────────────────
from .cf_clip_ops import NODE_CLASS_MAPPINGS as CLIP_OPS_NODES, NODE_DISPLAY_NAME_MAPPINGS as CLIP_OPS_DISPLAY
from .cf_speed_ramp import NODE_CLASS_MAPPINGS as SPEED_NODES, NODE_DISPLAY_NAME_MAPPINGS as SPEED_DISPLAY
from .cf_filters import NODE_CLASS_MAPPINGS as FILTER_NODES, NODE_DISPLAY_NAME_MAPPINGS as FILTER_DISPLAY
from .cf_temporal import NODE_CLASS_MAPPINGS as TEMPORAL_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEMPORAL_DISPLAY
from .cf_text import NODE_CLASS_MAPPINGS as TEXT_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEXT_DISPLAY
from .cf_composition import NODE_CLASS_MAPPINGS as COMP_NODES, NODE_DISPLAY_NAME_MAPPINGS as COMP_DISPLAY
from .cf_stabilize import NODE_CLASS_MAPPINGS as STAB_NODES, NODE_DISPLAY_NAME_MAPPINGS as STAB_DISPLAY
from .cf_audio_sync import NODE_CLASS_MAPPINGS as AUDIO_NODES, NODE_DISPLAY_NAME_MAPPINGS as AUDIO_DISPLAY
from .cf_preview import NODE_CLASS_MAPPINGS as PREVIEW_NODES, NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_DISPLAY
from .cf_utilities import NODE_CLASS_MAPPINGS as UTIL_NODES, NODE_DISPLAY_NAME_MAPPINGS as UTIL_DISPLAY

# ── Expanded nodes ───────────────────────────────────────────────
from .cf_bg_remover import NODE_CLASS_MAPPINGS as BGRM_NODES, NODE_DISPLAY_NAME_MAPPINGS as BGRM_DISPLAY
from .cf_layer_composer import NODE_CLASS_MAPPINGS as LAYER_NODES, NODE_DISPLAY_NAME_MAPPINGS as LAYER_DISPLAY
from .cf_ltx_bridge import NODE_CLASS_MAPPINGS as LTX_NODES, NODE_DISPLAY_NAME_MAPPINGS as LTX_DISPLAY
from .cf_transitions import NODE_CLASS_MAPPINGS as TRANS_NODES, NODE_DISPLAY_NAME_MAPPINGS as TRANS_DISPLAY
from .cf_audio_advanced import NODE_CLASS_MAPPINGS as ADV_AUDIO_NODES, NODE_DISPLAY_NAME_MAPPINGS as ADV_AUDIO_DISPLAY

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mapping in [CLIP_OPS_NODES, SPEED_NODES, FILTER_NODES, TEMPORAL_NODES,
                TEXT_NODES, COMP_NODES, STAB_NODES, AUDIO_NODES,
                PREVIEW_NODES, UTIL_NODES,
                BGRM_NODES, LAYER_NODES, LTX_NODES, TRANS_NODES, ADV_AUDIO_NODES]:
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in [CLIP_OPS_DISPLAY, SPEED_DISPLAY, FILTER_DISPLAY, TEMPORAL_DISPLAY,
                TEXT_DISPLAY, COMP_DISPLAY, STAB_DISPLAY, AUDIO_DISPLAY,
                PREVIEW_DISPLAY, UTIL_DISPLAY,
                BGRM_DISPLAY, LAYER_DISPLAY, LTX_DISPLAY, TRANS_DISPLAY, ADV_AUDIO_DISPLAY]:
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_core = sum(len(m) for m in [CLIP_OPS_NODES, SPEED_NODES, FILTER_NODES, TEMPORAL_NODES,
                               TEXT_NODES, COMP_NODES, STAB_NODES, AUDIO_NODES,
                               PREVIEW_NODES, UTIL_NODES])
_expanded = sum(len(m) for m in [BGRM_NODES, LAYER_NODES, LTX_NODES, TRANS_NODES, ADV_AUDIO_NODES])
print(f"\033[93m[S42 CutFlow]\033[0m Loaded {_core} core + {_expanded} expanded = {len(NODE_CLASS_MAPPINGS)} total nodes")