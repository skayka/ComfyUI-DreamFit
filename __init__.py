"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI using DreamFit with Flux
Based on: https://github.com/bytedance/DreamFit
"""

import os
import sys

# Add the current directory to Python path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the nodes
try:
    from nodes.dreamfit_loader import DreamFitCheckpointLoader, DreamFitModelValidator
    from nodes.dreamfit_encode import DreamFitEncode
    from nodes.dreamfit_adapter import DreamFitFluxAdapter
    from nodes.dreamfit_sampler import DreamFitKSampler, DreamFitSamplerAdvanced
    
    NODE_CLASS_MAPPINGS = {
        "DreamFitCheckpointLoader": DreamFitCheckpointLoader,
        "DreamFitModelValidator": DreamFitModelValidator,
        "DreamFitEncode": DreamFitEncode,
        "DreamFitFluxAdapter": DreamFitFluxAdapter,
        "DreamFitKSampler": DreamFitKSampler,
        "DreamFitSamplerAdvanced": DreamFitSamplerAdvanced,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DreamFitCheckpointLoader": "DreamFit Checkpoint Loader",
        "DreamFitModelValidator": "DreamFit Model Validator",
        "DreamFitEncode": "DreamFit Encode",
        "DreamFitFluxAdapter": "DreamFit Flux Adapter",
        "DreamFitKSampler": "DreamFit K-Sampler",
        "DreamFitSamplerAdvanced": "DreamFit Sampler Advanced",
    }
    
    print("[ComfyUI-DreamFit] Nodes loaded successfully")
    
except ImportError as e:
    print(f"[ComfyUI-DreamFit] Failed to import nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']