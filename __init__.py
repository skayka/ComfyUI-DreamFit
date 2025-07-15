"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI using DreamFit with Flux
Based on: https://github.com/bytedance/DreamFit
"""

# Import node mappings directly - ComfyUI expects these at module level
from .nodes.dreamfit_loader import DreamFitCheckpointLoader, DreamFitModelValidator
from .nodes.dreamfit_encode import DreamFitEncode
from .nodes.dreamfit_adapter import DreamFitFluxAdapter
from .nodes.dreamfit_sampler import DreamFitKSampler, DreamFitSamplerAdvanced

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']