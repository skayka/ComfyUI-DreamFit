"""
ComfyUI-DreamFit
Garment-centric human generation nodes for ComfyUI using DreamFit with Flux
Based on: https://github.com/bytedance/DreamFit
"""

# Import all node classes directly
from .nodes.dreamfit_loader import DreamFitCheckpointLoader, DreamFitModelValidator
from .nodes.dreamfit_encode import DreamFitEncode
from .nodes.dreamfit_adapter import DreamFitFluxAdapter
from .nodes.dreamfit_sampler import DreamFitKSampler, DreamFitSamplerAdvanced
from .nodes.dreamfit_adapter_v2 import DreamFitFluxAdapterV2
from .nodes.dreamfit_simple import DreamFitSimple

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DreamFitCheckpointLoader": DreamFitCheckpointLoader,
    "DreamFitModelValidator": DreamFitModelValidator,
    "DreamFitEncode": DreamFitEncode,
    "DreamFitFluxAdapter": DreamFitFluxAdapter,
    "DreamFitKSampler": DreamFitKSampler,
    "DreamFitSamplerAdvanced": DreamFitSamplerAdvanced,
    "DreamFitFluxAdapterV2": DreamFitFluxAdapterV2,
    "DreamFitSimple": DreamFitSimple,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitCheckpointLoader": "DreamFit Checkpoint Loader",
    "DreamFitModelValidator": "DreamFit Model Validator",
    "DreamFitEncode": "DreamFit Encode",
    "DreamFitFluxAdapter": "DreamFit Flux Adapter",
    "DreamFitKSampler": "DreamFit K-Sampler",
    "DreamFitSamplerAdvanced": "DreamFit Sampler Advanced",
    "DreamFitFluxAdapterV2": "DreamFit Flux Adapter V2",
    "DreamFitSimple": "DreamFit Simple",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']