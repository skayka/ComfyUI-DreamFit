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
from .nodes.dreamfit_sampler_v2 import DreamFitKSamplerV2
from .nodes.dreamfit_sampler_v3 import DreamFitKSamplerV3
from .nodes.dreamfit_adapter_v2 import DreamFitFluxAdapterV2
from .nodes.dreamfit_adapter_v3 import DreamFitFluxAdapterV3
from .nodes.dreamfit_unified import DreamFitUnified
from .nodes.dreamfit_unified_v2 import DreamFitUnifiedV2
from .nodes.dreamfit_simple import DreamFitSimple
from .nodes.dreamfit_sampler_v4 import DreamFitSamplerV4

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DreamFitCheckpointLoader": DreamFitCheckpointLoader,
    "DreamFitModelValidator": DreamFitModelValidator,
    "DreamFitEncode": DreamFitEncode,
    "DreamFitFluxAdapter": DreamFitFluxAdapter,
    "DreamFitKSampler": DreamFitKSampler,
    "DreamFitSamplerAdvanced": DreamFitSamplerAdvanced,
    "DreamFitKSamplerV2": DreamFitKSamplerV2,
    "DreamFitKSamplerV3": DreamFitKSamplerV3,
    "DreamFitFluxAdapterV2": DreamFitFluxAdapterV2,
    "DreamFitFluxAdapterV3": DreamFitFluxAdapterV3,
    "DreamFitUnified": DreamFitUnified,
    "DreamFitUnifiedV2": DreamFitUnifiedV2,
    "DreamFitSimple": DreamFitSimple,
    "DreamFitSamplerV4": DreamFitSamplerV4,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitCheckpointLoader": "DreamFit Checkpoint Loader",
    "DreamFitModelValidator": "DreamFit Model Validator",
    "DreamFitEncode": "DreamFit Encode",
    "DreamFitFluxAdapter": "DreamFit Flux Adapter",
    "DreamFitKSampler": "DreamFit K-Sampler",
    "DreamFitSamplerAdvanced": "DreamFit Sampler Advanced",
    "DreamFitKSamplerV2": "DreamFit K-Sampler V2",
    "DreamFitKSamplerV3": "DreamFit K-Sampler V3 (True Implementation)",
    "DreamFitFluxAdapterV2": "DreamFit Flux Adapter V2",
    "DreamFitFluxAdapterV3": "DreamFit Flux Adapter V3",
    "DreamFitUnified": "DreamFit Unified",
    "DreamFitUnifiedV2": "DreamFit Unified V2",
    "DreamFitSimple": "DreamFit Simple",
    "DreamFitSamplerV4": "DreamFit Sampler V4 (Read/Write)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']