# DreamFit nodes for ComfyUI

from .dreamfit_sampler import DreamFitSampler

NODE_CLASS_MAPPINGS = {
    "DreamFitSampler": DreamFitSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitSampler": "DreamFit Sampler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']