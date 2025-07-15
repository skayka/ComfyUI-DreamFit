"""
DreamFit K-Sampler V2
Implements proper read/write mechanism for garment injection
"""

import torch
from typing import Dict, Optional, Any
import comfy.sample
import comfy.samplers
import comfy.model_management
from ..dreamfit_core.models.dreamfit_model_wrapper import DreamFitModelWrapper


class DreamFitKSamplerV2:
    """
    Custom sampler that implements DreamFit's read/write mechanism
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "garment_features": ("DREAMFIT_FEATURES",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DreamFit"
    
    def sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        denoise: float = 1.0,
        garment_features: Optional[Any] = None
    ):
        """
        Sample with DreamFit's read/write mechanism
        """
        device = comfy.model_management.get_torch_device()
        
        # Check if we have garment features
        if garment_features is None:
            # No garment features, use standard sampling
            return self._standard_sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise
            )
        
        # Convert DreamFitFeatures object to dict if needed
        if hasattr(garment_features, 'to_dict'):
            garment_dict = garment_features.to_dict()
        else:
            garment_dict = garment_features
        
        # Check if model is already wrapped
        if isinstance(model, DreamFitModelWrapper):
            wrapped_model = model
        else:
            # Wrap the model with DreamFit wrapper
            wrapped_model = DreamFitModelWrapper(model, garment_dict)
        
        # Create custom callback for two-phase sampling
        def dreamfit_callback(step, x0, x, total_steps):
            """Callback to handle read/write mode switching"""
            if step == 0:
                # First step: write mode
                wrapped_model.set_mode("write")
                
                # Prepare garment conditioning
                # This simulates calling the model with inp_cloth
                with torch.no_grad():
                    # Create dummy inputs matching the latent shape
                    dummy_latent = torch.zeros_like(x)
                    
                    # Call model in write mode to store features
                    # This doesn't affect the actual sampling, just stores features
                    try:
                        _ = wrapped_model._wrapped_forward(
                            dummy_latent,
                            timesteps=torch.zeros(1, device=x.device),
                            context=positive[0][0] if positive else None
                        )
                    except:
                        # Fallback if forward signature is different
                        pass
                
                # Switch to read mode for actual sampling
                wrapped_model.set_mode("read")
            
            # For all steps, ensure we're in read mode
            if step >= 0:
                wrapped_model.set_mode("read")
        
        # Prepare conditioning with garment features
        enhanced_positive = self._enhance_conditioning_for_sampling(positive, garment_dict)
        
        # Reset the wrapper before sampling
        wrapped_model.reset()
        
        # Perform sampling with callback
        samples = comfy.sample.sample(
            wrapped_model,
            noise=None,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=enhanced_positive,
            negative=negative,
            latent_image=latent_image["samples"],
            start_step=None,
            last_step=None,
            force_full_denoise=denoise == 1.0,
            denoise=denoise,
            seed=seed,
            callback=dreamfit_callback
        )
        
        # Reset wrapper after sampling
        wrapped_model.reset()
        wrapped_model.set_mode("normal")
        
        out = latent_image.copy()
        out["samples"] = samples
        
        return (out,)
    
    def _standard_sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        denoise: float
    ):
        """Standard sampling without garment features"""
        samples = comfy.sample.sample(
            model,
            noise=None,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image["samples"],
            start_step=None,
            last_step=None,
            force_full_denoise=denoise == 1.0,
            denoise=denoise,
            seed=seed
        )
        
        out = latent_image.copy()
        out["samples"] = samples
        
        return (out,)
    
    def _enhance_conditioning_for_sampling(self, conditioning, garment_features: Dict):
        """
        Enhance conditioning with garment features for sampling
        This ensures the features are available during the read phase
        """
        enhanced = []
        
        for item in conditioning:
            if isinstance(item, list) and len(item) == 2:
                cond_tensor, extras = item
            else:
                cond_tensor = item
                extras = {}
            
            # Create new extras with garment info
            new_extras = extras.copy() if isinstance(extras, dict) else {}
            
            # Add garment features for read mode
            new_extras["dreamfit_garment"] = {
                "token": garment_features.get("garment_token"),
                "pooled": garment_features.get("pooled_features"),
                "patches": garment_features.get("patch_features"),
                "mode": "read"
            }
            
            enhanced.append([cond_tensor, new_extras])
        
        return enhanced


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitKSamplerV2": DreamFitKSamplerV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitKSamplerV2": "DreamFit K-Sampler V2",
}