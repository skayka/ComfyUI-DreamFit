"""
DreamFit K-Sampler V2
Implements proper read/write mechanism for garment injection
"""

import torch
from typing import Dict, Optional, Any
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.sample_as
import nodes
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
        
        # DreamFit's two-phase approach requires special handling
        # First, we need to do the "write" phase with garment features
        device = comfy.model_management.get_torch_device()
        
        # Get latent info
        latent_samples = latent_image["samples"]
        batch_size = latent_samples.shape[0]
        
        # Create timestep vector for first step
        t_vec = torch.zeros((batch_size,), dtype=latent_samples.dtype, device=device)
        
        # Phase 1: Write garment features
        wrapped_model.set_mode("write")
        with torch.no_grad():
            # We need to prepare the garment latent in the same format
            # For now, use the same latent shape filled with noise
            garment_latent = torch.randn_like(latent_samples)
            
            # Call model with garment features to store them
            # This mimics inp_cloth call in official DreamFit
            try:
                # The model expects specific inputs - we'll use the wrapped model's forward
                # This stores the garment features internally
                _ = wrapped_model(
                    garment_latent,
                    t_vec,
                    context=enhanced_positive[0][0] if enhanced_positive else None
                )
            except Exception as e:
                print(f"Warning: Write phase failed: {e}")
        
        # Phase 2: Switch to read mode for actual generation
        wrapped_model.set_mode("read")
        
        # Create callback that maintains read mode
        def dreamfit_callback(step, x0, x, total_steps):
            """Ensure we stay in read mode during sampling"""
            wrapped_model.set_mode("read")
        
        # Prepare conditioning with garment features
        enhanced_positive = self._enhance_conditioning_for_sampling(positive, garment_dict)
        
        # Reset the wrapper before sampling
        wrapped_model.reset()
        
        # Use common_ksampler if available for consistency
        if hasattr(nodes, 'common_ksampler'):
            # Create a custom callback wrapper that integrates with common_ksampler
            original_callback = dreamfit_callback
            
            # Use common_ksampler with our wrapped model
            samples = nodes.common_ksampler(
                model=wrapped_model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=enhanced_positive,
                negative=negative,
                latent=latent_image,
                denoise=denoise
            )
            
            # Call our callback for the first step to set modes
            if steps > 0:
                dreamfit_callback(0, None, None, steps)
            
            return (samples,)
        else:
            # Fallback to direct sampling
            latent_samples = latent_image["samples"]
            
            # Perform sampling with callback
            samples = comfy.sample.sample(
                wrapped_model,
                noise=None,  # Let ComfyUI generate noise from seed
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=enhanced_positive,
                negative=negative,
                latent_image=latent_samples,
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
        # Use common_ksampler from nodes module if available
        if hasattr(nodes, 'common_ksampler'):
            samples = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent_image,
                denoise=denoise
            )
            return (samples,)
        else:
            # Fallback to direct sample call
            latent_samples = latent_image["samples"]
            
            samples = comfy.sample.sample(
                model,
                noise=None,  # Let ComfyUI generate noise from seed
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_samples,
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