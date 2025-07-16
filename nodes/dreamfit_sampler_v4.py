"""
DreamFit Sampler V4 - Proper implementation of read/write mechanism
Based on the official DreamFit sampling approach
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import numpy as np

# ComfyUI imports
import comfy.sample
import comfy.samplers
import comfy.model_management
import node_helpers


class DreamFitSamplerV4:
    """
    Custom sampler that implements DreamFit's read/write mechanism correctly.
    This sampler orchestrates the two-pass approach:
    1. Write pass: Process garment image to store features
    2. Read pass: Generate with stored garment features
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "guidance_rescale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "timestep_to_start_cfg": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DreamFit/Sampling"
    
    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float = 1.0,
        guidance_rescale: float = 0.0,
        timestep_to_start_cfg: int = 0
    ):
        """
        Execute DreamFit sampling with proper read/write mechanism
        """
        # Check if model has DreamFit components
        if not hasattr(model, 'dreamfit_features'):
            # Fallback to standard sampling if no DreamFit features
            print("Warning: Model doesn't have DreamFit features, using standard sampling")
            return self._standard_sample(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise
            )
        
        # Validate dreamfit_features
        if not isinstance(model.dreamfit_features, dict):
            print("Warning: dreamfit_features is not a dictionary, using standard sampling")
            return self._standard_sample(
                model, positive, negative, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise
            )
        
        # Get garment features from model
        garment_features = model.dreamfit_features
        dreamfit_config = getattr(model, 'dreamfit_config', {})
        
        print("DreamFit Sampler V4: Starting read/write sampling process")
        
        # Set up device and dtype using ComfyUI's management
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Get actual model (handle ModelPatcher)
        actual_model = model.model if hasattr(model, 'model') else model
        
        # Reset all DreamFit processors
        self._reset_processors(actual_model)
        
        # Prepare garment latent (same size as generation latent)
        garment_latent = self._prepare_garment_latent(
            latent_image["samples"],
            garment_features,
            device,
            dtype
        )
        
        # Create garment conditioning from positive conditioning
        garment_positive = self._create_garment_conditioning(positive, garment_features)
        garment_negative = self._create_garment_conditioning(negative, None)
        
        # Step 1: Write pass - Store garment features
        print("DreamFit Sampler V4: Executing write pass...")
        self._set_processor_mode(actual_model, "write")
        
        # Run single forward pass with garment to store features
        _ = self._execute_write_pass(
            model,
            garment_positive,
            garment_latent,
            device,
            dtype
        )
        
        # Also store negative features
        print("DreamFit Sampler V4: Executing negative write pass...")
        self._set_processor_mode(actual_model, "neg_write")
        
        _ = self._execute_write_pass(
            model,
            garment_negative,
            garment_latent,
            device,
            dtype
        )
        
        # Step 2: Read pass - Generate with stored features
        print("DreamFit Sampler V4: Executing read pass for generation...")
        self._set_processor_mode(actual_model, "read")
        
        # Use ComfyUI's standard sampling with our prepared model
        samples = comfy.sample.sample(
            model,
            noise=None,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            seed=seed,
            return_with_leftover_noise=False,
            callback=self._create_callback(actual_model, timestep_to_start_cfg)
        )
        
        # Reset processors to normal mode
        self._set_processor_mode(actual_model, "normal")
        self._reset_processors(actual_model)
        
        return ({"samples": samples},)
    
    def _standard_sample(
        self, model, positive, negative, latent_image,
        seed, steps, cfg, sampler_name, scheduler, denoise
    ):
        """Fallback to standard sampling"""
        samples = comfy.sample.sample(
            model,
            noise=None,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            seed=seed
        )
        return ({"samples": samples},)
    
    def _reset_processors(self, model):
        """Reset all DreamFit processors"""
        if hasattr(model, 'attn_processors'):
            for processor in model.attn_processors.values():
                if hasattr(processor, 'reset'):
                    try:
                        processor.reset()
                    except Exception as e:
                        print(f"Warning: Failed to reset processor: {e}")
    
    def _set_processor_mode(self, model, mode: str):
        """Set mode for all DreamFit processors"""
        if hasattr(model, 'attn_processors'):
            for processor in model.attn_processors.values():
                if hasattr(processor, 'current_mode'):
                    try:
                        processor.current_mode = mode
                    except Exception as e:
                        print(f"Warning: Failed to set processor mode: {e}")
    
    def _prepare_garment_latent(self, base_latent, garment_features, device, dtype):
        """Prepare garment latent for write pass"""
        # Create latent of same shape as generation
        B, C, H, W = base_latent.shape
        
        # Use ComfyUI's device management
        device = device or comfy.model_management.intermediate_device()
        
        # Option 1: Use zeros (clean latent)
        garment_latent = torch.zeros(
            (B, C, H, W),
            device=device,
            dtype=dtype
        )
        
        # Option 2: Create from garment features (if available)
        # This is a simplified version - actual implementation might be more complex
        if "patch_features" in garment_features and garment_features["patch_features"] is not None:
            # Reshape patch features to latent dimensions
            # This is a placeholder - actual reshaping would be more sophisticated
            pass
        
        return garment_latent
    
    def _create_garment_conditioning(self, base_conditioning, garment_features):
        """Create conditioning specifically for garment"""
        # Use node_helpers to properly set conditioning values
        garment_values = {
            "is_garment_pass": True
        }
        
        if garment_features:
            garment_values["garment_features"] = garment_features
        
        # Use ComfyUI's proper conditioning manipulation
        garment_cond = node_helpers.conditioning_set_values(
            base_conditioning,
            garment_values
        )
        
        return garment_cond
    
    def _execute_write_pass(self, model, conditioning, latent, device, dtype):
        """Execute a single forward pass to store features"""
        # Get timestep 0 (no noise)
        timestep = torch.zeros((latent.shape[0],), device=device, dtype=torch.long)
        
        # For write pass with timestep=0, model_input is just the latent
        model_input = latent.to(device, dtype=dtype)
        
        # Execute forward pass
        with torch.no_grad():
            # This will trigger the write mode in processors
            if hasattr(model, 'apply_model'):
                try:
                    # Use ComfyUI's model interface
                    _ = model.apply_model(
                        model_input,
                        timestep,
                        c=conditioning
                    )
                except Exception as e:
                    print(f"Warning: Write pass failed: {e}")
                    raise
            else:
                # Model doesn't have apply_model method
                print("Warning: Model doesn't have apply_model method")
                raise ValueError("Model must have apply_model method for DreamFit sampling")
        
        return None
    
    def _create_callback(self, model, timestep_to_start_cfg):
        """Create callback to switch between read modes during sampling"""
        def callback(step, total_steps, x0, x, timestep):
            # Switch to neg_read mode after certain timestep for CFG
            if step >= timestep_to_start_cfg:
                # Model should already be in read mode
                # This is where we could switch between read/neg_read if needed
                pass
            return x
        
        return callback
    
    @classmethod
    def VALIDATE_INPUTS(cls, model, **kwargs):
        """Validate inputs before execution"""
        # Check if model has required methods
        if not hasattr(model, 'apply_model'):
            return "Model must have apply_model method for DreamFit sampling"
        
        # Check if model has DreamFit features when expected
        if hasattr(model, 'dreamfit_features') and not isinstance(model.dreamfit_features, dict):
            return "Model's dreamfit_features must be a dictionary"
        
        return True


# Node registration
NODE_CLASS_MAPPINGS = {
    "DreamFitSamplerV4": DreamFitSamplerV4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitSamplerV4": "DreamFit Sampler V4",
}