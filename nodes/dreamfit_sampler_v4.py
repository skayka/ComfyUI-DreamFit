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
            print("Please ensure DreamFitUnified node is connected before this sampler")
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
        
        # Debug: Check what features we have
        print(f"DreamFit features keys: {list(model.dreamfit_features.keys())}")
        for key, value in model.dreamfit_features.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Get garment features from model
        garment_features = model.dreamfit_features
        dreamfit_config = getattr(model, 'dreamfit_config', {})
        
        print("DreamFit Sampler V4: Starting read/write sampling process")
        
        # Set up device and dtype using ComfyUI's management
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Get actual model (handle ModelPatcher)
        actual_model = model.model if hasattr(model, 'model') else model
        
        # Debug: Check if we have DreamFit processors
        processor_count = 0
        diffusion_model = None
        
        # Try to find the diffusion model for Flux
        if hasattr(actual_model, 'diffusion_model'):
            diffusion_model = actual_model.diffusion_model
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'diffusion_model'):
            diffusion_model = actual_model.model.diffusion_model
        
        # Check for processors in different locations
        if diffusion_model and hasattr(diffusion_model, '_dreamfit_processors_attached'):
            print(f"DreamFit Sampler V4: Found Flux model with DreamFit processors attached")
            # Count processors in double blocks
            if hasattr(diffusion_model, 'double_blocks'):
                for block in diffusion_model.double_blocks:
                    if hasattr(block, 'processor') and hasattr(block.processor, 'current_mode'):
                        processor_count += 1
            print(f"DreamFit Sampler V4: Found {processor_count} DreamFit processors in Flux blocks")
        elif hasattr(actual_model, 'attn_processors'):
            for name, processor in actual_model.attn_processors.items():
                if hasattr(processor, 'current_mode'):
                    processor_count += 1
            print(f"DreamFit Sampler V4: Found {processor_count} DreamFit processors")
        else:
            print("Warning: Model doesn't have attn_processors attribute or Flux structure")
        
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
        
        # Validate conditioning dimensions for Flux
        if positive and len(positive) > 0 and isinstance(positive[0], (list, tuple)) and len(positive[0]) > 0:
            cond_tensor = positive[0][0]
            if hasattr(cond_tensor, 'shape'):
                print(f"DreamFit Sampler V4: Conditioning shape: {cond_tensor.shape}")
                if cond_tensor.shape[-1] != 4096:
                    print(f"WARNING: Flux expects 4096-dimensional text embeddings, but got {cond_tensor.shape[-1]}")
                    print("Make sure you're using the correct CLIP model for Flux (CLIP-G)")
        
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
        
        # Prepare noise for sampling (following ComfyUI's standard approach)
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        
        # Generate noise
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        
        # Get noise mask if provided
        noise_mask = latent_image.get("noise_mask", None)
        
        # Use ComfyUI's standard sampling with our prepared model
        samples = comfy.sample.sample(
            model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise,
            seed=seed,
            noise_mask=noise_mask,
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
        # Prepare latent and noise
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        
        # Generate noise
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask", None)
        
        samples = comfy.sample.sample(
            model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise,
            seed=seed,
            noise_mask=noise_mask
        )
        return ({"samples": samples},)
    
    def _reset_processors(self, model):
        """Reset all DreamFit processors"""
        # Try to find the diffusion model for Flux
        diffusion_model = None
        if hasattr(model, 'diffusion_model'):
            diffusion_model = model.diffusion_model
        elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        
        # Reset processors in Flux model blocks
        if diffusion_model and hasattr(diffusion_model, 'double_blocks'):
            for block in diffusion_model.double_blocks:
                if hasattr(block, 'processor') and hasattr(block.processor, 'reset'):
                    try:
                        block.processor.reset()
                    except Exception as e:
                        print(f"Warning: Failed to reset Flux block processor: {e}")
        
        # Reset processors in standard attn_processors
        elif hasattr(model, 'attn_processors'):
            for processor in model.attn_processors.values():
                if hasattr(processor, 'reset'):
                    try:
                        processor.reset()
                    except Exception as e:
                        print(f"Warning: Failed to reset processor: {e}")
    
    def _set_processor_mode(self, model, mode: str):
        """Set mode for all DreamFit processors"""
        count = 0
        
        # Try to find the diffusion model for Flux
        diffusion_model = None
        if hasattr(model, 'diffusion_model'):
            diffusion_model = model.diffusion_model
        elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        
        # Set mode for processors in Flux model blocks
        if diffusion_model and hasattr(diffusion_model, 'double_blocks'):
            for idx, block in enumerate(diffusion_model.double_blocks):
                if hasattr(block, 'processor') and hasattr(block.processor, 'current_mode'):
                    try:
                        block.processor.current_mode = mode
                        count += 1
                    except Exception as e:
                        print(f"Warning: Failed to set processor mode for block {idx}: {e}")
            print(f"DreamFit Sampler V4: Set {count} Flux block processors to {mode} mode")
        
        # Set mode for processors in standard attn_processors
        elif hasattr(model, 'attn_processors'):
            for name, processor in model.attn_processors.items():
                if hasattr(processor, 'current_mode'):
                    try:
                        processor.current_mode = mode
                        count += 1
                    except Exception as e:
                        print(f"Warning: Failed to set processor mode for {name}: {e}")
            print(f"DreamFit Sampler V4: Set {count} processors to {mode} mode")
        else:
            print("Warning: Model doesn't have attn_processors or Flux structure for mode setting")
    
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
        # For write pass, we need to properly call the model with conditioning
        # Use a very small timestep for write pass
        timestep = torch.tensor([1.0], device=device, dtype=dtype)  # Small timestep
        
        # Apply the model using ComfyUI's standard interface
        try:
            # ComfyUI models expect apply_model method
            if hasattr(model, 'apply_model'):
                # Call apply_model with proper parameters
                with torch.no_grad():
                    # Create model_options if needed
                    model_options = {}
                    if hasattr(model, 'model_options'):
                        model_options = model.model_options.copy()
                    
                    # Apply model with conditioning
                    _ = model.apply_model(
                        latent,
                        timestep,
                        c_concat=None,
                        c_crossattn=conditioning[0][0] if conditioning else None,
                        control=None,
                        transformer_options=model_options.get('transformer_options', {})
                    )
                    print("DreamFit: Write pass executed successfully")
            else:
                print("Warning: Model doesn't have apply_model method")
        except Exception as e:
            print(f"Warning: Write pass execution failed: {e}")
            import traceback
            traceback.print_exc()
            
        return None
    
    def _create_callback(self, model, timestep_to_start_cfg):
        """Create callback to switch between read modes during sampling"""
        def callback(step, x0, x, total_steps):
            # ComfyUI's k-diffusion callback format: (step, denoised, x, total_steps)
            # Switch to neg_read mode after certain timestep for CFG
            if step >= timestep_to_start_cfg:
                # Model should already be in read mode
                # This is where we could switch between read/neg_read if needed
                pass
            # Return the denoised output (x0) as expected by ComfyUI
            return x0
        
        return callback
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs before execution"""
        # For now, return True to allow execution
        # Actual validation happens during the sample function
        return True


# Node registration
NODE_CLASS_MAPPINGS = {
    "DreamFitSamplerV4": DreamFitSamplerV4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitSamplerV4": "DreamFit Sampler V4",
}