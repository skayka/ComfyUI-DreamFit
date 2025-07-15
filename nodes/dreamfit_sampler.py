"""
DreamFit K-Sampler Node
Custom sampler for garment-centric generation with Flux
"""

import torch
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


class DreamFitKSampler:
    """
    Custom K-Sampler for DreamFit generation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except ImportError:
            # Fallback values if running outside ComfyUI
            samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                       "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                       "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc", "uni_pc_bh2"]
            schedulers = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
        
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "dreamfit_conditioning": ("DREAMFIT_CONDITIONING",),
                "guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        dreamfit_conditioning: Optional[Dict] = None,
        guidance_rescale: float = 0.0
    ):
        """
        Sample with DreamFit conditioning
        
        Args:
            model: The adapted Flux model
            seed: Random seed
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Name of the sampler
            scheduler: Name of the scheduler
            positive: Positive conditioning from CLIP
            negative: Negative conditioning from CLIP
            latent_image: Input latent
            denoise: Denoising strength
            dreamfit_conditioning: Optional DreamFit conditioning
            guidance_rescale: Guidance rescale factor
            
        Returns:
            Generated latent
        """
        # Check if model has DreamFit components
        has_dreamfit = hasattr(model, 'dreamfit_components') and dreamfit_conditioning is not None
        
        if has_dreamfit:
            # Apply DreamFit conditioning to positive conditioning
            positive = self._apply_dreamfit_conditioning(
                positive,
                dreamfit_conditioning,
                model.dreamfit_components
            )
        
        # Set up the sampler
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        
        # Handle Flux-specific setup if needed
        if hasattr(model, "model") and hasattr(model.model, "model_type"):
            if "flux" in str(model.model.model_type).lower():
                # Apply Flux-specific sampling configuration
                model = self._configure_flux_sampling(model)
        
        # Create callback for DreamFit feature injection if applicable
        callback = None
        if has_dreamfit:
            callback = self._create_injection_callback(
                model.dreamfit_components,
                dreamfit_conditioning
            )
        
        # Perform sampling
        import comfy.sample
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
            seed=seed,
            callback=callback
        )
        
        out = latent_image.copy()
        out["samples"] = samples
        
        return (out,)
    
    def _apply_dreamfit_conditioning(
        self,
        positive,
        dreamfit_conditioning: Dict,
        dreamfit_components: Dict
    ):
        """
        Apply DreamFit conditioning to positive conditioning
        
        Args:
            positive: Original positive conditioning
            dreamfit_conditioning: DreamFit conditioning dict
            dreamfit_components: DreamFit model components
            
        Returns:
            Modified positive conditioning
        """
        # For now, just return the original conditioning
        # TODO: Implement proper DreamFit conditioning integration
        return positive
    
    def _configure_flux_sampling(self, model):
        """
        Configure model for Flux-specific sampling
        
        Args:
            model: The Flux model
            
        Returns:
            Configured model
        """
        try:
            # Apply Flux-specific sampling configuration
            from comfy_extras.nodes_model_advanced import ModelSamplingFlux
            flux_sampler = ModelSamplingFlux()
            
            # Get Flux sampling parameters
            # These would typically come from the model config
            width = 1024
            height = 1024
            
            # Apply sampling configuration
            patched = flux_sampler.patch(
                model,
                width=width,
                height=height,
                # Add other Flux-specific parameters as needed
            )
            # Handle both tuple and direct return
            model = patched[0] if isinstance(patched, (list, tuple)) else patched
            
        except Exception as e:
            print(f"Warning: Failed to apply Flux sampling configuration: {e}")
        
        return model
    
    def _create_injection_callback(
        self,
        dreamfit_components: Dict,
        dreamfit_conditioning: Dict
    ):
        """
        Create callback for feature injection during sampling
        
        Args:
            dreamfit_components: DreamFit model components
            dreamfit_conditioning: DreamFit conditioning
            
        Returns:
            Callback function
        """
        injector = dreamfit_components.get("attention_injector")
        garment_features = dreamfit_conditioning["garment_features"]
        
        def injection_callback(step, x0, x, total_steps):
            """
            Callback to handle feature injection during sampling
            
            Args:
                step: Current step
                x0: Predicted clean image
                x: Current noisy image
                total_steps: Total number of steps
            """
            # This is a placeholder for the actual injection logic
            # In practice, this would coordinate with the hooks set up
            # in the adapter to inject features at the right time
            
            # Update injection strength based on step if progressive mode
            if dreamfit_conditioning["injection_config"]["injection_mode"] == "progressive":
                progress = step / total_steps
                # Reduce injection strength over time
                current_strength = dreamfit_conditioning["injection_config"]["injection_strength"] * (1 - progress)
                
                # Update injector configuration
                if injector:
                    injector.config.injection_strength = current_strength
        
        return injection_callback


class DreamFitSamplerAdvanced:
    """
    Advanced sampler with more control options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except ImportError:
            # Fallback values if running outside ComfyUI
            samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                       "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                       "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc", "uni_pc_bh2"]
            schedulers = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
        
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "dreamfit_conditioning": ("DREAMFIT_CONDITIONING",),
                "noise_mode": (["default", "garment_aware", "structured"], {"default": "default"}),
                "injection_schedule": (["constant", "linear", "cosine", "step"], {"default": "constant"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DreamFit"
    
    def sample(
        self,
        model,
        add_noise: bool,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: bool,
        dreamfit_conditioning: Optional[Dict] = None,
        noise_mode: str = "default",
        injection_schedule: str = "constant"
    ):
        """
        Advanced sampling with more control
        """
        # Check if model has DreamFit components
        has_dreamfit = hasattr(model, 'dreamfit_components') and dreamfit_conditioning is not None
        
        if has_dreamfit:
            # Apply DreamFit conditioning
            positive = self._apply_dreamfit_conditioning_advanced(
                positive,
                dreamfit_conditioning,
                model.dreamfit_components,
                injection_schedule
            )
        
        # Generate noise based on mode
        if add_noise:
            noise = self._generate_noise(
                latent_image["samples"],
                noise_seed,
                noise_mode,
                dreamfit_conditioning
            )
        else:
            noise = None
        
        # Adjust steps
        actual_end_step = min(end_at_step, steps)
        
        # Sample
        import comfy.sample
        samples = comfy.sample.sample_custom(
            model,
            noise=noise,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image["samples"],
            noise_seed=noise_seed,
            start_step=start_at_step,
            last_step=actual_end_step,
            force_full_denoise=not return_with_leftover_noise and actual_end_step >= steps,
            denoise=1.0,
            disable_pbar=False
        )
        
        out = latent_image.copy()
        out["samples"] = samples
        
        return (out,)
    
    def _apply_dreamfit_conditioning_advanced(
        self,
        positive,
        dreamfit_conditioning: Dict,
        dreamfit_components: Dict,
        injection_schedule: str
    ):
        """Apply advanced DreamFit conditioning"""
        # For now, just return the original conditioning
        # TODO: Implement proper DreamFit conditioning integration with schedules
        return positive
    
    def _generate_noise(
        self,
        latent: torch.Tensor,
        seed: int,
        mode: str,
        dreamfit_conditioning: Optional[Dict]
    ) -> torch.Tensor:
        """Generate noise based on mode"""
        batch_size, channels, height, width = latent.shape
        device = latent.device
        
        # Set random seed
        generator = torch.Generator(device=device).manual_seed(seed)
        
        if mode == "default":
            # Standard Gaussian noise
            noise = torch.randn(
                batch_size, channels, height, width,
                generator=generator,
                device=device
            )
            
        elif mode == "garment_aware" and dreamfit_conditioning:
            # Generate noise that's aware of garment regions
            noise = torch.randn(
                batch_size, channels, height, width,
                generator=generator,
                device=device
            )
            
            # Reduce noise in garment regions if mask is available
            if "garment_mask" in dreamfit_conditioning:
                mask = dreamfit_conditioning["garment_mask"]
                # Resize mask to match latent size
                mask = torch.nn.functional.interpolate(
                    mask,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                # Reduce noise strength in garment areas
                noise = noise * (1 - 0.3 * mask)
                
        elif mode == "structured":
            # Generate structured noise with patterns
            noise = torch.randn(
                batch_size, channels, height, width,
                generator=generator,
                device=device
            )
            
            # Add some structure
            freq = 8
            x = torch.linspace(0, freq * np.pi, width, device=device)
            y = torch.linspace(0, freq * np.pi, height, device=device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            pattern = torch.sin(xx) * torch.cos(yy) * 0.1
            noise = noise + pattern.unsqueeze(0).unsqueeze(0)
            
        else:
            # Fallback to default
            noise = torch.randn(
                batch_size, channels, height, width,
                generator=generator,
                device=device
            )
        
        return noise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitKSampler": DreamFitKSampler,
    "DreamFitSamplerAdvanced": DreamFitSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitKSampler": "DreamFit K-Sampler",
    "DreamFitSamplerAdvanced": "DreamFit Sampler Advanced",
}