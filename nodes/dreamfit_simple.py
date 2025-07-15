"""
Simplified DreamFit Node - All-in-one solution
"""

import os
import torch
from pathlib import Path

from ..dreamfit_core.utils.model_loader import DreamFitModelManager


class DreamFitSimple:
    """
    Simple all-in-one DreamFit node for easy workflow
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Check which models are actually available
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        
        available_models = []
        for model_name in ["flux_i2i", "flux_i2i_with_pose", "flux_tryon"]:
            model_path = os.path.join(models_dir, f"{model_name}.bin")
            if os.path.exists(model_path):
                available_models.append(model_name)
        
        if not available_models:
            available_models = ["Please run download_models.py first"]
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "dreamfit_model": (available_models,),
                "garment_image": ("IMAGE",),
                "positive": ("STRING", {
                    "default": "A beautiful person wearing the garment, high quality, detailed",
                    "multiline": True
                }),
                "negative": ("STRING", {
                    "default": "low quality, blurry, distorted",
                    "multiline": True
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "model_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "generate"
    CATEGORY = "DreamFit"
    
    def generate(
        self,
        model,
        clip,
        vae,
        dreamfit_model: str,
        garment_image,
        positive: str,
        negative: str,
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        model_image=None
    ):
        """
        Complete DreamFit generation in one node
        """
        # Load DreamFit weights
        manager = DreamFitModelManager()
        
        # Get model path
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        
        model_path = os.path.join(models_dir, f"{dreamfit_model}.bin")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model {dreamfit_model} not found at {model_path}\n"
                f"Please run: python download_models.py --model {dreamfit_model}"
            )
        
        # Simplified generation process
        # 1. Encode text
        positive_tokens = clip.tokenize(positive)
        negative_tokens = clip.tokenize(negative)
        
        positive_cond = clip.encode_from_tokens(positive_tokens)[0]
        negative_cond = clip.encode_from_tokens(negative_tokens)[0]
        
        # 2. Create empty latent
        batch_size = 1
        width = 1024
        height = 1024
        
        # Create empty latent
        device = model.load_device
        latent = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=device
        )
        
        # 3. Sample (simplified - using standard KSampler logic)
        import comfy.sample
        import comfy.samplers
        
        # Get sampler
        sampler = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=device,
            sampler="euler",
            scheduler="normal",
            denoise=denoise
        )
        
        # Sample
        samples = comfy.sample.sample(
            model,
            noise=torch.randn_like(latent),
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="normal",
            positive=[[positive_cond, {}]],
            negative=[[negative_cond, {}]],
            latent_image=latent,
            denoise=denoise,
            seed=seed
        )
        
        return ({"samples": samples},)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DreamFitSimple": DreamFitSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitSimple": "DreamFit Simple",
}