"""
DreamFit Checkpoint Loader Node
Loads DreamFit model weights for garment-centric generation
"""

import os
import torch
from pathlib import Path

from ..dreamfit_core.utils.model_loader import DreamFitModelManager

class DreamFitCheckpointLoader:
    """
    Loads DreamFit checkpoint files containing:
    - Anything-Dressing Encoder weights
    - LoRA adapter weights for Flux
    - Attention injection configurations
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # Get available models from the models directory
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            # Fallback if running outside ComfyUI
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        os.makedirs(models_dir, exist_ok=True)
        
        # Check which models are actually available
        available_models = []
        model_choices = []
        for model_name in ["flux_i2i", "flux_i2i_with_pose", "flux_tryon"]:
            model_path = os.path.join(models_dir, f"{model_name}.bin")
            if os.path.exists(model_path):
                available_models.append(model_name)
                model_choices.append(model_name)
            else:
                model_choices.append(f"{model_name} (not downloaded)")
        
        if not available_models:
            model_choices = ["Please run download_models.py first"]
        
        return {
            "required": {
                "model_name": (model_choices,),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }
    
    RETURN_TYPES = ("DREAMFIT_MODEL", "DREAMFIT_ENCODER", "DREAMFIT_CONFIG")
    RETURN_NAMES = ("dreamfit_model", "encoder", "config")
    FUNCTION = "load_checkpoint"
    CATEGORY = "DreamFit"
    
    def load_checkpoint(self, model_name, device, dtype):
        """
        Load DreamFit checkpoint and return model components
        """
        # Initialize model manager
        manager = DreamFitModelManager()
        
        # Check for invalid selection
        if model_name == "Please run download_models.py first":
            raise ValueError(
                "No DreamFit models found!\n\n"
                "Please download models first by running:\n"
                "python download_models.py\n\n"
                "From the ComfyUI-DreamFit directory"
            )
        
        # Check if user selected a not-downloaded model
        if " (not downloaded)" in model_name:
            actual_model_name = model_name.replace(" (not downloaded)", "")
            raise ValueError(
                f"Model '{actual_model_name}' is not downloaded!\n\n"
                f"Please download it first by running:\n"
                f"python download_models.py --model {actual_model_name}\n\n"
                f"From the ComfyUI-DreamFit directory"
            )
        
        # Get model path
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            # Fallback if running outside ComfyUI
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        model_path = os.path.join(models_dir, f"{model_name}.bin")
        
        # Final check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n\n"
                f"Please run: python download_models.py --model {model_name}"
            )
        
        # Map dtype string to torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        # Load the checkpoint
        print(f"Loading DreamFit checkpoint: {model_name}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract components from checkpoint
        dreamfit_model = {
            "lora_weights": checkpoint.get("lora_weights", {}),
            "attention_configs": checkpoint.get("attention_configs", {}),
            "model_type": model_name,
            "device": device,
            "dtype": torch_dtype,
        }
        
        # Initialize encoder with weights
        from ..dreamfit_core.models.anything_dressing_encoder import AnythingDressingEncoder
        encoder_config = checkpoint.get("encoder_config", {})
        encoder = AnythingDressingEncoder(encoder_config)
        
        # Load encoder weights
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder = encoder.to(device).to(torch_dtype)
        encoder.eval()
        
        # Extract configuration
        config = {
            "model_type": model_name,
            "encoder_config": encoder_config,
            "attention_injection": checkpoint.get("attention_injection", {}),
            "lora_configs": checkpoint.get("lora_configs", {}),
            "dtype": dtype,
            "device": device,
        }
        
        print(f"✓ Loaded DreamFit {model_name} successfully")
        
        return (dreamfit_model, encoder, config)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Force reload if model files change
        return float("nan")


# Helper class for model validation
class DreamFitModelValidator:
    """
    Validates loaded DreamFit models
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dreamfit_model": ("DREAMFIT_MODEL",),
                "encoder": ("DREAMFIT_ENCODER",),
                "config": ("DREAMFIT_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    FUNCTION = "validate"
    CATEGORY = "DreamFit/Debug"
    OUTPUT_NODE = True
    
    def validate(self, dreamfit_model, encoder, config):
        """
        Validate and display information about loaded model
        """
        info_lines = [
            f"Model Type: {config['model_type']}",
            f"Device: {config['device']}",
            f"Dtype: {config['dtype']}",
            f"LoRA modules: {len(dreamfit_model['lora_weights'])}",
            f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M",
        ]
        
        if config.get('attention_injection'):
            info_lines.append(f"Attention injection layers: {len(config['attention_injection'])}")
        
        info = "\n".join(info_lines)
        return (info,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DreamFitCheckpointLoader": DreamFitCheckpointLoader,
    "DreamFitModelValidator": DreamFitModelValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitCheckpointLoader": "DreamFit Checkpoint Loader",
    "DreamFitModelValidator": "DreamFit Model Validator",
}