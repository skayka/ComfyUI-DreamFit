"""
DreamFit Checkpoint Loader Node
Loads DreamFit model weights for garment-centric generation
"""

import os
import torch
from pathlib import Path

from dreamfit_core.utils.model_loader import DreamFitModelManager

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
        
        # List of available models
        model_files = ["flux_i2i.bin", "flux_i2i_with_pose.bin", "flux_tryon.bin"]
        
        # Check which models are already downloaded
        available_models = []
        for model in model_files:
            model_name = model.replace(".bin", "")
            if os.path.exists(os.path.join(models_dir, model)):
                available_models.append(f"{model_name} (downloaded)")
            else:
                available_models.append(f"{model_name} (download required)")
        
        return {
            "required": {
                "model_name": (["flux_i2i", "flux_i2i_with_pose", "flux_tryon"],),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "download_if_missing": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("DREAMFIT_MODEL", "DREAMFIT_ENCODER", "DREAMFIT_CONFIG")
    RETURN_NAMES = ("dreamfit_model", "encoder", "config")
    FUNCTION = "load_checkpoint"
    CATEGORY = "DreamFit"
    
    def load_checkpoint(self, model_name, device, dtype, download_if_missing):
        """
        Load DreamFit checkpoint and return model components
        """
        # Initialize model manager
        manager = DreamFitModelManager()
        
        # Get model path
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            # Fallback if running outside ComfyUI
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        model_path = os.path.join(models_dir, f"{model_name}.bin")
        
        # Download if needed
        if not os.path.exists(model_path):
            if download_if_missing:
                print(f"Downloading {model_name} model...")
                manager.download_model(model_name, models_dir)
            else:
                raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
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
        from dreamfit_core.models.anything_dressing_encoder import AnythingDressingEncoder
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