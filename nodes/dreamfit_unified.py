"""
DreamFit Unified Node - Properly integrated with ComfyUI
Based on official DreamFit implementation
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import copy
import numpy as np
from PIL import Image
import re

# ComfyUI imports
import comfy.model_management
import comfy.utils

# DreamFit imports
from ..dreamfit_core.models.dreamfit_attention import DreamFitDoubleStreamProcessor, DreamFitSingleStreamProcessor
from ..dreamfit_core.models.anything_dressing_encoder import AnythingDressingEncoder, EncoderConfig
from ..dreamfit_core.utils.image_processing import preprocess_garment_image
from ..dreamfit_core.utils.model_loader import DreamFitModelManager


class DreamFitUnified:
    """
    Unified DreamFit node that properly integrates with ComfyUI's model system.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models
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
            available_models = ["Please download models first"]
        
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "garment_image": ("IMAGE",),
                "dreamfit_model": (available_models,),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "model_image": ("IMAGE",),
                "injection_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "lora_rank": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 128,
                    "step": 4
                }),
                "double_blocks": ("STRING", {
                    "default": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
                    "multiline": False
                }),
                "single_blocks": ("STRING", {
                    "default": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "debug_garment")
    FUNCTION = "apply_dreamfit"
    CATEGORY = "DreamFit"
    
    def apply_dreamfit(
        self,
        model,
        positive,
        negative,
        garment_image,
        dreamfit_model: str,
        strength: float = 1.0,
        model_image: Optional[torch.Tensor] = None,
        injection_strength: float = 0.8,
        lora_rank: int = 32,
        double_blocks: str = None,
        single_blocks: str = None
    ):
        """
        Apply DreamFit to a Flux model by attaching custom attention processors.
        """
        # Validate model selection
        if dreamfit_model == "Please download models first":
            raise ValueError(
                "No DreamFit models found!\n"
                "Please download models by running:\n"
                "python download_models.py"
            )
        
        # Load DreamFit checkpoint
        print(f"Loading DreamFit model: {dreamfit_model}")
        checkpoint = self._load_dreamfit_checkpoint(dreamfit_model)
        
        # Initialize encoder
        print("Initializing Anything-Dressing Encoder...")
        encoder = self._initialize_encoder(checkpoint)
        
        # Process garment image
        print("Processing garment image...")
        processed_garment, garment_features = self._process_garment_image(
            garment_image, encoder, model_image
        )
        
        # Clone model to avoid modifying original
        adapted_model = copy.deepcopy(model)
        
        # Parse block indices
        double_blocks_idx = self._parse_block_indices(double_blocks, default_max=19)
        single_blocks_idx = self._parse_block_indices(single_blocks, default_max=38)
        
        # Attach DreamFit processors
        print("Attaching DreamFit attention processors...")
        self._attach_dreamfit_processors(
            adapted_model,
            checkpoint,
            garment_features,
            strength,
            injection_strength,
            lora_rank,
            double_blocks_idx,
            single_blocks_idx
        )
        
        # Store garment features in model for sampler access
        adapted_model.dreamfit_features = garment_features
        adapted_model.dreamfit_config = {
            "model_type": dreamfit_model,
            "strength": strength,
            "injection_strength": injection_strength,
            "has_model_image": model_image is not None
        }
        
        # Convert processed garment back to ComfyUI format for debug output
        debug_garment = self._tensor_to_comfyui_image(processed_garment)
        
        return (adapted_model, positive, negative, debug_garment)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs before execution"""
        # For now, return True to allow execution
        # Actual validation happens during the execute function
        return True
    
    def _load_dreamfit_checkpoint(self, model_name: str) -> Dict:
        """Load DreamFit checkpoint"""
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
        except ImportError:
            models_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamfit")
        
        model_path = os.path.join(models_dir, f"{model_name}.bin")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model {model_name} not found at {model_path}\n"
                f"Please run: python download_models.py --model {model_name}"
            )
        
        device = comfy.model_management.get_torch_device()
        checkpoint = torch.load(model_path, map_location=device)
        
        return checkpoint
    
    def _initialize_encoder(self, checkpoint: Dict) -> AnythingDressingEncoder:
        """Initialize the Anything-Dressing Encoder"""
        encoder_config_dict = checkpoint.get("encoder_config", {})
        encoder_config = EncoderConfig(**encoder_config_dict) if encoder_config_dict else EncoderConfig()
        
        encoder = AnythingDressingEncoder(encoder_config)
        
        # Load encoder weights
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        
        device = comfy.model_management.get_torch_device()
        encoder = encoder.to(device).eval()
        
        return encoder
    
    def _process_garment_image(
        self, 
        garment_image: torch.Tensor, 
        encoder: AnythingDressingEncoder,
        model_image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Process garment image and extract features"""
        device = comfy.model_management.get_torch_device()
        
        # Preprocess garment image
        processed_garment = preprocess_garment_image(
            garment_image,
            target_size=224,
            normalize=True,
            device=device
        )
        
        # Extract features
        with torch.no_grad():
            encoder_output = encoder(processed_garment)
        
        garment_features = {
            "garment_token": encoder_output["garment_token"],
            "pooled_features": encoder_output["pooled_features"],
            "patch_features": encoder_output["patch_features"],
            "features": encoder_output.get("features", {}),
            "attention_weights": encoder_output.get("attention_weights", {})
        }
        
        # Process model image if provided
        if model_image is not None:
            processed_model = preprocess_garment_image(
                model_image,
                target_size=224,
                normalize=True,
                device=device
            )
            
            with torch.no_grad():
                model_output = encoder(processed_model)
            
            garment_features["model_features"] = {
                "pose_token": model_output["garment_token"],
                "pose_patches": model_output["patch_features"]
            }
        
        return processed_garment, garment_features
    
    def _parse_block_indices(self, block_str: Optional[str], default_max: int) -> List[int]:
        """Parse block indices from string"""
        if not block_str:
            return list(range(default_max))
        
        if block_str.strip() == "":
            return []
        
        try:
            indices = [int(idx.strip()) for idx in block_str.split(",")]
            return indices
        except ValueError:
            print(f"Warning: Invalid block indices '{block_str}', using defaults")
            return list(range(default_max))
    
    def _attach_dreamfit_processors(
        self,
        model,
        checkpoint: Dict,
        garment_features: Dict,
        strength: float,
        injection_strength: float,
        lora_rank: int,
        double_blocks_idx: List[int],
        single_blocks_idx: List[int]
    ):
        """Attach DreamFit attention processors to the model"""
        # Get the actual model (handle ComfyUI ModelPatcher)
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        # For ComfyUI Flux models, we need to check different attributes
        # Try to find the diffusion model
        diffusion_model = None
        if hasattr(actual_model, 'diffusion_model'):
            diffusion_model = actual_model.diffusion_model
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'diffusion_model'):
            diffusion_model = actual_model.model.diffusion_model
        
        # Check if we found a Flux-like model with double_blocks and single_blocks
        if diffusion_model and hasattr(diffusion_model, 'double_blocks') and hasattr(diffusion_model, 'single_blocks'):
            print("Found Flux model structure, attaching DreamFit processors...")
            self._attach_dreamfit_to_flux(diffusion_model, checkpoint, garment_features, 
                                         strength, injection_strength, lora_rank, 
                                         double_blocks_idx, single_blocks_idx)
            return
        
        # Fallback to standard attn_processors approach
        if not hasattr(actual_model, 'attn_processors'):
            print("Warning: Model doesn't have attn_processors attribute or recognizable Flux structure")
            return
        
        # Validate garment_features
        if not isinstance(garment_features, dict):
            raise ValueError("garment_features must be a dictionary")
        
        required_keys = ["garment_token", "pooled_features", "patch_features"]
        for key in required_keys:
            if key not in garment_features:
                raise ValueError(f"Missing required key '{key}' in garment_features")
        
        dreamfit_processors = {}
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Get current processors
        current_processors = actual_model.attn_processors
        
        for name, processor in current_processors.items():
            # Extract layer index from name
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            else:
                layer_index = -1
            
            # Determine if this layer should have DreamFit processor
            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                print(f"Attaching DreamFit processor to {name}")
                dreamfit_proc = DreamFitDoubleStreamProcessor(
                    hidden_size=3072,
                    num_heads=24,
                    rank=lora_rank,
                    lora_weight=strength
                )
                
                # Load LoRA weights if available
                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k and "processor" in k:
                        key = k.replace(f"{name}.processor.", "")
                        lora_state_dict[key] = checkpoint[k]
                
                if lora_state_dict:
                    try:
                        dreamfit_proc.load_state_dict(lora_state_dict, strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load LoRA weights for {name}: {e}")
                
                dreamfit_proc = dreamfit_proc.to(device, dtype=dtype)
                dreamfit_processors[name] = dreamfit_proc
                
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                print(f"Attaching DreamFit processor to {name}")
                dreamfit_proc = DreamFitSingleStreamProcessor(
                    hidden_size=3072,
                    num_heads=24,
                    rank=lora_rank,
                    lora_weight=strength
                )
                
                # Load LoRA weights if available
                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k and "processor" in k:
                        key = k.replace(f"{name}.processor.", "")
                        lora_state_dict[key] = checkpoint[k]
                
                if lora_state_dict:
                    try:
                        dreamfit_proc.load_state_dict(lora_state_dict, strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load LoRA weights for {name}: {e}")
                
                dreamfit_proc = dreamfit_proc.to(device, dtype=dtype)
                dreamfit_processors[name] = dreamfit_proc
            else:
                # Keep original processor
                dreamfit_processors[name] = processor
        
        # Set the new processors
        actual_model.set_attn_processor(dreamfit_processors)
        
        # Store features in processors for access during sampling
        for proc in dreamfit_processors.values():
            if isinstance(proc, (DreamFitDoubleStreamProcessor, DreamFitSingleStreamProcessor)):
                proc.garment_features = garment_features
                proc.injection_strength = injection_strength
    
    def _attach_dreamfit_to_flux(
        self,
        diffusion_model,
        checkpoint: Dict,
        garment_features: Dict,
        strength: float,
        injection_strength: float,
        lora_rank: int,
        double_blocks_idx: List[int],
        single_blocks_idx: List[int]
    ):
        """Attach DreamFit processors directly to Flux model blocks"""
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Import the processor we'll use
        from ..dreamfit_core.models.dreamfit_attention import DreamFitDoubleStreamProcessor as DreamFitDoubleBlockProcessor
        
        # Process double blocks
        if hasattr(diffusion_model, 'double_blocks'):
            for idx, block in enumerate(diffusion_model.double_blocks):
                if idx in double_blocks_idx:
                    print(f"Attaching DreamFit processor to double_block_{idx}")
                    # Create and set processor
                    # Get block dimensions
                    hidden_size = 3072  # Default Flux hidden size
                    num_heads = 24      # Default Flux num_heads
                    
                    # Try to get actual dimensions from block
                    if hasattr(block, 'hidden_size'):
                        hidden_size = block.hidden_size
                    if hasattr(block, 'num_heads'):
                        num_heads = block.num_heads
                    
                    processor = DreamFitDoubleBlockProcessor(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        rank=lora_rank,
                        lora_weight=strength
                    )
                    
                    # Initialize from checkpoint if available
                    processor_key = f"double_blocks.{idx}"
                    if processor_key in checkpoint:
                        try:
                            processor.load_state_dict(checkpoint[processor_key])
                        except Exception as e:
                            print(f"Warning: Failed to load checkpoint for {processor_key}: {e}")
                    
                    processor.to(device, dtype)
                    
                    # Store garment features in the processor
                    processor.garment_features = garment_features
                    processor.injection_strength = injection_strength
                    
                    # Replace the forward method or set processor
                    if hasattr(block, 'set_processor'):
                        block.set_processor(processor)
                    else:
                        # Store original forward and replace
                        if not hasattr(block, '_original_forward'):
                            block._original_forward = block.forward
                        
                        # Create a wrapper that properly handles the processor
                        original_block = block
                        
                        def forward_with_processor(img, txt, vec, pe, **kwargs):
                            # Call processor with the block
                            return processor(original_block, img, txt, vec, pe, **kwargs)
                        
                        block.forward = forward_with_processor
                        block.processor = processor
                        block.current_mode = "normal"  # Add mode tracking
        
        # Store a flag to indicate processors are attached
        diffusion_model._dreamfit_processors_attached = True
        print(f"DreamFit processors attached to {len(double_blocks_idx)} double blocks")
    
    def _tensor_to_comfyui_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert processed tensor back to ComfyUI image format"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)
        
        # Convert from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        
        return tensor


# Node registration
NODE_CLASS_MAPPINGS = {
    "DreamFitUnified": DreamFitUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitUnified": "DreamFit Unified",
}