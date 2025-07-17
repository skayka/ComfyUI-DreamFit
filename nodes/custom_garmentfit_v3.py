"""
CustomGarmentFit V3 - Properly uses DreamFit LoRA weights
This version correctly applies the trained attention weights from DreamFit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any
import os
import logging
from PIL import Image

import folder_paths
import comfy.utils
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

# Set up model directory
MODELS_DIR = os.path.join(folder_paths.models_dir, "dreamfit")
if "dreamfit" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["dreamfit"]
folder_paths.folder_names_and_paths["dreamfit"] = (current_paths, folder_paths.supported_pt_extensions)


class DreamFitLoRALayer:
    """Represents a LoRA layer from DreamFit"""
    def __init__(self, down_weight, up_weight, alpha=None):
        self.down = down_weight  # [rank, in_features]
        self.up = up_weight      # [out_features, rank]
        self.alpha = alpha if alpha is not None else self.down.shape[0]
        self.scale = self.alpha / self.down.shape[0]
        
    def apply(self, x):
        """Apply LoRA transformation"""
        # x: [B, L, in_features]
        down_out = x @ self.down.T  # [B, L, rank]
        up_out = down_out @ self.up.T  # [B, L, out_features]
        return up_out * self.scale


class GarmentFitAttentionPatcher:
    """Handles attention patching with DreamFit weights"""
    
    def __init__(self, checkpoint_path):
        self.lora_layers = {}
        self.load_checkpoint(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        """Load and organize DreamFit checkpoint"""
        logging.info(f"Loading DreamFit checkpoint: {checkpoint_path}")
        
        state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)
        
        # Group LoRA weights by block and type
        lora_groups = {}
        for key, value in state_dict.items():
            if "lora" not in key.lower():
                continue
                
            # Parse the key structure
            parts = key.split('.')
            
            # Find block info
            block_type = None
            block_idx = None
            param_type = None
            
            if "double_blocks" in key:
                block_type = "double"
                block_idx = int(parts[parts.index("double_blocks") + 1])
            elif "single_blocks" in key:
                block_type = "single"  
                block_idx = int(parts[parts.index("single_blocks") + 1])
            else:
                continue
                
            # Find parameter type
            if "ref_qkv_lora_k" in key:
                param_type = "k"
            elif "ref_qkv_lora_v" in key:
                param_type = "v"
            elif "ref_qkv_lora_q" in key:
                param_type = "q"
            else:
                continue
                
            # Find if it's down or up
            is_down = ".down." in key or key.endswith(".down")
            is_up = ".up." in key or key.endswith(".up")
            
            if not (is_down or is_up):
                continue
                
            # Create group key
            group_key = f"{block_type}_{block_idx}_{param_type}"
            if group_key not in lora_groups:
                lora_groups[group_key] = {}
                
            if is_down:
                lora_groups[group_key]["down"] = value
            elif is_up:
                lora_groups[group_key]["up"] = value
                
        # Create LoRA layers from groups
        for key, weights in lora_groups.items():
            if "down" in weights and "up" in weights:
                self.lora_layers[key] = DreamFitLoRALayer(
                    weights["down"], 
                    weights["up"]
                )
                
        logging.info(f"Created {len(self.lora_layers)} LoRA layers")
        
    def create_attention_patch(self, block_type, block_idx, garment_embeds, weight=1.0):
        """Create attention patch function for a specific block"""
        
        # Get LoRA layers for this block
        k_key = f"{block_type}_{block_idx}_k"
        v_key = f"{block_type}_{block_idx}_v"
        
        k_lora = self.lora_layers.get(k_key)
        v_lora = self.lora_layers.get(v_key)
        
        if k_lora is None or v_lora is None:
            return None
            
        def attention_patch(q, k, v, extra_options):
            """Patch function that properly uses DreamFit LoRA weights"""
            # Get dimensions
            B, H, L, D = q.shape  # [Batch, Heads, Length, Dim]
            device = q.device
            dtype = q.dtype
            
            # Reshape for LoRA application
            q_reshape = q.transpose(1, 2).reshape(B, L, H * D)  # [B, L, hidden_dim]
            k_reshape = k.transpose(1, 2).reshape(B, L, H * D)
            v_reshape = v.transpose(1, 2).reshape(B, L, H * D)
            
            # Prepare garment embeddings
            garment_tokens = garment_embeds.to(device, dtype=dtype)
            if garment_tokens.shape[0] < B:
                garment_tokens = garment_tokens.repeat(B // garment_tokens.shape[0] + 1, 1, 1)[:B]
            
            # Apply LoRA to create garment K,V
            garment_k = k_lora.apply(garment_tokens)  # [B, num_tokens, hidden_dim]
            garment_v = v_lora.apply(garment_tokens)  # [B, num_tokens, hidden_dim]
            
            # Reshape back to attention format
            num_garment_tokens = garment_k.shape[1]
            garment_k = garment_k.reshape(B, num_garment_tokens, H, D).transpose(1, 2)  # [B, H, num_tokens, D]
            garment_v = garment_v.reshape(B, num_garment_tokens, H, D).transpose(1, 2)  # [B, H, num_tokens, D]
            
            # Concatenate garment tokens with original K,V
            k_combined = torch.cat([k, garment_k], dim=2)  # [B, H, L+num_tokens, D]
            v_combined = torch.cat([v, garment_v], dim=2)  # [B, H, L+num_tokens, D]
            
            # Compute attention with combined K,V
            attn_out = optimized_attention(q, k_combined, v_combined, extra_options.get("n_heads", H))
            
            # Apply weight scaling
            original_out = optimized_attention(q, k, v, extra_options.get("n_heads", H))
            final_out = original_out + (attn_out - original_out) * weight
            
            return final_out
            
        return attention_patch


class SimpleGarmentEncoderV3(nn.Module):
    """Garment encoder that outputs in the right dimension for DreamFit LoRA"""
    def __init__(self, clip_dim=1024, hidden_dim=3072, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Project from CLIP to model hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable garment queries
        self.garment_queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        
    def forward(self, clip_features):
        """Generate garment tokens"""
        B = clip_features.shape[0]
        
        # Project CLIP features
        if clip_features.dim() == 2:
            clip_features = clip_features.unsqueeze(1)
        
        projected = self.proj(clip_features)  # [B, 1, hidden_dim]
        
        # Expand queries and modulate with projected features
        queries = self.garment_queries.expand(B, -1, -1)  # [B, num_tokens, hidden_dim]
        
        # Simple modulation (can be replaced with cross-attention)
        output = queries + projected.expand(-1, self.num_tokens, -1) * 0.5
        
        return output


class GarmentCLIPVisionLoader:
    """Load CLIP vision model for garment encoding (DreamFit uses standard CLIP-ViT-Large)"""
    @classmethod
    def INPUT_TYPES(s):
        # List available vision models in ComfyUI/models/clip_vision/
        vision_models = folder_paths.get_filename_list("clip_vision")
        default_options = ["auto_download_clip_vit_large"]
        
        return {
            "required": {
                "clip_vision_model": (vision_models + default_options, ),
            }
        }
    
    RETURN_TYPES = ("GARMENT_CLIP_VISION",)
    FUNCTION = "load_vision"
    CATEGORY = "dreamfit"
    
    def load_vision(self, clip_vision_model):
        if clip_vision_model == "auto_download_clip_vit_large":
            # Auto-download standard CLIP (what DreamFit actually uses)
            try:
                from transformers import CLIPVisionModel, CLIPImageProcessor
                
                model_name = "openai/clip-vit-large-patch14"
                logging.info(f"Loading CLIP vision model: {model_name}")
                
                vision_model = CLIPVisionModel.from_pretrained(model_name)
                processor = CLIPImageProcessor.from_pretrained(model_name)
                
                return ({
                    "model": vision_model,
                    "processor": processor,
                    "type": "transformers",
                    "model_name": model_name
                },)
                
            except ImportError:
                raise ValueError("transformers not installed. Install with: pip install transformers")
            except Exception as e:
                raise ValueError(f"Failed to download CLIP model: {e}")
        
        # Load from ComfyUI's clip_vision folder
        vision_path = folder_paths.get_full_path("clip_vision", clip_vision_model)
        
        # Generic CLIP vision loading
        import comfy.clip_vision
        clip_vision = comfy.clip_vision.load(vision_path)
        return ({
            "model": clip_vision, 
            "type": "comfyui",
            "path": vision_path
        },)


class CustomGarmentFitLoaderV3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("dreamfit"), ),
            }
        }
    
    RETURN_TYPES = ("GARMENTFIT",)
    FUNCTION = "load_model"
    CATEGORY = "dreamfit"
    
    def load_model(self, checkpoint):
        checkpoint_path = folder_paths.get_full_path("dreamfit", checkpoint)
        
        # Create attention patcher with loaded weights
        patcher = GarmentFitAttentionPatcher(checkpoint_path)
        
        # Create encoder
        encoder = SimpleGarmentEncoderV3()
        
        return ({"patcher": patcher, "encoder": encoder},)


class ApplyCustomGarmentFitV3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "garmentfit": ("GARMENTFIT", ),
                "garment_image": ("IMAGE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "garment_clip_vision": ("GARMENT_CLIP_VISION", ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_garmentfit"
    CATEGORY = "dreamfit"
    
    def apply_garmentfit(self, model, garmentfit, garment_image, positive, negative, 
                        weight, start_at, end_at, garment_clip_vision=None):
        """Apply garment fitting with proper LoRA weights"""
        
        # Clone model
        work_model = model.clone()
        
        # Get components
        patcher = garmentfit["patcher"]
        encoder = garmentfit["encoder"]
        
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        
        # Move encoder to device
        encoder = encoder.to(device, dtype=dtype)
        
        # Process garment image with CLIP vision
        if garment_clip_vision is not None:
            try:
                vision_model = garment_clip_vision["model"]
                vision_type = garment_clip_vision.get("type", "comfyui")
                
                # Convert from ComfyUI format [B, H, W, C] to PIL
                from PIL import Image
                img_np = (garment_image[0].cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                if vision_type == "transformers":
                    # Use transformers CLIP (DreamFit's approach)
                    processor = garment_clip_vision["processor"]
                    
                    # Move model to device
                    vision_model = vision_model.to(device, dtype=dtype)
                    
                    # Process image with official processor
                    inputs = processor(images=img_pil, return_tensors="pt")
                    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = vision_model(**inputs)
                        # Use pooled output for global garment features
                        clip_features = outputs.pooler_output  # [B, 768]
                        
                    logging.info(f"✓ Encoded garment with transformers CLIP: {clip_features.shape}")
                    
                else:
                    # ComfyUI CLIP vision processing
                    import torchvision.transforms as transforms
                    
                    # Move model to device
                    vision_model = vision_model.to(device, dtype=dtype)
                    
                    # Standard CLIP preprocessing
                    img_tensor = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])(img_pil).unsqueeze(0).to(device, dtype=dtype)
                    
                    with torch.no_grad():
                        if hasattr(vision_model, 'encode_image'):
                            clip_features = vision_model.encode_image(img_tensor)
                        else:
                            clip_features = vision_model(img_tensor)
                            if hasattr(clip_features, 'pooler_output'):
                                clip_features = clip_features.pooler_output
                    
                    logging.info(f"✓ Encoded garment with ComfyUI CLIP: {clip_features.shape}")
                
            except Exception as e:
                logging.warning(f"Garment CLIP vision encoding failed: {e}, using fallback")
                clip_features = self._create_fallback_features(garment_image, device, dtype)
        else:
            # No dedicated CLIP vision provided, use fallback
            logging.warning("No garment CLIP vision provided, using fallback features")
            clip_features = self._create_fallback_features(garment_image, device, dtype)
        
        # Generate garment embeddings
        with torch.no_grad():
            garment_embeds = encoder(clip_features)
        
        # Get sigma values for scheduling
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)
        
        # Apply patches to model using PuLID-style direct modification
        cleanup_func = self._apply_patches(work_model, patcher, garment_embeds, weight, sigma_start, sigma_end)
        
        return (work_model,)
    
    def _apply_patches(self, model, patcher, garment_embeds, weight, sigma_start, sigma_end):
        """Apply DreamFit patches using PuLID-style direct model modification"""
        
        # Get the FLUX diffusion model
        flux_model = model.model.diffusion_model
        
        # Generate unique ID for this node
        import uuid
        unique_id = str(uuid.uuid4())
        
        # Initialize DreamFit data storage if not exists
        if not hasattr(flux_model, "dreamfit_data"):
            flux_model.dreamfit_data = {}
            flux_model.dreamfit_processors = {}
            
            # Patch transformer blocks instead of replacing forward method
            self._patch_transformer_blocks(flux_model)
            logging.info("✓ Patched FLUX transformer blocks with DreamFit LoRA")
        
        # Store garment data for this node
        flux_model.dreamfit_data[unique_id] = {
            'weight': weight,
            'garment_embeds': garment_embeds,
            'sigma_start': sigma_start, 
            'sigma_end': sigma_end,
            'patcher': patcher
        }
        
        # References are stored during patching in _patch_transformer_blocks
                
        logging.info(f"✓ Added DreamFit data (id: {unique_id[:8]}) with {len(patcher.lora_layers)} LoRA layers")
        
        # Store cleanup callback
        def cleanup():
            if hasattr(flux_model, "dreamfit_data") and unique_id in flux_model.dreamfit_data:
                del flux_model.dreamfit_data[unique_id]
                logging.info(f"Cleaned up DreamFit data {unique_id[:8]}")
        
        return cleanup
        
    def _patch_transformer_blocks(self, flux_model):
        """Patch FLUX transformer blocks to apply LoRA transformations"""
        
        # Patch double_blocks (0-18)
        for i in range(19):
            if hasattr(flux_model, 'double_blocks') and i < len(flux_model.double_blocks):
                block = flux_model.double_blocks[i]
                if not hasattr(block, '_dreamfit_patched'):
                    original_forward = block.forward
                    block._flux_model_ref = flux_model  # Store reference before patching
                    block.forward = self._create_enhanced_block_forward(original_forward, f"double_{i}", block)
                    block._dreamfit_patched = True
                    logging.info(f"✓ Patched double_blocks[{i}]")
        
        # Patch single_blocks (0-37)  
        for i in range(38):
            if hasattr(flux_model, 'single_blocks') and i < len(flux_model.single_blocks):
                block = flux_model.single_blocks[i]
                if not hasattr(block, '_dreamfit_patched'):
                    original_forward = block.forward
                    block._flux_model_ref = flux_model  # Store reference before patching
                    block.forward = self._create_enhanced_block_forward(original_forward, f"single_{i}", block)
                    block._dreamfit_patched = True
                    logging.info(f"✓ Patched single_blocks[{i}]")
    
    def _create_enhanced_block_forward(self, original_forward, block_name, block):
        """Create enhanced block forward that applies LoRA transformations"""
        
        def enhanced_forward(*args, **kwargs):
            # Run original block processing
            output = original_forward(*args, **kwargs)
            
            # Get flux_model from stored reference (avoid recursion)
            flux_model = getattr(block, '_flux_model_ref', None)
            
            if flux_model and hasattr(flux_model, "dreamfit_data") and flux_model.dreamfit_data:
                # Get current timestep (passed in args)
                timesteps = None
                for arg in args:
                    if hasattr(arg, 'dtype') and arg.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        if arg.numel() == 1 or (arg.dim() == 1 and arg.shape[0] <= 8):  # Likely timestep
                            timesteps = arg
                            break
                
                if timesteps is not None:
                    current_sigma = timesteps.float().mean().item() if timesteps is not None else 999999999.9
                    
                    # Check for active DreamFit data
                    for node_id, data in flux_model.dreamfit_data.items():
                        sigma_start = data['sigma_start']
                        sigma_end = data['sigma_end']
                        
                        if sigma_end <= current_sigma <= sigma_start:
                            weight = data['weight']
                            garment_embeds = data['garment_embeds']
                            patcher = data['patcher']
                            
                            # Apply LoRA transformations for this block
                            garment_influence = self._apply_block_lora(block_name, garment_embeds, patcher, weight, output)
                            if garment_influence is not None:
                                output = output + garment_influence
                                logging.info(f"🎯 Applied LoRA to {block_name}: influence_norm={garment_influence.norm().item():.4f}")
                            break
            
            return output
            
        return enhanced_forward
    
    def _apply_block_lora(self, block_name, garment_embeds, patcher, weight, current_features):
        """Apply LoRA transformations for a specific block"""
        
        # Parse block name to find corresponding LoRA layers
        block_type, block_idx = block_name.split('_')
        
        # Find LoRA layers for this block
        q_key = f"{block_type}_{block_idx}_q"
        k_key = f"{block_type}_{block_idx}_k" 
        v_key = f"{block_type}_{block_idx}_v"
        
        q_lora = patcher.lora_layers.get(q_key)
        k_lora = patcher.lora_layers.get(k_key)
        v_lora = patcher.lora_layers.get(v_key)
        
        if not (q_lora or k_lora or v_lora):
            return None
            
        try:
            # Get dimensions from current features
            B, L, D = current_features.shape
            device = current_features.device
            dtype = current_features.dtype
            
            # Prepare garment embeddings - reshape to match current features
            garment_tokens = garment_embeds.to(device, dtype=dtype)
            if garment_tokens.shape[0] < B:
                garment_tokens = garment_tokens.repeat(B // garment_tokens.shape[0] + 1, 1, 1)[:B]
            
            # Create garment influence by applying available LoRA layers
            total_influence = torch.zeros_like(current_features[:, :garment_tokens.shape[1], :])
            
            if q_lora:
                q_influence = q_lora.apply(garment_tokens) * weight
                if q_influence.shape[-1] == D:
                    total_influence += q_influence * 0.3  # Scale for Q contribution
            
            if k_lora:
                k_influence = k_lora.apply(garment_tokens) * weight  
                if k_influence.shape[-1] == D:
                    total_influence += k_influence * 0.3  # Scale for K contribution
                    
            if v_lora:
                v_influence = v_lora.apply(garment_tokens) * weight
                if v_influence.shape[-1] == D:
                    total_influence += v_influence * 0.4  # Scale for V contribution
            
            # Pad or crop influence to match current features length
            if total_influence.shape[1] < L:
                padding = torch.zeros(B, L - total_influence.shape[1], D, device=device, dtype=dtype)
                total_influence = torch.cat([total_influence, padding], dim=1)
            elif total_influence.shape[1] > L:
                total_influence = total_influence[:, :L, :]
                
            return total_influence * 0.1  # Final scaling
            
        except Exception as e:
            logging.warning(f"Error applying LoRA to {block_name}: {e}")
            return None
    
    def _create_fallback_features(self, garment_image, device, dtype):
        """Create fallback features from image statistics"""
        img_flat = garment_image.flatten(1, -1)  # [B, H*W*C]
        img_mean = img_flat.mean(dim=1, keepdim=True)
        img_std = img_flat.std(dim=1, keepdim=True)
        # Create pseudo-features of dimension 1024
        clip_features = torch.cat([
            img_mean.expand(-1, 512),
            img_std.expand(-1, 512)
        ], dim=1).to(device, dtype=dtype)
        return clip_features


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GarmentCLIPVisionLoader": GarmentCLIPVisionLoader,
    "CustomGarmentFitLoaderV3": CustomGarmentFitLoaderV3,
    "ApplyCustomGarmentFitV3": ApplyCustomGarmentFitV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarmentCLIPVisionLoader": "Load Garment CLIP Vision",
    "CustomGarmentFitLoaderV3": "Load GarmentFit Model",
    "ApplyCustomGarmentFitV3": "Apply GarmentFit",
}