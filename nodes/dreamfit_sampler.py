"""
DreamFit Sampler Node for ComfyUI
Implements garment-centric human generation using DreamFit's two-pass sampling
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Add DreamFit source to Python path
DREAMFIT_PATH = os.path.join(os.path.dirname(__file__), "..", "DreamFit-official", "src")
if DREAMFIT_PATH not in sys.path:
    sys.path.insert(0, DREAMFIT_PATH)

# ComfyUI imports
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import node_helpers
from comfy import model_management
from comfy.ldm.flux.layers import timestep_embedding

# Import DreamFit components
try:
    from flux.modules.layers_dreamfit import (
        DoubleStreamBlockLoraProcessor, 
        SingleStreamBlockLoraProcessor
    )
    from flux.sampling import denoise, prepare, prepare_img, prepare_txt, get_schedule, unpack
    from flux.util import load_checkpoint, get_lora_rank
    print("Successfully imported DreamFit components")
except ImportError as e:
    print(f"Error importing DreamFit components: {e}")
    print(f"Make sure DreamFit-official is in the correct location: {DREAMFIT_PATH}")
    raise

import math
import torch.nn as nn


class ModulationWrapper(nn.Module):
    """Wrapper that makes ComfyUI's Modulation accept DreamFit's rw_mode parameter"""
    def __init__(self, modulation):
        super().__init__()
        self.modulation = modulation
    
    def forward(self, vec, rw_mode=None):
        # Ignore rw_mode, just call original modulation
        return self.modulation(vec)


class ProcessorWrapper:
    """Wrapper for DreamFit processors that handles compatibility with ComfyUI's blocks"""
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, block, *args, **kwargs):
        # Store original methods
        orig_img_mod = getattr(block, 'img_mod', None)
        orig_txt_mod = getattr(block, 'txt_mod', None)
        orig_modulation = getattr(block, 'modulation', None)
        
        # Temporarily wrap modulation methods
        if orig_img_mod is not None:
            block.img_mod = ModulationWrapper(orig_img_mod)
        
        if orig_txt_mod is not None:
            block.txt_mod = ModulationWrapper(orig_txt_mod)
        
        if orig_modulation is not None:
            block.modulation = ModulationWrapper(orig_modulation)
        
        try:
            # Call the processor with wrapped block
            result = self.processor(block, *args, **kwargs)
            return result
        finally:
            # Restore original methods
            if orig_img_mod is not None:
                block.img_mod = orig_img_mod
            if orig_txt_mod is not None:
                block.txt_mod = orig_txt_mod
            if orig_modulation is not None:
                block.modulation = orig_modulation


class FixedLoRALinearLayer(nn.Module):
    """LoRA layer that properly handles device placement"""
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()
        # Create layers in float32 for initialization, then convert to target dtype
        init_dtype = torch.float32 if dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else dtype
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=init_dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=init_dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        # Convert to target dtype if needed
        if dtype != init_dtype:
            self.down = self.down.to(dtype=dtype)
            self.up = self.up.to(dtype=dtype)
    
    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        
        return up_hidden_states.to(orig_dtype)


class FixedDoubleStreamBlockLoraProcessor(DoubleStreamBlockLoraProcessor):
    """Fixed version that properly handles device placement"""
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, device=None, dtype=None):
        # Don't call parent init, recreate everything with proper device
        super(DoubleStreamBlockLoraProcessor, self).__init__()  # Skip parent, go to ModelMixin
        
        # Create LoRA layers with device/dtype
        self.ref_qkv_lora_q = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_k = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_v = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_proj_lora1 = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        
        self.lora_weight = lora_weight
        
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None


class FixedSingleStreamBlockLoraProcessor(SingleStreamBlockLoraProcessor):
    """Fixed version that properly handles device placement"""
    def __init__(self, dim: int, rank: int = 4, network_alpha=None, lora_weight: float = 1, ip_scale=1.0, device=None, dtype=None):
        # Don't call parent init, recreate everything with proper device
        super(SingleStreamBlockLoraProcessor, self).__init__()  # Skip parent, go to ModelMixin
        
        # Create LoRA layers with device/dtype
        self.ref_qkv_lora_q = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_k = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_qkv_lora_v = FixedLoRALinearLayer(dim, dim, rank, network_alpha, device=device, dtype=dtype)
        self.ref_proj_lora = FixedLoRALinearLayer(15360, dim, rank*2, network_alpha*2, device=device, dtype=dtype)
        
        self.lora_weight = lora_weight
        self.ip_scale = ip_scale
        
        self.bank_img_q = None
        self.bank_img_k = None
        self.bank_img_v = None
        self.bank_neg_img_q = None
        self.bank_neg_img_k = None
        self.bank_neg_img_v = None


def forward_orig_dreamfit(
    self,
    img, img_ids, txt, txt_ids, timesteps, y, guidance, control, transformer_options, attn_mask=None
):
    """
    Custom forward_orig for DreamFit that supports rw_mode and processors.
    Based on PuLID-Flux approach but adapted for DreamFit's two-pass mechanism.
    """
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")
    
    # Extract rw_mode from transformer_options
    rw_mode = transformer_options.get('rw_mode', 'normal')
    
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    
    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)
    
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)
    
    # Process through double blocks
    for i, block in enumerate(self.double_blocks):
        # Check if we have a DreamFit processor for this block
        if hasattr(self, 'dreamfit_processors'):
            processor_name = f"double_blocks.{i}.processor"
            if processor_name in self.dreamfit_processors:
                # Use DreamFit processor instead of normal forward
                processor = self.dreamfit_processors[processor_name]
                img, txt = processor(block, img, txt, vec, pe, rw_mode=rw_mode)
            else:
                # Normal block forward
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        else:
            # Normal block forward
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        
        # Handle controlnet if present
        if control is not None:
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add
    
    img = torch.cat((txt, img), 1)
    
    # Process through single blocks
    for i, block in enumerate(self.single_blocks):
        # Check if we have a DreamFit processor for this block
        if hasattr(self, 'dreamfit_processors'):
            processor_name = f"single_blocks.{i}.processor"
            if processor_name in self.dreamfit_processors:
                # Use DreamFit processor instead of normal forward
                processor = self.dreamfit_processors[processor_name]
                # Extract real_img (without txt part)
                real_img, txt_part = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                # Call processor on real_img only
                real_img = processor(block, real_img, vec, pe, rw_mode=rw_mode)
                # Recombine
                img = torch.cat((txt_part, real_img), 1)
            else:
                # Normal block forward
                img = block(img, vec=vec, pe=pe)
        else:
            # Normal block forward
            img = block(img, vec=vec, pe=pe)
        
        # Handle controlnet if present
        if control is not None:
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add
    
    img = img[:, txt.shape[1]:, ...]
    
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


class FluxModelWrapper:
    """
    Wrapper to make ComfyUI's Flux model compatible with DreamFit's expected interface.
    
    DreamFit expects a model callable with (img, img_ids, txt, txt_ids, y, timesteps, guidance, rw_mode)
    ComfyUI's Flux model expects (x, timestep, context, y, guidance, ...)
    """
    
    def __init__(self, comfy_model, device, dtype):
        """
        Args:
            comfy_model: The ComfyUI model (could be ModelPatcher or the actual model)
            device: torch device
            dtype: torch dtype
        """
        # Extract the actual diffusion model
        if hasattr(comfy_model, 'model'):
            # It's a ModelPatcher
            self.model_patcher = comfy_model
            if hasattr(comfy_model.model, 'diffusion_model'):
                self.diffusion_model = comfy_model.model.diffusion_model
            else:
                self.diffusion_model = comfy_model.model
        else:
            # Direct model
            self.model_patcher = None
            self.diffusion_model = comfy_model
            
        self.device = device
        self.dtype = dtype
        
        # Storage for DreamFit's read/write mechanism
        self.stored_features = {}
        
    def __call__(self, img, img_ids, txt, txt_ids, y, timesteps, guidance, 
                 rw_mode="normal", ref_img_ids=None, **kwargs):
        """
        Translate DreamFit's calling convention to ComfyUI's Flux model interface.
        
        Args:
            img: Image latents (DreamFit format - already packed)
            img_ids: Image position embeddings
            txt: Text embeddings
            txt_ids: Text position embeddings
            y: CLIP embeddings (vec)
            timesteps: Timestep tensor
            guidance: Guidance strength
            rw_mode: One of "write", "neg_write", "read", "neg_read", "normal"
            ref_img_ids: Reference image IDs for read mode
        """
        # Debug output
        print(f"\nFluxModelWrapper called with:")
        print(f"  - img shape: {img.shape}")
        print(f"  - txt shape: {txt.shape}")
        print(f"  - timesteps: {timesteps}")
        print(f"  - rw_mode: {rw_mode}")
        print(f"  - diffusion_model type: {type(self.diffusion_model)}")
        print(f"  - has forward_orig: {hasattr(self.diffusion_model, 'forward_orig')}")
        
        # Create transformer_options with rw_mode
        transformer_options = kwargs.get('transformer_options', {})
        transformer_options['rw_mode'] = rw_mode
        
        # Call forward_orig with the packed format
        if hasattr(self.diffusion_model, 'forward_orig'):
            print(f"  - Using forward_orig method")
            return self.diffusion_model.forward_orig(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance,
                control=kwargs.get('control'),
                transformer_options=transformer_options,
                attn_mask=kwargs.get('attn_mask')
            )
        else:
            # This shouldn't happen with ComfyUI's Flux model
            raise RuntimeError("Model doesn't have forward_orig method")


class DreamFitSampler:
    """
    DreamFit sampler node that implements garment-centric generation
    Supports three modes: garment_generation, pose_control, virtual_tryon
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Core inputs
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "garment_image": ("IMAGE",),
                
                # Mode selection
                "mode": (["garment_generation", "pose_control", "virtual_tryon"], {
                    "default": "garment_generation"
                }),
                
                # Sampling parameters
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
                "pose_image": ("IMAGE",),
                "person_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "dreamfit"
    
    def sample(self, model, vae, positive: List, negative: List, latent_image: Dict,
               garment_image: torch.Tensor, mode: str, seed: int, steps: int, 
               cfg: float, sampler_name: str, scheduler: str, denoise_strength: float,
               lora_path: str = "", pose_image: Optional[torch.Tensor] = None,
               person_image: Optional[torch.Tensor] = None) -> Tuple[Dict]:
        """
        Perform DreamFit two-pass sampling
        
        Args:
            model: ComfyUI model object
            vae: ComfyUI VAE object for encoding images
            positive: Positive conditioning (list of tuples)
            negative: Negative conditioning (list of tuples)
            latent_image: Latent image dict with 'samples' key
            garment_image: Garment image tensor [B, H, W, C]
            mode: One of 'garment_generation', 'pose_control', 'virtual_tryon'
            seed: Random seed
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Name of sampler (unused, using DreamFit's sampler)
            scheduler: Name of scheduler (unused, using DreamFit's scheduler)
            denoise_strength: Denoising strength
            lora_path: Optional custom LoRA path
            pose_image: Optional pose image for pose_control mode
            person_image: Optional person image for virtual_tryon mode
            
        Returns:
            Tuple containing dict with 'samples' key
        """
        
        # Validate mode-specific inputs
        if mode == "pose_control" and pose_image is None:
            raise ValueError("pose_image is required for pose_control mode")
        if mode == "virtual_tryon" and person_image is None:
            raise ValueError("person_image is required for virtual_tryon mode")
        
        # Auto-select LoRA based on mode if not provided
        if not lora_path:
            # Try to find ComfyUI's model directory
            try:
                import folder_paths
                models_dir = os.path.join(folder_paths.models_dir, "dreamfit")
            except:
                # Fallback to relative path
                models_dir = os.path.join(os.path.dirname(__file__), "..", "pretrained_models")
            
            lora_map = {
                "garment_generation": os.path.join(models_dir, "flux_i2i.bin"),
                "pose_control": os.path.join(models_dir, "flux_i2i_with_pose.bin"),
                "virtual_tryon": os.path.join(models_dir, "flux_tryon.bin")
            }
            lora_path = lora_map[mode]
            print(f"Auto-selected LoRA for {mode}: {lora_path}")
        
        # Check if LoRA file exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        
        # Get device and dtype early
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        
        # Load LoRA checkpoint
        print(f"Loading DreamFit LoRA from: {lora_path}")
        # load_checkpoint expects (local_path, repo_id, name) - we only have local path
        checkpoint = load_checkpoint(lora_path, None, None)
        
        # Move all checkpoint tensors to the correct device
        checkpoint = {k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor) else v 
                     for k, v in checkpoint.items()}
        
        rank = get_lora_rank(checkpoint)
        print(f"LoRA rank: {rank}")
        
        # Get the actual model (handle ModelPatcher)
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        # Get the diffusion model for processor application
        diffusion_model = actual_model
        if hasattr(actual_model, 'diffusion_model'):
            diffusion_model = actual_model.diffusion_model
            print(f"Found diffusion_model: {type(diffusion_model)}")
        
        # Store original processors
        original_processors = getattr(diffusion_model, 'attn_processors', {}).copy() if hasattr(diffusion_model, 'attn_processors') else {}
        
        # Initialize wrapped_model outside try block
        wrapped_model = None
        
        try:
            # Apply DreamFit processors using PuLID-style patching
            print("Applying DreamFit processors...")
            
            # Create the model wrapper
            print("Creating FluxModelWrapper...")
            wrapped_model = FluxModelWrapper(model, device, dtype)
            
            # Check if we already patched the model
            if not hasattr(wrapped_model.diffusion_model, 'dreamfit_processors'):
                print("Patching model with DreamFit support...")
                # Add DreamFit data storage
                wrapped_model.diffusion_model.dreamfit_processors = {}
                wrapped_model.diffusion_model.dreamfit_rw_mode = 'normal'
                
                # Replace forward_orig with our custom version
                if hasattr(wrapped_model.diffusion_model, 'forward_orig'):
                    # Bind our custom forward_orig to the model
                    new_forward = forward_orig_dreamfit.__get__(wrapped_model.diffusion_model, wrapped_model.diffusion_model.__class__)
                    setattr(wrapped_model.diffusion_model, 'forward_orig', new_forward)
                    print("Patched forward_orig with DreamFit support")
            
            # Create and load processors
            lora_processors = self._create_and_load_processors(wrapped_model.diffusion_model, checkpoint, rank)
            
            # Wrap processors to handle ComfyUI compatibility
            wrapped_processors = {}
            for name, processor in lora_processors.items():
                wrapped_processors[name] = ProcessorWrapper(processor)
            
            wrapped_model.diffusion_model.dreamfit_processors = wrapped_processors
            print(f"Loaded {len(lora_processors)} DreamFit processors")
            
            # Load modulation LoRA weights
            self._load_modulation_lora(diffusion_model, checkpoint)
            
            # Prepare inputs for DreamFit
            print("Preparing inputs...")
            
            # Get latent samples
            latent_samples = latent_image["samples"]
            batch_size = latent_samples.shape[0]
            
            # Convert garment image to latent
            garment_latent = self._encode_image_to_latent(vae, garment_image)
            
            # Prepare negative garment (zeros)
            neg_garment_latent = torch.zeros_like(garment_latent)
            
            # Get text encoders from model
            clip_encoder, t5_encoder = self._get_text_encoders(model)
            
            # Prepare conditioning in DreamFit format
            print("Preparing DreamFit conditioning...")
            
            if clip_encoder is not None and t5_encoder is not None:
                # Use DreamFit's text encoders
                positive_prompt = self._extract_prompt_from_conditioning(positive)
                negative_prompt = self._extract_prompt_from_conditioning(negative)
                
                inp_person = prepare(t5_encoder, clip_encoder, latent_samples, positive_prompt)
                inp_cloth = prepare(t5_encoder, clip_encoder, garment_latent, ["cloth"] * batch_size)
                neg_inp_cond = prepare(t5_encoder, clip_encoder, neg_garment_latent, negative_prompt)
            else:
                # Fallback: Use ComfyUI's pre-encoded conditioning
                print("Using ComfyUI conditioning (text encoders not available)")
                inp_person = self._prepare_conditioning_from_comfy(positive, latent_samples, device, dtype)
                
                # For garment, we still need to create conditioning
                # Use empty text conditioning for "cloth"
                inp_cloth = self._prepare_conditioning_from_comfy(positive, garment_latent, device, dtype)
                # Override text with minimal embedding for "cloth"
                inp_cloth["txt"] = torch.zeros_like(inp_cloth["txt"][:, :1, :])  # Single token
                inp_cloth["txt_ids"] = torch.zeros_like(inp_cloth["txt_ids"][:, :1, :])
                
                # Negative conditioning
                neg_inp_cond = self._prepare_conditioning_from_comfy(negative, neg_garment_latent, device, dtype)
            
            # Move to correct device
            for key in inp_person:
                inp_person[key] = inp_person[key].to(device, dtype=dtype)
            for key in inp_cloth:
                inp_cloth[key] = inp_cloth[key].to(device, dtype=dtype)
            for key in neg_inp_cond:
                neg_inp_cond[key] = neg_inp_cond[key].to(device, dtype=dtype)
            
            # Get timesteps
            height, width = latent_samples.shape[2] * 8, latent_samples.shape[3] * 8
            timesteps = get_schedule(
                num_steps=steps,
                image_seq_len=(width // 8) * (height // 8) // 4,
                shift=True
            )
            
            # Set seed
            torch.manual_seed(seed)
            
            # Run DreamFit two-pass sampling
            print(f"Running DreamFit {mode} sampling for {steps} steps...")
            print(f"Model type: {type(actual_model)}")
            print(f"Is callable: {callable(actual_model)}")
            
            # Check if model has a forward method or diffusion_model attribute
            if hasattr(actual_model, 'diffusion_model'):
                print(f"Found diffusion_model: {type(actual_model.diffusion_model)}")
                if hasattr(actual_model.diffusion_model, 'forward'):
                    print("diffusion_model has forward method")
            
            # The issue is that ComfyUI's model expects different calling convention
            # DreamFit expects model(img=..., txt=..., etc)
            # But ComfyUI might expect model.forward(...) or a different interface
            
            # For now, let's try to understand what's available
            if hasattr(actual_model, 'forward'):
                print("Model has forward method")
            if hasattr(actual_model, '__call__'):
                print("Model has __call__ method")
            
            # Debug: Check if denoise is imported correctly
            print(f"denoise function type: {type(denoise)}")
            print(f"denoise function callable: {callable(denoise)}")
            print(f"denoise function location: {denoise.__module__}.{denoise.__name__}")
            
            # Debug: Also check the parameter that was shadowing
            print(f"denoise_strength parameter: {denoise_strength} (type: {type(denoise_strength)})")
            
            # Debug: Let's see what we're actually passing
            print(f"About to call denoise with:")
            print(f"  - model type: {type(actual_model)}")
            print(f"  - model callable: {callable(actual_model)}")
            print(f"  - model dir: {dir(actual_model)[:20]}...")  # First 20 attributes
            print(f"  - guidance (cfg): {cfg} (type: {type(cfg)})")
            print(f"  - true_gs: {3.5} (type: {type(3.5)})")
            print(f"  - timesteps length: {len(timesteps)}")
            print(f"  - timesteps sample: {timesteps[:5] if len(timesteps) > 5 else timesteps}")
            print(f"  - inp_person keys: {inp_person.keys()}")
            print(f"  - inp_cloth keys: {inp_cloth.keys()}")
            
            # Check if model has the expected Flux structure
            if hasattr(actual_model, 'double_blocks'):
                print(f"  - double_blocks: {len(actual_model.double_blocks)}")
            if hasattr(actual_model, 'single_blocks'):
                print(f"  - single_blocks: {len(actual_model.single_blocks)}")
                
            # Check what the actual diffusion_model looks like
            if hasattr(actual_model, 'diffusion_model'):
                dm = actual_model.diffusion_model
                print(f"\nDiffusion model details:")
                print(f"  - Type: {type(dm)}")
                print(f"  - Has double_blocks: {hasattr(dm, 'double_blocks')}")
                print(f"  - Has single_blocks: {hasattr(dm, 'single_blocks')}")
                if hasattr(dm, 'double_blocks'):
                    print(f"  - double_blocks count: {len(dm.double_blocks)}")
                if hasattr(dm, 'single_blocks'):
                    print(f"  - single_blocks count: {len(dm.single_blocks)}")
                    
                # Check the forward method signature
                if hasattr(dm, 'forward'):
                    import inspect
                    sig = inspect.signature(dm.forward)
                    print(f"  - forward signature: {sig}")
                    print(f"  - forward parameters: {list(sig.parameters.keys())}")
            
            # The wrapped_model is already created above
            
            # Add try-except to get more detailed error info
            try:
                with torch.no_grad():
                    samples = denoise(
                        model=wrapped_model,
                        inp_person=inp_person,
                        inp_cloth=inp_cloth,
                        neg_inp_cond=neg_inp_cond,
                        timesteps=timesteps,
                        guidance=cfg,
                        true_gs=3.5,  # From DreamFit config
                        timestep_to_start_cfg=51,  # From DreamFit config
                        num_steps=steps
                    )
            except TypeError as e:
                print(f"TypeError details: {e}")
                print(f"Error args: {e.args}")
                import traceback
                traceback.print_exc()
                raise
            
            print("Sampling complete!")
            
            # Return in ComfyUI format
            return ({"samples": samples},)
            
        finally:
            # Always restore original processors and forward_orig
            print("Restoring original model state...")
            if wrapped_model and hasattr(wrapped_model.diffusion_model, 'dreamfit_processors'):
                # Clear our processors
                wrapped_model.diffusion_model.dreamfit_processors = {}
            
            # Note: We don't restore forward_orig as other nodes might be using the patched version
            # This is similar to how PuLID handles it
    
    def _create_and_load_processors(self, model, checkpoint, rank):
        """Create DreamFit processors and load their weights"""
        processors = {}
        # Get device and dtype from model
        if hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
        else:
            device = torch.device('cpu')
            dtype = torch.float32
        
        print(f"Creating processors on device: {device}, dtype: {dtype}")
        
        # Process all double blocks (0-18)
        for i in range(19):
            name = f"double_blocks.{i}"
            processor = FixedDoubleStreamBlockLoraProcessor(
                dim=3072,
                rank=rank,
                network_alpha=16,
                lora_weight=1.0,
                device=device,
                dtype=dtype
            )
            
            
            # Load weights for this processor
            processor_state_dict = {}
            for key in checkpoint:
                if name in key and "processor" in key:
                    # Extract the part after processor
                    new_key = key.split(f"{name}.processor.")[-1]
                    # Checkpoint tensors are already on correct device
                    processor_state_dict[new_key] = checkpoint[key]
            
            if processor_state_dict:
                processor.load_state_dict(processor_state_dict, strict=False)
            
            # Force all parameters to correct device after loading
            # This is needed because load_state_dict might create new tensors
            for param in processor.parameters():
                param.data = param.data.to(device, dtype=dtype)
            
            processors[f"{name}.processor"] = processor
        
        # Process all single blocks (0-37)
        for i in range(38):
            name = f"single_blocks.{i}"
            processor = FixedSingleStreamBlockLoraProcessor(
                dim=3072,
                rank=rank,
                network_alpha=16,
                lora_weight=1.0,
                device=device,
                dtype=dtype
            )
            
            
            # Load weights
            processor_state_dict = {}
            for key in checkpoint:
                if name in key and "processor" in key:
                    new_key = key.split(f"{name}.processor.")[-1]
                    # Checkpoint tensors are already on correct device
                    processor_state_dict[new_key] = checkpoint[key]
            
            if processor_state_dict:
                processor.load_state_dict(processor_state_dict, strict=False)
            
            # Force all parameters to correct device after loading
            # This is needed because load_state_dict might create new tensors
            for param in processor.parameters():
                param.data = param.data.to(device, dtype=dtype)
            
            processors[f"{name}.processor"] = processor
        
        # Keep other processors unchanged
        if hasattr(model, 'attn_processors'):
            for name, proc in model.attn_processors.items():
                if name not in processors:
                    processors[name] = proc
        
        return processors
    
    def _load_modulation_lora(self, model, checkpoint):
        """Load LoRA weights for modulation layers"""
        # Get device and dtype from model
        if hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype
        else:
            device = torch.device('cpu')
            dtype = torch.float32
            
        # Extract modulation LoRA weights
        modulation_state_dict = {}
        for name, param in checkpoint.items():
            if 'lin_lora' in name:
                modulation_state_dict[name] = param.to(device, dtype=dtype)
        
        if modulation_state_dict:
            # Load into model
            missing, unexpected = model.load_state_dict(modulation_state_dict, strict=False)
            if missing:
                print(f"Missing modulation keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected modulation keys: {len(unexpected)}")
            
            # Force all model parameters to correct device after loading modulation
            # This ensures any newly created tensors are on the right device
            for param in model.parameters():
                param.data = param.data.to(device, dtype=dtype)
    
    def _encode_image_to_latent(self, vae, image):
        """Encode image to latent using VAE"""
        # Image is [B, H, W, C] in range [0, 1]
        # Ensure we only use RGB channels (first 3)
        image_rgb = image[:, :, :, :3]
        
        # ComfyUI's VAE.encode expects the image in [0, 1] range
        # and handles the conversion internally
        latent = vae.encode(image_rgb)
        
        return latent
    
    def _get_text_encoders(self, model):
        """Extract CLIP and T5 encoders from model"""
        # In ComfyUI, the text encoders are typically loaded separately
        # For DreamFit, we need to load them from the official implementation
        
        try:
            from flux.util import load_clip, load_t5
            device = model_management.get_torch_device()
            
            print("Loading text encoders...")
            clip_encoder = load_clip(device)
            t5_encoder = load_t5(device, max_length=512)
            
            return clip_encoder, t5_encoder
            
        except Exception as e:
            print(f"Error loading text encoders: {e}")
            # Fallback: Create wrapper that uses ComfyUI's conditioning directly
            return None, None
    
    def _extract_prompt_from_conditioning(self, conditioning):
        """Extract text prompt from ComfyUI conditioning format"""
        # In ComfyUI, we typically don't have access to the original prompt
        # The conditioning already contains the encoded text
        # For DreamFit, we need the raw text to re-encode with DreamFit's encoders
        
        # This is a limitation - we'll need to either:
        # 1. Add prompt as explicit input to the node
        # 2. Use the pre-encoded conditioning directly
        
        # For now, return empty prompts and rely on the conditioning tensors
        if isinstance(conditioning, list) and len(conditioning) > 0:
            batch_size = conditioning[0][0].shape[0]
            return [""] * batch_size
        return [""]
    
    def _prepare_conditioning_from_comfy(self, conditioning, latent, device, dtype):
        """Convert ComfyUI conditioning to DreamFit format"""
        # ComfyUI conditioning is [(tensor, dict), ...]
        # DreamFit expects dict with 'txt', 'txt_ids', 'vec', 'img', 'img_ids'
        
        if not conditioning or len(conditioning) == 0:
            raise ValueError("Empty conditioning")
        
        cond_tensor, cond_dict = conditioning[0]
        batch_size = latent.shape[0]
        
        # Prepare image embeddings from latent
        # DreamFit expects packed format: rearrange "b c (h ph) (w pw) -> b (h w) (c ph pw)"
        c, h, w = latent.shape[1], latent.shape[2], latent.shape[3]
        # Pack with patch size 2
        img = latent.view(batch_size, c, h // 2, 2, w // 2, 2)
        img = img.permute(0, 2, 4, 1, 3, 5)
        img = img.reshape(batch_size, (h // 2) * (w // 2), c * 4)
        
        # Create image position embeddings for packed dimensions
        h_packed, w_packed = h // 2, w // 2
        img_ids = torch.zeros(h_packed, w_packed, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h_packed, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_packed, device=device)[None, :]
        img_ids = img_ids.reshape(1, -1, 3).repeat(batch_size, 1, 1)
        
        # For text, we'll use the conditioning tensor
        # Flux expects separate CLIP and T5 embeddings
        # We'll approximate by using the conditioning as T5 output
        txt = cond_tensor  # This is the text conditioning from ComfyUI
        txt_ids = torch.zeros(batch_size, txt.shape[1], 3, device=device, dtype=dtype)
        
        # CLIP pooled output
        # Check if ComfyUI provided pooled output in the conditioning dict
        if 'pooled_output' in cond_dict:
            vec = cond_dict['pooled_output']
        else:
            # Fallback: pool over sequence dimension
            vec = torch.mean(txt, dim=1)
        
        return {
            "img": img.to(device, dtype=dtype),
            "img_ids": img_ids,
            "txt": txt.to(device, dtype=dtype),
            "txt_ids": txt_ids,
            "vec": vec.to(device, dtype=dtype),
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "DreamFitSampler": DreamFitSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamFitSampler": "DreamFit Sampler"
}