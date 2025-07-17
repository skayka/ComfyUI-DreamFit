#!/usr/bin/env python3
"""
Download FLUX.1-dev model files for ComfyUI
Downloads the model, VAE, and text encoders and places them in the correct ComfyUI directories

Usage:
    # Set your HF token first:
    export HF_TOKEN="your_hugging_face_token"
    python download_flux_models.py
    
    # Or pass it directly:
    python download_flux_models.py --hf-token "your_token"

Note: If you already have a FLUX model (e.g., fp8 version), the script will
      download the fp16 version as flux1-dev_fp16.safetensors
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

def download_file(url, dest_path, desc=None, headers=None, force_suffix=None):
    """Download a file with progress bar"""
    # Check if file already exists
    if force_suffix and os.path.exists(dest_path):
        # Add suffix to filename if original exists
        base, ext = os.path.splitext(dest_path)
        dest_path = f"{base}_{force_suffix}{ext}"
    
    if os.path.exists(dest_path):
        print(f"✓ {desc or dest_path} already exists, skipping...")
        return dest_path
    
    print(f"Downloading {desc or url}...")
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(block_size):
                pbar.update(len(chunk))
                f.write(chunk)
    
    print(f"✓ Downloaded {desc or dest_path}")
    return dest_path

def main():
    parser = argparse.ArgumentParser(description='Download FLUX.1-dev models for ComfyUI')
    parser.add_argument('--comfyui-path', type=str, default='../..',
                        help='Path to ComfyUI installation (default: ../.. for custom_nodes/ComfyUI-DreamFit)')
    parser.add_argument('--hf-token', type=str, default=os.environ.get('HF_TOKEN'),
                        help='Hugging Face token for gated models (or set HF_TOKEN env var)')
    args = parser.parse_args()
    
    # Base paths
    comfyui_path = Path(args.comfyui_path).resolve()
    if not comfyui_path.exists():
        print(f"Error: ComfyUI path not found: {comfyui_path}")
        sys.exit(1)
    
    models_path = comfyui_path / "models"
    
    # Create directories if they don't exist
    diffusion_models_path = models_path / "diffusion_models"
    vae_path = models_path / "vae"
    clip_path = models_path / "text_encoders"
    
    for path in [diffusion_models_path, vae_path, clip_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"ComfyUI path: {comfyui_path}")
    print(f"Models will be downloaded to: {models_path}")
    print()
    
    # HuggingFace base URL
    base_url = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main"
    
    # Headers for authentication
    headers = {"Authorization": f"Bearer {args.hf_token}"} if args.hf_token else None
    
    if not args.hf_token:
        print("WARNING: No Hugging Face token provided. FLUX.1-dev is a gated model and requires authentication.")
        print("Please set HF_TOKEN environment variable or use --hf-token argument.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Check if fp8 model already exists
    existing_model = diffusion_models_path / "flux1-dev.safetensors"
    force_suffix = None
    if existing_model.exists():
        print(f"\nNote: Found existing model at {existing_model}")
        print("Will download fp16 version as flux1-dev_fp16.safetensors")
        force_suffix = "fp16"
    
    # Files to download
    downloads = [
        # Main model - using fp16 version
        {
            "url": f"{base_url}/flux1-dev.safetensors",
            "dest": diffusion_models_path / "flux1-dev.safetensors",
            "desc": "FLUX.1-dev model (fp16)",
            "force_suffix": force_suffix
        },
        # VAE
        {
            "url": f"{base_url}/ae.safetensors",
            "dest": vae_path / "ae.safetensors",
            "desc": "FLUX VAE"
        },
    ]
    
    # Also need CLIP and T5 encoders from the text encoder repo
    text_encoder_base = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main"
    
    downloads.extend([
        # CLIP-L encoder
        {
            "url": f"{text_encoder_base}/clip_l.safetensors",
            "dest": clip_path / "clip_l.safetensors",
            "desc": "CLIP-L text encoder"
        },
        # T5-XXL encoder (fp16 version to match model)
        {
            "url": f"{text_encoder_base}/t5xxl_fp16.safetensors",
            "dest": clip_path / "t5xxl_fp16.safetensors",
            "desc": "T5-XXL text encoder (fp16)"
        },
    ])
    
    print(f"Will download {len(downloads)} files...\n")
    
    # Download all files
    downloaded_files = {}
    for item in downloads:
        try:
            # Use headers for FLUX.1-dev downloads (gated model)
            item_headers = headers if "black-forest-labs" in item["url"] else None
            actual_path = download_file(
                item["url"], 
                item["dest"], 
                item["desc"], 
                headers=item_headers,
                force_suffix=item.get("force_suffix")
            )
            if actual_path:
                downloaded_files[item["desc"]] = actual_path
        except Exception as e:
            print(f"Error downloading {item['desc']}: {e}")
            continue
    
    print("\n✓ All downloads complete!")
    print("\nModel locations:")
    
    # Show actual downloaded paths
    if force_suffix:
        print(f"  Diffusion model (fp16): {diffusion_models_path / 'flux1-dev_fp16.safetensors'}")
        print(f"  Note: You already have a model at {diffusion_models_path / 'flux1-dev.safetensors'}")
    else:
        print(f"  Diffusion model: {diffusion_models_path / 'flux1-dev.safetensors'}")
    
    print(f"  VAE: {vae_path / 'ae.safetensors'}")
    print(f"  CLIP encoder: {clip_path / 'clip_l.safetensors'}")
    print(f"  T5 encoder: {clip_path / 't5xxl_fp16.safetensors'}")
    
    print("\nYou can now use these models in ComfyUI with the 'Load Diffusion Model' node!")
    
    if force_suffix:
        print("\nTo use the fp16 model instead of fp8, select 'flux1-dev_fp16.safetensors' in the Load Diffusion Model node.")

if __name__ == "__main__":
    main()