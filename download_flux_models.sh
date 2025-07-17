#!/bin/bash
# Simple bash script to download FLUX.1-dev models for ComfyUI

# Set ComfyUI path (from custom_nodes/ComfyUI-DreamFit)
COMFYUI_PATH="../.."

# Create directories
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/text_encoders"

echo "Downloading FLUX.1-dev models..."

# Download main model (fp16 version)
wget -c "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
     -O "$COMFYUI_PATH/models/diffusion_models/flux1-dev.safetensors"

# Download VAE
wget -c "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
     -O "$COMFYUI_PATH/models/vae/ae.safetensors"

# Download text encoders
wget -c "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
     -O "$COMFYUI_PATH/models/text_encoders/clip_l.safetensors"

wget -c "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
     -O "$COMFYUI_PATH/models/text_encoders/t5xxl_fp16.safetensors"

echo "✓ Download complete!"
echo ""
echo "Model locations:"
echo "  Diffusion model: $COMFYUI_PATH/models/diffusion_models/flux1-dev.safetensors"
echo "  VAE: $COMFYUI_PATH/models/vae/ae.safetensors"
echo "  CLIP encoder: $COMFYUI_PATH/models/text_encoders/clip_l.safetensors"
echo "  T5 encoder: $COMFYUI_PATH/models/text_encoders/t5xxl_fp16.safetensors"