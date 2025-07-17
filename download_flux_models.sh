#!/bin/bash
# Simple bash script to download FLUX.1-dev models for ComfyUI
#
# Usage:
#   export HF_TOKEN="your_hugging_face_token"
#   ./download_flux_models.sh

# Set ComfyUI path (from custom_nodes/ComfyUI-DreamFit)
COMFYUI_PATH="../.."

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "FLUX.1-dev is a gated model and requires authentication."
    echo ""
    echo "Please set your Hugging Face token:"
    echo "  export HF_TOKEN=\"your_token_here\""
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Create directories
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/text_encoders"

echo "Downloading FLUX.1-dev models..."

# Download main model (fp16 version)
wget --header="Authorization: Bearer $HF_TOKEN" \
     -c "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
     -O "$COMFYUI_PATH/models/diffusion_models/flux1-dev.safetensors"

# Download VAE
wget --header="Authorization: Bearer $HF_TOKEN" \
     -c "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
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