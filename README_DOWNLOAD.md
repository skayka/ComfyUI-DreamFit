# Downloading FLUX.1-dev Models

FLUX.1-dev is a gated model that requires authentication. You'll need a Hugging Face token.

## Getting your Hugging Face Token

1. Create an account at https://huggingface.co (if you don't have one)
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with read permissions
4. Copy the token (starts with `hf_`)

## Using the Download Scripts

### Option 1: Python Script (Recommended)

```bash
# Set your token as environment variable
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Run the download script
python download_flux_models.py
```

Or pass the token directly:
```bash
python download_flux_models.py --hf-token "hf_YOUR_TOKEN_HERE"
```

### Option 2: Bash Script

```bash
# Set your token as environment variable
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Make script executable and run
chmod +x download_flux_models.sh
./download_flux_models.sh
```

## What Gets Downloaded

- **FLUX.1-dev model** (fp16) → `models/diffusion_models/flux1-dev.safetensors`
- **VAE** → `models/vae/ae.safetensors`
- **CLIP-L encoder** → `models/text_encoders/clip_l.safetensors`
- **T5-XXL encoder** (fp16) → `models/text_encoders/t5xxl_fp16.safetensors`

The scripts automatically skip files that already exist.