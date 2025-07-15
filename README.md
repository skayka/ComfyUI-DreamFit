# ComfyUI-DreamFit

Garment-centric human generation nodes for ComfyUI using DreamFit with Flux.

> **Note**: Replace `yourusername` in the installation instructions with your actual GitHub username after creating your repository.

DreamFit is a powerful adapter system that enhances Flux models with garment-aware generation capabilities, enabling high-quality fashion and clothing generation.

## Features

- 🎨 **Garment-Centric Generation**: Generate humans wearing specific garments with high fidelity
- 👗 **Virtual Try-On**: Try different garments on models
- 🎭 **Pose Control**: Generate with specific poses while maintaining garment details
- 🔧 **Flux Integration**: Seamlessly works with Flux models in ComfyUI
- ⚡ **Adaptive Attention**: Smart injection of garment features into the generation process
- 🎯 **LoRA Adaptation**: Efficient model adaptation without full fine-tuning

## Installation

### Prerequisites
- ComfyUI installed and working
- Python 3.8 or higher
- Git (for installation method 1)

### Method 1: Git Clone (Recommended)
```bash
# Navigate to your ComfyUI custom nodes directory
cd ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/yourusername/ComfyUI-DreamFit.git

# Navigate to the installed directory
cd ComfyUI-DreamFit

# Install Python dependencies
pip install -r requirements.txt

# Optional: Download models (do this after restarting ComfyUI)
python download_models.py
```

### Method 2: ComfyUI Manager
If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:
1. Open ComfyUI Manager
2. Search for "DreamFit"
3. Click Install
4. Restart ComfyUI

### Method 3: Manual Installation
1. Download the repository as ZIP
2. Extract to `ComfyUI/custom_nodes/ComfyUI-DreamFit`
3. Open a terminal in the extracted directory
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Download Models
After installation, download the required DreamFit models:

```bash
# Navigate to the ComfyUI-DreamFit directory
cd ComfyUI/custom_nodes/ComfyUI-DreamFit

# Download all models (855MB total)
python download_models.py

# Or download specific models
python download_models.py --model flux_i2i
python download_models.py --model flux_tryon

# List available models
python download_models.py --list

# Verify downloaded models
python download_models.py --verify
```

The models will be automatically downloaded to:
- If inside ComfyUI: `ComfyUI/models/dreamfit/`
- Otherwise: `./dreamfit_models/`

### Troubleshooting Installation

#### "No module named 'folder_paths'" during install
This is normal! The installation will complete successfully. This error only appears because the installer tries to verify the installation outside of ComfyUI.

#### Models not found
1. Make sure you've downloaded the models using `python download_models.py`
2. Check that models are in `ComfyUI/models/dreamfit/`
3. Try using the node's "Download if missing" option

#### Import errors in ComfyUI
1. Make sure you've installed requirements: `pip install -r requirements.txt`
2. Restart ComfyUI after installation
3. Check the ComfyUI console for specific error messages

## Available Nodes

### 1. DreamFit Checkpoint Loader
Loads DreamFit model checkpoints and initializes the Anything-Dressing Encoder.

**Inputs:**
- `model_name`: Choose from available DreamFit models
- `download_missing`: Auto-download if model not found

**Outputs:**
- `DREAMFIT_MODEL`: The loaded model configuration
- `DREAMFIT_ENCODER`: The Anything-Dressing Encoder
- `DREAMFIT_CONFIG`: Model configuration

### 2. DreamFit Encode
Encodes garment images and text prompts into conditioning for generation.

**Inputs:**
- `dreamfit_model`: From checkpoint loader
- `encoder`: From checkpoint loader
- `garment_image`: The garment to generate
- `text_prompt`: Description of desired output
- `model_image` (optional): Reference pose/model
- `enhance_prompt`: Auto-enhance prompts
- `injection_strength`: Control garment influence (0-1)

**Outputs:**
- `conditioning`: DreamFit conditioning for sampling
- `enhanced_prompt`: Improved text prompt
- `processed_garment`: Preprocessed garment image

### 3. DreamFit Flux Adapter
Applies DreamFit adaptation to a Flux model.

**Inputs:**
- `flux_model`: Your Flux model
- `dreamfit_model`: From checkpoint loader
- `conditioning`: From encode node
- `lora_strength`: LoRA adaptation strength
- `merge_lora`: Merge LoRA weights into model

**Outputs:**
- `adapted_model`: Flux model with DreamFit
- `adapter_info`: Adaptation details

### 4. DreamFit K-Sampler
Custom sampler optimized for DreamFit generation.

**Inputs:**
- Standard KSampler inputs (model, seed, steps, cfg, etc.)
- `dreamfit_conditioning`: From encode node

**Outputs:**
- `LATENT`: Generated image latent

### 5. DreamFit Sampler Advanced
Advanced sampler with additional controls.

**Features:**
- Noise modes: default, garment_aware, structured
- Injection schedules: constant, linear, cosine, step
- Step control for multi-stage generation

## Workflow Examples

### Basic Garment Generation
1. Load Flux model → Load VAE → Load CLIP
2. **DreamFit Checkpoint Loader** (flux_i2i)
3. Load garment image
4. **DreamFit Encode** (garment + "A person wearing the garment")
5. **DreamFit Flux Adapter** (apply to Flux model)
6. Empty Latent Image
7. **DreamFit K-Sampler**
8. VAE Decode → Save Image

### Virtual Try-On
1. Load Flux model → Load VAE → Load CLIP
2. **DreamFit Checkpoint Loader** (flux_tryon)
3. Load garment image + model/pose image
4. **DreamFit Encode** (with model_image input)
5. **DreamFit Flux Adapter**
6. **DreamFit Sampler Advanced** (garment_aware noise)
7. VAE Decode → Save Image

## Tips for Best Results

1. **Image Quality**: Use high-quality garment images with clean backgrounds
2. **Prompts**: Be specific about the desired style and context
3. **Injection Strength**: Start with 0.5 and adjust based on results
4. **Sampling Steps**: 20-30 steps usually sufficient
5. **CFG Scale**: 7-8 works well for most cases

## Model Information

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| flux_i2i | 284MB | Basic garment generation | General fashion images |
| flux_i2i_with_pose | 284MB | Pose-controlled generation | Specific poses/positions |
| flux_tryon | 287MB | Virtual try-on | Trying clothes on models |

## Troubleshooting

### "No module named 'dreamfit_core'"
- Ensure you're in the correct directory when installing
- Try `python -m pip install -e .` from the ComfyUI-DreamFit directory

### Models not downloading
- Check internet connection
- Manually download from HuggingFace and place in `ComfyUI/models/dreamfit/`

### Out of memory errors
- Reduce batch size to 1
- Use CPU offloading in ComfyUI settings
- Try the fp16 versions of models

## Technical Details

DreamFit uses:
- **Anything-Dressing Encoder**: 83.4M parameter encoder for garment features
- **Adaptive Attention Injection**: Injects garment features into Flux attention layers
- **LoRA Adaptation**: Efficient 16-rank LoRA for model adaptation

## Credits

- Original DreamFit: [ByteDance Research](https://github.com/bytedance/DreamFit)
- ComfyUI: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## License

This project follows the same license as the original DreamFit implementation.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

- Report issues on [GitHub Issues](https://github.com/yourusername/ComfyUI-DreamFit/issues)
- Join the discussion in ComfyUI Discord