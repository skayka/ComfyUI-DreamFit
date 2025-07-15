# ComfyUI-DreamFit

Garment-centric human generation nodes for ComfyUI using DreamFit with Flux.


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
git clone https://github.com/skayka/ComfyUI-DreamFit.git

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
3. Restart ComfyUI after downloading models

#### Import errors in ComfyUI
1. Make sure you've installed requirements: `pip install -r requirements.txt`
2. Restart ComfyUI after installation
3. Check the ComfyUI console for specific error messages

## Available Nodes

### 1. DreamFit Checkpoint Loader
Loads DreamFit model checkpoints and initializes the Anything-Dressing Encoder.

**Inputs:**
- `model_name`: Choose from available DreamFit models
- `device`: Processing device (cuda/cpu)
- `dtype`: Model precision (fp16/bf16/fp32)

**Outputs:**
- `DREAMFIT_MODEL`: The loaded model configuration
- `DREAMFIT_ENCODER`: The Anything-Dressing Encoder
- `DREAMFIT_CONFIG`: Model configuration

### 2. DreamFit Encode
Encodes garment images and text prompts into conditioning for generation.

**Inputs:**
- `encoder`: From checkpoint loader
- `garment_image`: The garment to generate
- `positive_prompt`: Description of desired output
- `negative_prompt`: What to avoid in generation
- `model_image` (optional): Reference pose/model
- `garment_description`: Brief garment description
- `garment_category`: Type of garment (casual/formal/sportswear/traditional)
- `enhance_prompt`: Auto-enhance prompts
- `use_model_parse`: Parse model image for better results
- `injection_strength`: Control garment influence (0.1-2.0)

**Outputs:**
- `conditioning`: DreamFit conditioning for sampling
- `enhanced_prompt`: Improved positive prompt
- `enhanced_negative`: Improved negative prompt

### 3. DreamFit Flux Adapter V2
Applies DreamFit adaptation to a Flux model with CLIP integration.

**Inputs:**
- `model`: Your Flux model
- `clip`: CLIP model for text encoding
- `dreamfit_conditioning`: From encode node
- `positive`: Enhanced positive prompt
- `negative`: Enhanced negative prompt
- `lora_strength`: LoRA adaptation strength (0.0-2.0)
- `injection_strength`: Feature injection strength (0.0-2.0)
- `lora_merge_mode`: How to merge LoRA weights
- `injection_mode`: Attention injection strategy
- `attention_mode`: Which attention layers to modify
- `use_cached_embeddings`: Speed optimization

**Outputs:**
- `model`: Flux model with DreamFit adaptation
- `positive`: Positive conditioning
- `negative`: Negative conditioning

### 4. DreamFit K-Sampler
Custom sampler optimized for DreamFit generation.

**Inputs:**
- Standard KSampler inputs (model, seed, steps, cfg, etc.)
- `dreamfit_conditioning`: From encode node

**Outputs:**
- `LATENT`: Generated image latent

### 5. DreamFit Unified
Complete DreamFit integration in a single node.

**Inputs:**
- `model`: Flux diffusion model (from UNETLoader)
- `positive/negative`: Pre-encoded conditioning from CLIP
- `garment_image`: Garment to process
- `dreamfit_model`: Select model type
- `strength`: Overall adaptation strength
- `model_image` (optional): Reference pose for try-on
- `injection_strength`: Garment feature strength
- `injection_mode`: Feature injection strategy

**Outputs:**
- `model`: Enhanced Flux model
- `positive/negative`: Enhanced conditioning
- `debug_garment`: Processed garment (224x224) for debugging

### 6. DreamFit Simple
All-in-one node for easy DreamFit generation.

**Inputs:**
- `model`: Flux model
- `clip`: CLIP model
- `vae`: VAE model
- `dreamfit_model`: Select DreamFit model
- `garment_image`: Garment to generate
- `positive/negative`: Text prompts
- `seed`, `steps`, `cfg`, `denoise`: Standard generation parameters
- `model_image` (optional): Reference model/pose

**Outputs:**
- `samples`: Generated latent image

### 6. DreamFit Sampler Advanced
Advanced sampler with additional controls.

**Features:**
- Noise modes: default, garment_aware, structured
- Injection schedules: constant, linear, cosine, step
- Step control for multi-stage generation

## Workflow Examples

Four example workflows are included in the `workflows/` directory:

### 1. Simple Workflow (`dreamfit_simple_workflow.json`)
- Uses the all-in-one **DreamFit Simple** node
- Minimal setup required
- Best for quick testing and basic generation

### 2. Basic Workflow (`dreamfit_basic_workflow.json`)
- Full node setup with individual components
- More control over the generation process
- Good balance of simplicity and flexibility

### 3. Advanced Workflow (`dreamfit_advanced_workflow.json`)
- Complete setup with all advanced features
- Virtual try-on with model images
- Advanced sampling strategies
- Maximum control and customization

### Loading Workflows
1. Open ComfyUI
2. Click "Load" in the menu
3. Navigate to `ComfyUI/custom_nodes/ComfyUI-DreamFit/workflows/`
4. Select the desired workflow JSON file

### 4. Unified Workflow (`dreamfit_unified_workflow.json`)
- Uses the new **DreamFit Unified** node
- Proper Flux model loading (UNETLoader, DualCLIPLoader, VAELoader)
- Shows debug output of processed garment
- Best for understanding the complete pipeline

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
- Try running with specific model: `python download_models.py --model flux_i2i`
- Manually download from [HuggingFace](https://huggingface.co/bytedance-research/Dreamfit) and place in `ComfyUI/models/dreamfit/`

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

- Report issues on [GitHub Issues](https://github.com/skayka/ComfyUI-DreamFit/issues)
- Join the discussion in ComfyUI Discord