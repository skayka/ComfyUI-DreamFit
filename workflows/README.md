# DreamFit Workflow Examples

This directory contains example workflows for using DreamFit nodes in ComfyUI.

## Available Workflows

### 1. Simple Workflow (`dreamfit_simple_workflow.json`)
**Best for:** Quick testing and basic garment generation

This workflow uses the all-in-one **DreamFit Simple** node which handles most of the complexity internally.

**Features:**
- Minimal node setup
- Single node for complete generation
- Pre-configured optimal settings
- Easy to understand and modify

**Required inputs:**
- Flux model checkpoint
- Garment image
- Text prompts (positive/negative)

### 2. Basic Workflow (`dreamfit_basic_workflow.json`)
**Best for:** Standard garment generation with more control

This workflow shows the standard DreamFit pipeline with individual nodes for each step.

**Features:**
- Full control over each processing step
- DreamFit Flux Adapter V2 with CLIP integration
- Enhanced prompt generation
- Separate encoding and adaptation stages

**Node chain:**
1. DreamFit Checkpoint Loader → Load model
2. DreamFit Encode → Process garment
3. DreamFit Flux Adapter V2 → Apply to Flux
4. DreamFit KSampler → Generate image

### 3. Advanced Workflow (`dreamfit_advanced_workflow.json`)
**Best for:** Virtual try-on and complex generation tasks

This workflow demonstrates advanced features including model images, custom sampling, and fine-tuned control.

**Features:**
- Virtual try-on with model/pose images
- Advanced sampling strategies
- Flux guidance integration
- Custom noise modes
- Model validation node
- Composite image generation

**Advanced techniques:**
- Image compositing for inpainting-style generation
- Custom sampler configuration
- Multi-stage generation pipeline
- Mask-based regional control

## How to Use These Workflows

### Loading a Workflow
1. Open ComfyUI in your browser
2. Click the **"Load"** button in the menu
3. Navigate to `ComfyUI/custom_nodes/ComfyUI-DreamFit/workflows/`
4. Select the desired `.json` file
5. The workflow will load with all nodes connected

### Before Running
1. **Download Models**: Ensure you've run `python download_models.py`
2. **Load Images**: Replace placeholder image paths with your own:
   - `garment.png` → Your garment image
   - `model.png` → Your model/pose reference (advanced workflow)
3. **Check Model Paths**: Ensure Flux model checkpoint paths are correct

### Customizing Workflows

#### Changing Models
- **DreamFit Models**: 
  - `flux_i2i`: Basic garment generation
  - `flux_i2i_with_pose`: Pose-aware generation
  - `flux_tryon`: Virtual try-on
  
#### Key Parameters
- **Injection Strength** (0.1-2.0): Controls garment influence
  - Lower (0.1-0.5): Subtle garment features
  - Medium (0.5-1.0): Balanced generation
  - Higher (1.0-2.0): Strong garment preservation

- **LoRA Strength** (0.0-2.0): Model adaptation intensity
  - 0.0: No adaptation
  - 1.0: Standard adaptation
  - 2.0: Maximum adaptation

- **Steps**: Generation quality
  - 15-20: Fast preview
  - 20-30: Standard quality
  - 30-50: High quality

- **CFG Scale**: Prompt adherence
  - 5-7: More creative
  - 7-8: Balanced
  - 8-12: Strict adherence

## Workflow Tips

### For Best Results

1. **Image Preparation**
   - Use high-quality garment images
   - Clean or white backgrounds work best
   - Ensure garment is clearly visible
   - Square images (1024x1024) recommended

2. **Prompt Engineering**
   - Be specific about context and style
   - Include details about lighting and pose
   - Use the negative prompt effectively
   - Enable prompt enhancement for better results

3. **Performance Optimization**
   - Start with fp16 precision for speed
   - Use CPU offloading if running out of VRAM
   - Reduce resolution for testing
   - Cache models between generations

### Common Issues

**"Model not found"**
- Run `python download_models.py` first
- Check model dropdown in DreamFit Checkpoint Loader
- Ensure models are in `ComfyUI/models/dreamfit/`

**Poor garment preservation**
- Increase injection strength
- Improve garment description
- Use higher resolution images
- Try the `flux_tryon` model

**Out of memory**
- Switch to fp16 precision
- Reduce batch size to 1
- Enable CPU offloading
- Lower resolution

## Creating Custom Workflows

### Basic Structure
1. **Input Stage**: Load models and images
2. **Encoding Stage**: Process garment with DreamFit Encode
3. **Adaptation Stage**: Apply to Flux model
4. **Generation Stage**: Sample with DreamFit sampler
5. **Output Stage**: Decode and save

### Node Connections
- `DREAMFIT_MODEL` → Connects encoding to adaptation
- `DREAMFIT_ENCODER` → Processes garment features
- `DREAMFIT_CONDITIONING` → Carries garment information
- Standard ComfyUI types (MODEL, CLIP, VAE, etc.) work normally

### Advanced Techniques
- **Multi-garment**: Chain multiple encode nodes
- **Style mixing**: Blend multiple conditionings
- **Regional control**: Use masks for specific areas
- **Progressive generation**: Multi-stage sampling

## Sharing Workflows

When sharing workflows:
1. Use relative paths for images
2. Document required models
3. Include recommended settings
4. Test on clean installation
5. Export with "Save (API Format)" for compatibility

## Need Help?

- Check the main [README](../README.md) for node documentation
- Report issues on [GitHub](https://github.com/skayka/ComfyUI-DreamFit/issues)
- Join ComfyUI Discord for community support