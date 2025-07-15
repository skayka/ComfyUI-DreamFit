# DreamFit Example Workflows

This directory contains example workflows demonstrating how to use ComfyUI-DreamFit nodes.

## Available Workflows

### 1. basic_garment_generation.json
A simple workflow for generating images of people wearing specific garments.

**What it does:**
- Loads a garment image
- Encodes it with DreamFit
- Generates a person wearing the garment
- Uses the basic flux_i2i model

**Key settings:**
- Model: flux_i2i
- Injection strength: 0.5
- Sampling: 20 steps, CFG 7.5

### 2. virtual_tryon_advanced.json
Advanced workflow for virtual try-on with pose control.

**What it does:**
- Loads both garment and model/pose images
- Uses garment mask for better results
- Applies advanced sampling with garment-aware noise
- Shows adapter information

**Key settings:**
- Model: flux_tryon
- Injection strength: 0.7
- Noise mode: garment_aware
- Injection schedule: cosine
- Sampling: 25 steps, CFG 8.0

## How to Use

1. **Import the workflow:**
   - Open ComfyUI
   - Click "Load" button
   - Select the JSON file
   
2. **Load your images:**
   - Replace "garment.png" with your garment image
   - For virtual try-on, also replace "model_pose.png"
   
3. **Adjust parameters:**
   - **Injection strength**: Higher = stronger garment influence
   - **Text prompt**: Describe the desired output
   - **Sampling steps**: More steps = higher quality but slower
   
4. **Run the workflow:**
   - Click "Queue Prompt" to generate

## Tips

- **Image preparation:**
  - Use garment images with clean, white backgrounds
  - Ensure good lighting and clear details
  - Square images (1024x1024) work best

- **Prompt engineering:**
  - Be specific about style and setting
  - Include details about lighting and quality
  - Use negative prompts to avoid unwanted features

- **Performance:**
  - Start with lower resolution for testing
  - Use fewer sampling steps for quick previews
  - Enable CPU offloading if running out of VRAM

## Common Issues

**"Model not found"**
- Run `python install.py --download-models` to download DreamFit models

**Out of memory**
- Reduce batch size to 1
- Lower the resolution
- Enable CPU offloading in ComfyUI settings

**Poor quality results**
- Check your garment image quality
- Adjust injection strength
- Try different noise modes (for advanced sampler)
- Increase sampling steps

## Creating Custom Workflows

You can create your own workflows by:
1. Combining DreamFit nodes with other ComfyUI nodes
2. Using ControlNet for additional pose control
3. Adding image preprocessing nodes
4. Implementing multi-stage generation

For more information, see the main README.