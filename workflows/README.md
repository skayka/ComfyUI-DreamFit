# DreamFit Workflow Examples

This directory contains example workflows for ComfyUI-DreamFit.

## Available Workflows

### 1. dreamfit_basic_workflow.json
Basic garment-centric generation workflow demonstrating:
- Loading DreamFit models
- Encoding garment images with text prompts
- Adapting Flux models with DreamFit features
- Generating images with optimized sampling

**How to use:**
1. Load the workflow in ComfyUI
2. Replace the garment image with your own
3. Adjust the text prompt to describe your desired output
4. Run the workflow

**Key Parameters:**
- `encoding_strength`: Controls garment feature strength (0.0-2.0)
- `injection_strength`: Controls attention injection (0.0-2.0)
- `garment_guidance_scale`: Additional guidance for garments (0.0-5.0)

## Tips for Best Results

1. **Image Quality**
   - Use high-quality garment images with clean backgrounds
   - Ensure good lighting and clear details
   - Square images work best (1024x1024)

2. **Prompts**
   - Be specific about garment details
   - Include style preferences (elegant, casual, etc.)
   - Use the garment_description field for accuracy

3. **Advanced Features**
   - Try different feature_injection_schedules
   - Experiment with attention_mode settings
   - Use multiple garments with DreamFitEncodeMultiple

## Workflow Components

### Essential Nodes
- `DreamFitCheckpointLoader`: Loads DreamFit models
- `DreamFitEncode`: Processes garment images
- `DreamFitFluxAdapter`: Adapts Flux models
- `DreamFitKSampler`: Optimized sampling

### Optional Nodes
- `DreamFitEncodeMultiple`: Combine multiple garments
- `DreamFitAdvancedSampler`: Fine-grained control
- `DreamFitRemoveAdapter`: Clean up models

## Creating Custom Workflows

1. Start with the basic workflow
2. Add your specific requirements
3. Connect additional processing nodes
4. Save and share your workflows!

For more examples and community workflows, visit our [GitHub repository](https://github.com/yourusername/ComfyUI-DreamFit).