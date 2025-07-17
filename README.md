# ComfyUI-DreamFit

Custom nodes for garment-centric human generation in ComfyUI using DreamFit's trained models.

## Overview

This implementation adapts [DreamFit](https://github.com/bytedance/DreamFit) weights for use in ComfyUI, using a PuLID-inspired architecture that:
- Properly loads and applies DreamFit's trained LoRA weights
- Uses single-pass attention injection (simpler than DreamFit's two-pass)
- Integrates seamlessly with ComfyUI's model system

## Installation

1. Clone this repository to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-DreamFit
```

2. Install requirements:
```bash
cd ComfyUI-DreamFit
pip install -r requirements.txt
```

3. Download DreamFit checkpoint:
```bash
# Download flux_i2i.bin from DreamFit and place it in:
ComfyUI/models/dreamfit/flux_i2i.bin
```

## Nodes

### Load GarmentFit Model
Loads the DreamFit checkpoint and prepares it for use.
- **Input**: checkpoint path (select flux_i2i.bin)
- **Output**: GARMENTFIT model

### Apply GarmentFit
Applies garment features to your generation.
- **Inputs**:
  - `model`: Your base FLUX model
  - `clip`: CLIP model for encoding
  - `garmentfit`: The loaded GarmentFit model
  - `garment_image`: The garment image to apply
  - `positive/negative`: Your text conditioning
  - `weight`: Strength of garment influence (0.0-2.0, default 0.8)
  - `start_at/end_at`: When to apply garment influence (0.0-1.0)
- **Output**: Modified MODEL

## Usage

1. Load your FLUX checkpoint and CLIP model as usual
2. Load the GarmentFit model using "Load GarmentFit Model" node
3. Connect everything to "Apply GarmentFit" node
4. Use the output model in your KSampler

Example workflow:
```
[Load Checkpoint] → [Load CLIP]
                          ↓
[Load GarmentFit Model] → [Apply GarmentFit] → [KSampler] → [VAE Decode]
                          ↑
                    [Load Image]
```

## Technical Details

This implementation:
- Extracts LoRA weights from DreamFit's checkpoint (down/up weight pairs)
- Creates proper attention patches for K,V projections
- Injects garment tokens into the attention mechanism
- Supports scheduling (start_at/end_at) for fine control

### Architecture

The key innovation is properly using DreamFit's trained LoRA weights:

1. **Weight Loading**: Parses the checkpoint to find all LoRA down/up weight pairs
2. **LoRA Application**: Combines down/up weights to create effective transformations
3. **Attention Patching**: Injects garment K,V into attention layers
4. **Scheduling**: Only applies within specified timestep range

### Performance

- Memory efficient: Only loads necessary weights
- Fast inference: Single-pass instead of two-pass
- Compatible with all FLUX models and samplers

## Troubleshooting

### Models not found
Ensure `flux_i2i.bin` is in `ComfyUI/models/dreamfit/`

### Out of memory
- Reduce batch size
- Lower the weight parameter
- Use fewer sampling steps

### No effect
- Increase weight parameter
- Check that garment image is clear with good contrast
- Ensure start_at < end_at

## Credits

- Original DreamFit by ByteDance Research
- Architecture inspired by PuLID
- Built for ComfyUI