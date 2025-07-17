# How DreamFit Works - Technical Deep Dive

This document explains the inner workings of DreamFit (ByteDance's garment-centric human generation system) based on our implementation journey.

## Architecture Overview

DreamFit is built on top of FLUX.1-dev but with crucial modifications:

1. **Custom Model Structure**: Uses `model_dreamfit.py` instead of standard FLUX model
2. **Modified Blocks**: Custom `DoubleStreamBlock` and `SingleStreamBlock` from `layers_dreamfit.py` with built-in LoRA layers
3. **Two-Pass Sampling**: Write → Read mechanism for garment feature injection

## Key Components

### 1. Model Architecture

DreamFit loads standard FLUX weights but into a custom model structure:

```python
# In config:
model_path: src.flux.model_dreamfit.Flux

# Custom DoubleStreamBlock includes:
self.img_mlp_lora_1 = LoRALinearLayer(hidden_size, mlp_hidden_dim, rank=32)
self.img_mlp_lora_2 = LoRALinearLayer(mlp_hidden_dim, hidden_size, rank=32)
```

### 2. Processor System

DreamFit uses custom processors that intercept the forward pass:

- **DoubleStreamBlockLoraProcessor**: Processes img and txt separately
- **SingleStreamBlockLoraProcessor**: Processes combined [txt, img] sequence

Processors implement the two-pass mechanism:
- **Write Mode**: Stores attention features (Q, K, V) from garment
- **Read Mode**: Retrieves and uses stored features for generation

### 3. Two-Pass Sampling Mechanism

The core innovation - how garment features guide generation:

```python
# Pass 1: Write mode
# Process garment image, store attention features
processor.bank_img_q = q  # Store garment features
processor.bank_img_k = k
processor.bank_img_v = v

# Pass 2: Read mode  
# Use stored features to guide person generation
q = q + self.ref_qkv_lora_q(self.bank_img_q) * self.lora_weight
k = k + self.ref_qkv_lora_k(self.bank_img_k) * self.lora_weight
v = v + self.ref_qkv_lora_v(self.bank_img_v) * self.lora_weight
```

### 4. Conditioning Format

DreamFit uses special conditioning tokens:
- Person generation: uses actual text prompt
- Garment processing: uses "cloth" token
- Negative: uses empty/negative prompt

## Key Differences from Standard FLUX

1. **Built-in LoRA layers**: Not added via ComfyUI's LoRA system but part of model structure
2. **Processor injection**: Replaces normal attention with DreamFit processors
3. **Feature storage**: Maintains banks of attention features between passes
4. **Special tokens**: "cloth" token for garment conditioning

## Memory Considerations

- Creates processors only for blocks with trained weights (not all 57)
- Stores attention features during write pass (memory intensive)
- Needs cleanup of stored features after generation

## Integration Challenges with ComfyUI

1. **Model Structure Mismatch**: ComfyUI's FLUX model lacks built-in LoRA layers
2. **Dynamic Patching**: Must add LoRA layers and processors at runtime
3. **Interface Translation**: DreamFit expects different calling conventions than ComfyUI
4. **Memory Management**: Two-pass system with feature storage needs careful cleanup

## The Three DreamFit Modes

1. **Garment Generation**: Generate person wearing specific garment
2. **Pose Control**: Add pose guidance to garment generation  
3. **Virtual Try-On**: Replace garment on existing person image

Each mode uses different LoRA checkpoints but same underlying mechanism.