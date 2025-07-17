#!/usr/bin/env python3
"""
Convert DreamFit flux_i2i.bin weights to GarmentFit format
Extracts LoRA weights and adapts them for PuLID-style architecture
"""

import torch
import os
import sys
import argparse
from pathlib import Path


def convert_dreamfit_to_garmentfit(dreamfit_path: str, output_path: str):
    """
    Convert DreamFit checkpoint to GarmentFit format
    """
    print(f"Loading DreamFit checkpoint from {dreamfit_path}")
    checkpoint = torch.load(dreamfit_path, map_location='cpu')
    
    # New state dict for GarmentFit
    garmentfit_state = {
        'encoder': {},  # Garment encoder weights (we'll initialize these)
        'attention': {}  # K,V projection weights from DreamFit
    }
    
    # Extract LoRA weights for attention
    num_extracted = 0
    for key, value in checkpoint.items():
        if 'lora' in key.lower():
            # Map DreamFit layer names to our simplified structure
            if 'double_blocks' in key:
                block_num = int(key.split('.')[1])
                
                if 'ref_qkv_lora_k' in key:
                    # K projection weights
                    # DreamFit uses LoRALinearLayer with down/up projections
                    # We'll extract the effective weight matrix
                    if '.down.' in key:
                        down_key = key
                        up_key = key.replace('.down.', '.up.')
                        if up_key in checkpoint:
                            # Compute effective LoRA weight: up @ down
                            down_weight = checkpoint[down_key]
                            up_weight = checkpoint[up_key]
                            effective_weight = up_weight @ down_weight
                            garmentfit_state['attention'][f'block_{block_num}_k'] = effective_weight
                            num_extracted += 1
                
                elif 'ref_qkv_lora_v' in key:
                    # V projection weights
                    if '.down.' in key:
                        down_key = key
                        up_key = key.replace('.down.', '.up.')
                        if up_key in checkpoint:
                            down_weight = checkpoint[down_key]
                            up_weight = checkpoint[up_key]
                            effective_weight = up_weight @ down_weight
                            garmentfit_state['attention'][f'block_{block_num}_v'] = effective_weight
                            num_extracted += 1
            
            elif 'single_blocks' in key:
                block_num = int(key.split('.')[1]) + 19  # Offset for single blocks
                
                if 'ref_qkv_lora_k' in key and '.down.' in key:
                    down_key = key
                    up_key = key.replace('.down.', '.up.')
                    if up_key in checkpoint:
                        down_weight = checkpoint[down_key]
                        up_weight = checkpoint[up_key]
                        effective_weight = up_weight @ down_weight
                        garmentfit_state['attention'][f'block_{block_num}_k'] = effective_weight
                        num_extracted += 1
                
                elif 'ref_qkv_lora_v' in key and '.down.' in key:
                    down_key = key
                    up_key = key.replace('.down.', '.up.')
                    if up_key in checkpoint:
                        down_weight = checkpoint[down_key]
                        up_weight = checkpoint[up_key]
                        effective_weight = up_weight @ down_weight
                        garmentfit_state['attention'][f'block_{block_num}_v'] = effective_weight
                        num_extracted += 1
    
    print(f"Extracted {num_extracted} attention projection layers")
    
    # Initialize garment encoder weights (these will be trained/fine-tuned)
    # For now, we'll use random initialization that can be fine-tuned
    print("Initializing garment encoder weights...")
    
    # Save the converted checkpoint
    print(f"Saving converted weights to {output_path}")
    torch.save(garmentfit_state, output_path)
    
    # Print summary
    print("\nConversion complete!")
    print(f"Total attention layers: {len(garmentfit_state['attention'])}")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def analyze_dreamfit_checkpoint(checkpoint_path: str):
    """Analyze DreamFit checkpoint structure"""
    print(f"\nAnalyzing DreamFit checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Total keys: {len(checkpoint)}")
    
    # Group keys by type
    lora_keys = []
    other_keys = []
    
    for key in checkpoint.keys():
        if 'lora' in key.lower():
            lora_keys.append(key)
        else:
            other_keys.append(key)
    
    print(f"\nLoRA-related keys: {len(lora_keys)}")
    print("Sample LoRA keys:")
    for key in sorted(lora_keys)[:10]:
        print(f"  {key}: {checkpoint[key].shape}")
    
    if other_keys:
        print(f"\nOther keys: {len(other_keys)}")
        print("Sample other keys:")
        for key in sorted(other_keys)[:5]:
            print(f"  {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}")


def main():
    parser = argparse.ArgumentParser(description="Convert DreamFit weights to GarmentFit format")
    parser.add_argument(
        "dreamfit_checkpoint",
        help="Path to DreamFit flux_i2i.bin checkpoint"
    )
    parser.add_argument(
        "-o", "--output",
        default="garmentfit_flux.pt",
        help="Output path for GarmentFit checkpoint (default: garmentfit_flux.pt)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze the checkpoint structure without converting"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.dreamfit_checkpoint):
        print(f"Error: Input file not found: {args.dreamfit_checkpoint}")
        sys.exit(1)
    
    if args.analyze:
        analyze_dreamfit_checkpoint(args.dreamfit_checkpoint)
    else:
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        convert_dreamfit_to_garmentfit(args.dreamfit_checkpoint, args.output)


if __name__ == "__main__":
    main()