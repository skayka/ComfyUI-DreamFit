#!/usr/bin/env python3
"""
Test script for DreamFitSampler node
Verifies the node can be imported and has correct structure
"""

import sys
import os
from pathlib import Path

# Add the module to path
sys.path.insert(0, str(Path(__file__).parent))


def test_node_import():
    """Test that the node can be imported"""
    print("Testing DreamFitSampler import...")
    try:
        from nodes.dreamfit_sampler import DreamFitSampler
        print("✓ DreamFitSampler imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_structure():
    """Test that the node has correct ComfyUI structure"""
    print("\nTesting node structure...")
    try:
        from nodes.dreamfit_sampler import DreamFitSampler
        
        # Check class methods
        required_methods = ['INPUT_TYPES', 'sample']
        for method in required_methods:
            if hasattr(DreamFitSampler, method):
                print(f"✓ {method} method exists")
            else:
                print(f"✗ {method} method missing")
                return False
        
        # Check class attributes
        if hasattr(DreamFitSampler, 'RETURN_TYPES'):
            print(f"✓ RETURN_TYPES defined: {DreamFitSampler.RETURN_TYPES}")
        else:
            print("✗ RETURN_TYPES missing")
            return False
            
        if hasattr(DreamFitSampler, 'FUNCTION'):
            print(f"✓ FUNCTION defined: {DreamFitSampler.FUNCTION}")
        else:
            print("✗ FUNCTION missing")
            return False
            
        if hasattr(DreamFitSampler, 'CATEGORY'):
            print(f"✓ CATEGORY defined: {DreamFitSampler.CATEGORY}")
        else:
            print("✗ CATEGORY missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Structure test error: {e}")
        return False


def test_input_types():
    """Test INPUT_TYPES configuration"""
    print("\nTesting INPUT_TYPES...")
    try:
        from nodes.dreamfit_sampler import DreamFitSampler
        
        input_types = DreamFitSampler.INPUT_TYPES()
        
        # Check required inputs
        required = input_types.get("required", {})
        expected_required = [
            "model", "positive", "negative", "latent_image", 
            "garment_image", "mode", "seed", "steps", "cfg",
            "sampler_name", "scheduler", "denoise"
        ]
        
        for inp in expected_required:
            if inp in required:
                print(f"✓ Required input '{inp}' defined")
            else:
                print(f"✗ Required input '{inp}' missing")
                return False
        
        # Check optional inputs
        optional = input_types.get("optional", {})
        expected_optional = ["lora_path", "pose_image", "person_image"]
        
        for inp in expected_optional:
            if inp in optional:
                print(f"✓ Optional input '{inp}' defined")
            else:
                print(f"✗ Optional input '{inp}' missing")
        
        # Check mode options
        mode_options = required.get("mode", [None])[0]
        expected_modes = ["garment_generation", "pose_control", "virtual_tryon"]
        if mode_options == expected_modes:
            print(f"✓ Mode options correct: {mode_options}")
        else:
            print(f"✗ Mode options incorrect: {mode_options}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ INPUT_TYPES test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_registration():
    """Test that node is properly registered"""
    print("\nTesting node registration...")
    try:
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        if "DreamFitSampler" in NODE_CLASS_MAPPINGS:
            print("✓ DreamFitSampler in NODE_CLASS_MAPPINGS")
        else:
            print("✗ DreamFitSampler missing from NODE_CLASS_MAPPINGS")
            return False
            
        if "DreamFitSampler" in NODE_DISPLAY_NAME_MAPPINGS:
            print(f"✓ Display name: {NODE_DISPLAY_NAME_MAPPINGS['DreamFitSampler']}")
        else:
            print("✗ DreamFitSampler missing from NODE_DISPLAY_NAME_MAPPINGS")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Node registration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dreamfit_imports():
    """Test that DreamFit components can be imported"""
    print("\nTesting DreamFit imports...")
    
    # Add DreamFit to path as the node does
    dreamfit_path = os.path.join(os.path.dirname(__file__), "DreamFit-official", "src")
    if dreamfit_path not in sys.path:
        sys.path.insert(0, dreamfit_path)
    
    try:
        from flux.modules.layers_dreamfit import (
            DoubleStreamBlockLoraProcessor, 
            SingleStreamBlockLoraProcessor
        )
        print("✓ DreamFit processors imported")
        
        from flux.sampling import denoise, prepare, get_schedule
        print("✓ DreamFit sampling functions imported")
        
        from flux.util import load_checkpoint, get_lora_rank
        print("✓ DreamFit utility functions imported")
        
        return True
    except Exception as e:
        print(f"✗ DreamFit import error: {e}")
        print("\nMake sure DreamFit-official is cloned in the correct location")
        print(f"Expected path: {dreamfit_path}")
        return False


def check_lora_files():
    """Check if LoRA checkpoint files exist"""
    print("\nChecking LoRA checkpoint files...")
    
    lora_files = {
        "garment_generation": "pretrained_models/flux_i2i.bin",
        "pose_control": "pretrained_models/flux_i2i_with_pose.bin",
        "virtual_tryon": "pretrained_models/flux_tryon.bin"
    }
    
    all_exist = True
    for mode, filename in lora_files.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {mode}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {mode}: {filename} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\nTo download LoRA files:")
        print("1. Create pretrained_models directory")
        print("2. Download from: https://huggingface.co/bytedance-research/Dreamfit/tree/main")
    
    return all_exist


def main():
    """Run all tests"""
    print("=== DreamFitSampler Node Test ===\n")
    
    tests = [
        ("Node Import", test_node_import),
        ("Node Structure", test_node_structure),
        ("Input Types", test_input_types),
        ("Node Registration", test_node_registration),
        ("DreamFit Imports", test_dreamfit_imports),
        ("LoRA Files", check_lora_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All tests passed! DreamFitSampler is ready to use.")
        print("\nNext steps:")
        print("1. Start ComfyUI")
        print("2. The 'DreamFit Sampler' node should appear in the 'dreamfit' category")
        print("3. Connect: LoadImage → DreamFitSampler → VAEDecode → SaveImage")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()