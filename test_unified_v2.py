#!/usr/bin/env python3
"""
Test script for DreamFitUnifiedV2 node
Verifies basic functionality and structure
"""

import sys
import os
from pathlib import Path

# Add the module to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        # Test custom type import
        from dreamfit_types import DreamFitFeatures
        print("✓ DreamFitFeatures type imported successfully")
        
        # Test node import
        from nodes.dreamfit_unified_v2 import DreamFitUnifiedV2
        print("✓ DreamFitUnifiedV2 node imported successfully")
        
        # Test debug utils import
        from utils.debug_visualization import create_debug_grid, create_attention_heatmap
        print("✓ Debug visualization utilities imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_node_structure():
    """Test that the node has correct structure"""
    print("\nTesting node structure...")
    try:
        from nodes.dreamfit_unified_v2 import DreamFitUnifiedV2
        
        # Check class methods
        required_methods = ['INPUT_TYPES', 'process']
        for method in required_methods:
            if hasattr(DreamFitUnifiedV2, method):
                print(f"✓ {method} method exists")
            else:
                print(f"✗ {method} method missing")
                return False
        
        # Check class attributes
        if hasattr(DreamFitUnifiedV2, 'RETURN_TYPES'):
            print(f"✓ RETURN_TYPES defined: {DreamFitUnifiedV2.RETURN_TYPES}")
        else:
            print("✗ RETURN_TYPES missing")
            return False
            
        if hasattr(DreamFitUnifiedV2, 'FUNCTION'):
            print(f"✓ FUNCTION defined: {DreamFitUnifiedV2.FUNCTION}")
        else:
            print("✗ FUNCTION missing")
            return False
            
        if hasattr(DreamFitUnifiedV2, 'CATEGORY'):
            print(f"✓ CATEGORY defined: {DreamFitUnifiedV2.CATEGORY}")
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
        from nodes.dreamfit_unified_v2 import DreamFitUnifiedV2
        
        input_types = DreamFitUnifiedV2.INPUT_TYPES()
        
        # Check required inputs
        required = input_types.get("required", {})
        expected_required = ["model", "positive", "negative", "garment_image", "dreamfit_checkpoint", "strength", "injection_strength"]
        
        for inp in expected_required:
            if inp in required:
                print(f"✓ Required input '{inp}' defined")
            else:
                print(f"✗ Required input '{inp}' missing")
                return False
        
        # Check optional inputs
        optional = input_types.get("optional", {})
        expected_optional = ["model_image", "injection_mode", "debug_mode"]
        
        for inp in expected_optional:
            if inp in optional:
                print(f"✓ Optional input '{inp}' defined")
            else:
                print(f"✗ Optional input '{inp}' missing")
        
        return True
    except Exception as e:
        print(f"✗ INPUT_TYPES test error: {e}")
        return False


def test_custom_type():
    """Test DreamFitFeatures custom type"""
    print("\nTesting DreamFitFeatures custom type...")
    try:
        from dreamfit_types import DreamFitFeatures
        
        # Test initialization
        test_data = {
            "garment_token": "test_token",
            "pooled_features": "test_pooled",
            "patch_features": "test_patch",
            "features": {"test": "features"},
            "attention_weights": {"layer1": "weights"},
            "encoder_config": {"dim": 768},
            "pose_features": None
        }
        
        features = DreamFitFeatures(test_data)
        print("✓ DreamFitFeatures initialized successfully")
        
        # Test attributes
        if features.garment_token == "test_token":
            print("✓ Attributes accessible")
        else:
            print("✗ Attribute access failed")
            return False
        
        # Test methods
        if hasattr(features, 'to_dict'):
            print("✓ to_dict method exists")
        else:
            print("✗ to_dict method missing")
            return False
            
        if hasattr(features, 'get_injection_config'):
            print("✓ get_injection_config method exists")
        else:
            print("✗ get_injection_config method missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Custom type test error: {e}")
        return False


def test_node_mappings():
    """Test that node is properly registered in __init__.py"""
    print("\nTesting node registration...")
    try:
        # Import from __init__.py
        from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        if "DreamFitUnifiedV2" in NODE_CLASS_MAPPINGS:
            print("✓ DreamFitUnifiedV2 in NODE_CLASS_MAPPINGS")
        else:
            print("✗ DreamFitUnifiedV2 missing from NODE_CLASS_MAPPINGS")
            return False
            
        if "DreamFitUnifiedV2" in NODE_DISPLAY_NAME_MAPPINGS:
            print(f"✓ Display name: {NODE_DISPLAY_NAME_MAPPINGS['DreamFitUnifiedV2']}")
        else:
            print("✗ DreamFitUnifiedV2 missing from NODE_DISPLAY_NAME_MAPPINGS")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Node mapping test error: {e}")
        return False


def test_debug_utils():
    """Test debug visualization utilities"""
    print("\nTesting debug visualization utilities...")
    try:
        from utils import debug_visualization
        
        # Check available functions
        expected_functions = [
            'create_debug_grid',
            'create_attention_heatmap',
            'apply_colormap',
            'overlay_attention_on_image',
            'create_feature_visualization',
            'create_comparison_grid'
        ]
        
        for func_name in expected_functions:
            if hasattr(debug_visualization, func_name):
                print(f"✓ {func_name} function available")
            else:
                print(f"✗ {func_name} function missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Debug utils test error: {e}")
        return False


def main():
    """Run all tests"""
    print("=== DreamFitUnifiedV2 Implementation Test ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Node Structure", test_node_structure),
        ("Input Types", test_input_types),
        ("Custom Type", test_custom_type),
        ("Node Mappings", test_node_mappings),
        ("Debug Utils", test_debug_utils)
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
        print("\n✨ All tests passed! DreamFitUnifiedV2 implementation is ready.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()