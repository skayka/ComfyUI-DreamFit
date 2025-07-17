#!/usr/bin/env python3
"""Test if nodes can be imported"""

import sys
import os

print(f"Python: {sys.version}")
print(f"Current dir: {os.getcwd()}")

try:
    from nodes.custom_garmentfit_v3 import CustomGarmentFitLoaderV3, ApplyCustomGarmentFitV3
    print("✓ Nodes imported successfully!")
    
    # Check node structure
    print("\nNode 1: CustomGarmentFitLoaderV3")
    print(f"  - Category: {CustomGarmentFitLoaderV3.CATEGORY}")
    print(f"  - Function: {CustomGarmentFitLoaderV3.FUNCTION}")
    
    print("\nNode 2: ApplyCustomGarmentFitV3")
    print(f"  - Category: {ApplyCustomGarmentFitV3.CATEGORY}")
    print(f"  - Function: {ApplyCustomGarmentFitV3.FUNCTION}")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Test __init__.py
print("\nTesting __init__.py...")
try:
    from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"✓ Found {len(NODE_CLASS_MAPPINGS)} nodes")
    for name, display in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"  - {name}: {display}")
except Exception as e:
    print(f"✗ __init__.py error: {e}")