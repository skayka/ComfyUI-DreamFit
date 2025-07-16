#!/usr/bin/env python3
"""Test if the DreamFit node can be imported"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing import of __init__.py...")
    from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"Successfully imported: {NODE_CLASS_MAPPINGS}")
    print(f"Display names: {NODE_DISPLAY_NAME_MAPPINGS}")
except Exception as e:
    print(f"Error importing from __init__.py: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting direct import of dreamfit_sampler...")
try:
    from nodes.dreamfit_sampler import DreamFitSampler
    print(f"Successfully imported DreamFitSampler: {DreamFitSampler}")
except Exception as e:
    print(f"Error importing dreamfit_sampler: {e}")
    import traceback
    traceback.print_exc()