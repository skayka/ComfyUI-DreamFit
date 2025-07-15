#!/usr/bin/env python3
"""
Installation script for ComfyUI-DreamFit
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_comfyui():
    """Check if we're in a ComfyUI custom_nodes directory"""
    current_dir = Path.cwd()
    if current_dir.name != "ComfyUI-DreamFit":
        print("❌ This script should be run from the ComfyUI-DreamFit directory")
        return False
    
    parent_dir = current_dir.parent
    if parent_dir.name != "custom_nodes":
        print("❌ ComfyUI-DreamFit should be in ComfyUI/custom_nodes directory")
        return False
    
    return True


def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Python requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def download_models(model_names=None):
    """Download DreamFit models"""
    print("\n📥 Downloading DreamFit models...")
    
    from dreamfit_core.utils.model_loader import DreamFitModelManager
    
    manager = DreamFitModelManager()
    
    if model_names is None:
        # Download all models
        model_names = list(manager.MODEL_INFO.keys())
    
    for model_name in model_names:
        print(f"\nDownloading {model_name}...")
        try:
            path = manager.download_model(model_name)
            print(f"✅ Downloaded {model_name} to {path}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")


def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    # Check if all modules can be imported
    try:
        import nodes
        from dreamfit_core import models, utils
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check models
    try:
        from dreamfit_core.utils.model_loader import verify_dreamfit_installation
        if verify_dreamfit_installation():
            print("✅ All models verified")
        else:
            print("⚠️  Some models are missing (run with --download-models to download)")
    except Exception as e:
        print(f"⚠️  Could not verify models: {e}")
    
    print("\n✨ Installation complete!")
    print("Restart ComfyUI to use DreamFit nodes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install ComfyUI-DreamFit")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download DreamFit models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["flux_i2i", "flux_i2i_with_pose", "flux_tryon"],
        help="Specific models to download"
    )
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Skip installing Python requirements"
    )
    
    args = parser.parse_args()
    
    print("🚀 ComfyUI-DreamFit Installation")
    print("================================\n")
    
    # Check environment
    if not check_comfyui():
        sys.exit(1)
    
    # Install requirements
    if not args.skip_requirements:
        if not install_requirements():
            sys.exit(1)
    
    # Download models if requested
    if args.download_models:
        download_models(args.models)
    
    # Verify installation
    verify_installation()


if __name__ == "__main__":
    main()