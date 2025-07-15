#!/usr/bin/env python3
"""
Standalone script to download DreamFit models from HuggingFace
Can be run independently of ComfyUI installation
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
import json


class StandaloneDreamFitDownloader:
    """Standalone model downloader for DreamFit"""
    
    MODEL_INFO = {
        "flux_i2i": {
            "url": "https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_i2i.bin",
            "size_mb": 284,
            "sha256": None,
            "description": "Basic garment-centric generation for Flux"
        },
        "flux_i2i_with_pose": {
            "url": "https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_i2i_with_pose.bin",
            "size_mb": 284,
            "sha256": None,
            "description": "Garment generation with pose control for Flux"
        },
        "flux_tryon": {
            "url": "https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_tryon.bin",
            "size_mb": 287,
            "sha256": None,
            "description": "Virtual try-on mode for Flux"
        }
    }
    
    def __init__(self, models_dir=None, comfyui_dir=None):
        """
        Initialize downloader
        
        Args:
            models_dir: Direct path to download models to
            comfyui_dir: Path to ComfyUI installation (will use ComfyUI/models/dreamfit)
        """
        self.models_dir = self._determine_models_dir(models_dir, comfyui_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "model_metadata.json"
        self._load_metadata()
        
    def _determine_models_dir(self, models_dir, comfyui_dir):
        """Determine where to download models"""
        if models_dir:
            return Path(models_dir)
        
        if comfyui_dir:
            return Path(comfyui_dir) / "models" / "dreamfit"
        
        # Try to detect ComfyUI installation
        # Check parent directories
        current = Path.cwd()
        for _ in range(3):  # Check up to 3 levels up
            if (current / "ComfyUI" / "models").exists():
                print(f"✓ Found ComfyUI at: {current / 'ComfyUI'}")
                return current / "ComfyUI" / "models" / "dreamfit"
            elif (current / "models").exists() and (current / "comfy").exists():
                # We're inside ComfyUI directory
                print(f"✓ Found ComfyUI at: {current}")
                return current / "models" / "dreamfit"
            current = current.parent
        
        # Default to local dreamfit_models directory
        print("⚠️  ComfyUI not found. Using local directory: ./dreamfit_models")
        return Path.cwd() / "dreamfit_models"
    
    def _load_metadata(self):
        """Load model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_model(self, model_name, force=False):
        """
        Download a specific model
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if file exists
        """
        if model_name not in self.MODEL_INFO:
            print(f"❌ Unknown model: {model_name}")
            print(f"Available models: {', '.join(self.MODEL_INFO.keys())}")
            return False
        
        model_info = self.MODEL_INFO[model_name]
        filepath = self.models_dir / f"{model_name}.bin"
        
        # Check if already exists
        if filepath.exists() and not force:
            print(f"✓ {model_name} already exists at {filepath}")
            if self._verify_file(filepath, model_info['size_mb']):
                return True
            else:
                print(f"⚠️  File size mismatch, re-downloading...")
        
        # Download
        print(f"\n📥 Downloading {model_name} ({model_info['size_mb']}MB)")
        print(f"   {model_info['description']}")
        print(f"   From: {model_info['url']}")
        print(f"   To: {filepath}")
        
        try:
            response = requests.get(model_info['url'], stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Compute checksum
            sha256 = self._compute_sha256(filepath)
            self.metadata[model_name] = {
                "path": str(filepath),
                "sha256": sha256,
                "size": os.path.getsize(filepath)
            }
            self._save_metadata()
            
            print(f"✓ Downloaded {model_name} successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Download interrupted")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False
    
    def _verify_file(self, filepath, expected_size_mb):
        """Verify file size is within acceptable range"""
        actual_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        # Allow 5% tolerance
        return abs(actual_size_mb - expected_size_mb) / expected_size_mb < 0.05
    
    def _compute_sha256(self, filepath):
        """Compute SHA256 checksum"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def download_all(self, force=False):
        """Download all models"""
        print(f"\n🚀 Downloading all DreamFit models to: {self.models_dir}\n")
        
        success_count = 0
        for model_name in self.MODEL_INFO:
            if self.download_model(model_name, force):
                success_count += 1
            print()  # Empty line between models
        
        print(f"\n✨ Downloaded {success_count}/{len(self.MODEL_INFO)} models successfully!")
        if success_count == len(self.MODEL_INFO):
            print(f"✓ All models ready at: {self.models_dir}")
        
        return success_count == len(self.MODEL_INFO)
    
    def list_models(self):
        """List all models and their status"""
        print("\n📋 DreamFit Models Status:\n")
        print(f"{'Model':<25} {'Size':<10} {'Status':<15} {'Description'}")
        print("-" * 80)
        
        for model_name, info in self.MODEL_INFO.items():
            filepath = self.models_dir / f"{model_name}.bin"
            if filepath.exists():
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                status = f"✓ Downloaded ({size_mb:.1f}MB)"
            else:
                status = "⬇ Not downloaded"
            
            print(f"{model_name:<25} {info['size_mb']}MB{'':<5} {status:<15} {info['description']}")
        
        print(f"\nModels directory: {self.models_dir}")
    
    def verify_all(self):
        """Verify all downloaded models"""
        print("\n🔍 Verifying downloaded models...\n")
        
        all_valid = True
        for model_name in self.MODEL_INFO:
            filepath = self.models_dir / f"{model_name}.bin"
            if filepath.exists():
                if self._verify_file(filepath, self.MODEL_INFO[model_name]['size_mb']):
                    print(f"✓ {model_name}: Valid")
                else:
                    print(f"❌ {model_name}: Invalid size")
                    all_valid = False
            else:
                print(f"⬇ {model_name}: Not downloaded")
                all_valid = False
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Download DreamFit models for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models (auto-detect ComfyUI)
  python download_models.py
  
  # Download specific model
  python download_models.py --model flux_i2i
  
  # Download to specific directory
  python download_models.py --models-dir /path/to/models
  
  # Specify ComfyUI location
  python download_models.py --comfyui-dir /path/to/ComfyUI
  
  # List models without downloading
  python download_models.py --list
  
  # Force re-download
  python download_models.py --force
"""
    )
    
    parser.add_argument(
        "--model",
        choices=["flux_i2i", "flux_i2i_with_pose", "flux_tryon"],
        help="Download specific model (default: all)"
    )
    parser.add_argument(
        "--models-dir",
        help="Directory to download models to"
    )
    parser.add_argument(
        "--comfyui-dir",
        help="Path to ComfyUI installation"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List models and their status"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = StandaloneDreamFitDownloader(
        models_dir=args.models_dir,
        comfyui_dir=args.comfyui_dir
    )
    
    # Handle commands
    if args.list:
        downloader.list_models()
    elif args.verify:
        if downloader.verify_all():
            print("\n✓ All models verified successfully!")
        else:
            print("\n⚠️  Some models are missing or invalid")
            sys.exit(1)
    elif args.model:
        # Download specific model
        if not downloader.download_model(args.model, args.force):
            sys.exit(1)
    else:
        # Download all models
        if not downloader.download_all(args.force):
            sys.exit(1)


if __name__ == "__main__":
    main()