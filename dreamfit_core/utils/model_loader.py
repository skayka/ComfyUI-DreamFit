"""
Model loading and management utilities for DreamFit
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union
import torch
import json

# Try to import ComfyUI's folder_paths
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False
    folder_paths = None


class DreamFitModelManager:
    """
    Manages DreamFit model downloading, caching, and loading
    """
    
    MODEL_INFO = {
        "flux_i2i": {
            "url": "https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_i2i.bin",
            "size_mb": 284,
            "sha256": None,  # Will be computed on first download
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
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self._load_metadata()
    
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory for models"""
        # Try ComfyUI models directory first
        try:
            import folder_paths
            return Path(folder_paths.models_dir) / "dreamfit"
        except ImportError:
            # Fallback to user cache directory
            return Path.home() / ".cache" / "dreamfit"
    
    def _load_metadata(self):
        """Load model metadata from cache"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save model metadata to cache"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_model(self, model_name: str, target_dir: Optional[str] = None) -> Path:
        """
        Download a DreamFit model
        
        Args:
            model_name: Name of the model to download
            target_dir: Target directory (uses cache_dir if None)
            
        Returns:
            Path to the downloaded model file
        """
        if model_name not in self.MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_INFO.keys())}")
        
        model_info = self.MODEL_INFO[model_name]
        target_dir = Path(target_dir) if target_dir else self.cache_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{model_name}.bin"
        filepath = target_dir / filename
        
        # Check if already downloaded
        if filepath.exists():
            print(f"Model {model_name} already exists at {filepath}")
            return filepath
        
        # Download with progress bar
        print(f"Downloading {model_name} ({model_info['size_mb']}MB)...")
        try:
            import requests
        except ImportError:
            raise ImportError("requests library is required for downloading models. Install with: pip install requests")
        
        response = requests.get(model_info['url'], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filepath, 'wb') as f:
            try:
                from tqdm import tqdm
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            except ImportError:
                # Fallback without progress bar
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
        
        # Compute and store checksum
        sha256 = self._compute_sha256(filepath)
        self.metadata[model_name] = {
            "path": str(filepath),
            "sha256": sha256,
            "size": os.path.getsize(filepath)
        }
        self._save_metadata()
        
        print(f"✓ Downloaded {model_name} successfully")
        return filepath
    
    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def verify_model(self, model_name: str, filepath: Path) -> bool:
        """
        Verify integrity of a downloaded model
        
        Args:
            model_name: Name of the model
            filepath: Path to the model file
            
        Returns:
            True if model is valid, False otherwise
        """
        if not filepath.exists():
            return False
        
        # Check file size
        expected_size = self.MODEL_INFO[model_name]['size_mb'] * 1024 * 1024
        actual_size = os.path.getsize(filepath)
        
        # Allow 5% tolerance for size
        if abs(actual_size - expected_size) / expected_size > 0.05:
            print(f"Warning: File size mismatch for {model_name}")
            return False
        
        return True
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get path to a model file
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model file if it exists, None otherwise
        """
        # Check metadata first
        if model_name in self.metadata:
            path = Path(self.metadata[model_name]['path'])
            if path.exists():
                return path
        
        # Check default locations
        search_dirs = [self.cache_dir]
        if HAS_FOLDER_PATHS and folder_paths is not None:
            search_dirs.append(Path(folder_paths.models_dir) / "dreamfit")
        
        for dir_path in search_dirs:
            filepath = dir_path / f"{model_name}.bin"
            if filepath.exists():
                return filepath
        
        return None
    
    def list_downloaded_models(self) -> Dict[str, Dict]:
        """
        List all downloaded models with their info
        
        Returns:
            Dictionary of model info
        """
        downloaded = {}
        for model_name in self.MODEL_INFO:
            path = self.get_model_path(model_name)
            if path:
                downloaded[model_name] = {
                    "path": str(path),
                    "size_mb": os.path.getsize(path) / (1024 * 1024),
                    "verified": self.verify_model(model_name, path)
                }
        return downloaded
    
    def clean_cache(self, keep_models: Optional[list] = None):
        """
        Clean model cache, optionally keeping specific models
        
        Args:
            keep_models: List of model names to keep
        """
        keep_models = keep_models or []
        
        for filepath in self.cache_dir.glob("*.bin"):
            model_name = filepath.stem
            if model_name not in keep_models and model_name in self.MODEL_INFO:
                print(f"Removing {model_name}...")
                filepath.unlink()
                if model_name in self.metadata:
                    del self.metadata[model_name]
        
        self._save_metadata()
        print("✓ Cache cleaned")


# Convenience functions
def download_all_dreamfit_models(target_dir: Optional[str] = None):
    """Download all DreamFit models"""
    manager = DreamFitModelManager()
    for model_name in manager.MODEL_INFO:
        manager.download_model(model_name, target_dir)


def verify_dreamfit_installation() -> bool:
    """Verify that all DreamFit models are properly installed"""
    manager = DreamFitModelManager()
    all_valid = True
    
    for model_name in manager.MODEL_INFO:
        path = manager.get_model_path(model_name)
        if path and manager.verify_model(model_name, path):
            print(f"✓ {model_name}: OK")
        else:
            print(f"✗ {model_name}: Missing or invalid")
            all_valid = False
    
    return all_valid