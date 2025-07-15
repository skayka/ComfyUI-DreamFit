"""
Debug visualization utilities for DreamFit nodes
Provides functions for creating debug outputs and visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any


def create_debug_grid(
    images: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    grid_size: Tuple[int, int] = (2, 2),
    padding: int = 2,
    background_color: float = 0.0
) -> torch.Tensor:
    """
    Create a grid of images for debug visualization
    
    Args:
        images: List of image tensors [C, H, W] or [B, C, H, W]
        titles: Optional titles for each image (not rendered, for reference)
        grid_size: (rows, cols) for the grid layout
        padding: Padding between images in pixels
        background_color: Background color value (0-1)
    
    Returns:
        Grid tensor in ComfyUI format [B, H, W, C]
    """
    if not images:
        raise ValueError("No images provided for grid")
    
    # Ensure all images are 3D tensors [C, H, W]
    processed_images = []
    for img in images:
        if img.dim() == 4:
            img = img[0]  # Take first batch item
        if img.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {img.dim()}D")
        processed_images.append(img)
    
    # Get dimensions
    C, H, W = processed_images[0].shape
    rows, cols = grid_size
    
    # Create grid with padding
    grid_h = rows * H + (rows - 1) * padding
    grid_w = cols * W + (cols - 1) * padding
    
    grid = torch.full((C, grid_h, grid_w), background_color, dtype=processed_images[0].dtype, device=processed_images[0].device)
    
    # Place images in grid
    for idx, img in enumerate(processed_images):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        y_start = row * (H + padding)
        x_start = col * (W + padding)
        
        grid[:, y_start:y_start + H, x_start:x_start + W] = img
    
    # Convert to ComfyUI format [B, H, W, C]
    grid = grid.unsqueeze(0).permute(0, 2, 3, 1)
    
    return grid


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    target_size: Tuple[int, int],
    colormap: str = "hot",
    normalize: bool = True
) -> torch.Tensor:
    """
    Create a heatmap visualization from attention weights
    
    Args:
        attention_weights: Attention tensor [heads, seq_len, seq_len] or [seq_len, seq_len]
        target_size: (H, W) target size for the heatmap
        colormap: Color scheme ("hot", "cool", "viridis", "plasma")
        normalize: Whether to normalize values to [0, 1]
    
    Returns:
        RGB heatmap tensor [3, H, W]
    """
    H, W = target_size
    
    # Handle different attention weight formats
    if attention_weights.dim() == 3:
        # Average across heads
        attention_weights = attention_weights.mean(dim=0)
    
    if attention_weights.dim() != 2:
        raise ValueError(f"Expected 2D or 3D attention weights, got {attention_weights.dim()}D")
    
    # Extract spatial attention (assuming first token is CLS)
    if attention_weights.shape[0] > 1:
        # Get attention to first token (usually CLS or garment token)
        spatial_attn = attention_weights[1:, 0]
    else:
        spatial_attn = attention_weights.flatten()
    
    # Determine spatial dimensions
    num_patches = spatial_attn.shape[0]
    patch_size = int(np.sqrt(num_patches))
    
    if patch_size * patch_size != num_patches:
        # Fallback for non-square attention
        spatial_attn = spatial_attn[:patch_size * patch_size]
    
    # Reshape to 2D
    spatial_attn = spatial_attn.reshape(patch_size, patch_size)
    
    # Normalize if requested
    if normalize:
        spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)
    
    # Upsample to target size
    spatial_attn = spatial_attn.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(spatial_attn, size=(H, W), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze()
    
    # Apply colormap
    heatmap_rgb = apply_colormap(heatmap, colormap)
    
    return heatmap_rgb


def apply_colormap(tensor: torch.Tensor, colormap: str = "hot") -> torch.Tensor:
    """
    Apply a colormap to a single-channel tensor
    
    Args:
        tensor: Single-channel tensor [H, W] with values in [0, 1]
        colormap: Color scheme name
    
    Returns:
        RGB tensor [3, H, W]
    """
    H, W = tensor.shape
    device = tensor.device
    
    # Ensure values are in [0, 1]
    tensor = tensor.clamp(0, 1)
    
    if colormap == "hot":
        # Hot colormap: black -> red -> yellow -> white
        r = tensor.clamp(0, 1)
        g = (tensor - 0.5).clamp(0, 1) * 2
        b = (tensor - 0.75).clamp(0, 1) * 4
    
    elif colormap == "cool":
        # Cool colormap: cyan -> blue -> magenta
        r = tensor
        g = 1 - tensor
        b = 1.0
    
    elif colormap == "viridis":
        # Simplified viridis: dark purple -> blue -> green -> yellow
        r = tensor
        g = tensor ** 0.5
        b = (1 - tensor) ** 2
    
    elif colormap == "plasma":
        # Simplified plasma: dark blue -> purple -> pink -> yellow
        r = tensor ** 0.7
        g = tensor ** 1.5
        b = (1 - tensor) ** 0.5
    
    else:
        # Default grayscale
        r = g = b = tensor
    
    # Stack into RGB
    rgb = torch.stack([r, g, b], dim=0)
    
    return rgb


def overlay_attention_on_image(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "hot"
) -> torch.Tensor:
    """
    Overlay attention heatmap on an image
    
    Args:
        image: Image tensor [C, H, W] or [B, C, H, W]
        attention_weights: Attention weights
        alpha: Blending factor (0 = only image, 1 = only heatmap)
        colormap: Colormap for the heatmap
    
    Returns:
        Blended image tensor [C, H, W]
    """
    # Ensure image is [C, H, W]
    if image.dim() == 4:
        image = image[0]
    
    C, H, W = image.shape
    
    # Create heatmap
    heatmap = create_attention_heatmap(attention_weights, (H, W), colormap)
    
    # Blend
    blended = (1 - alpha) * image + alpha * heatmap
    
    return blended.clamp(0, 1)


def create_feature_visualization(
    features: Dict[str, torch.Tensor],
    feature_type: str = "patch",
    max_features: int = 16
) -> torch.Tensor:
    """
    Visualize extracted features as a grid
    
    Args:
        features: Dictionary containing feature tensors
        feature_type: Type of features to visualize ("patch", "pooled", "token")
        max_features: Maximum number of features to show
    
    Returns:
        Visualization grid [B, H, W, C]
    """
    if feature_type not in features:
        # Return a placeholder image
        placeholder = torch.ones(1, 224, 224, 3) * 0.5
        return placeholder
    
    feat = features[feature_type]
    
    if feature_type == "patch":
        # Visualize patch features as a grid
        if feat.dim() == 3:  # [B, num_patches, dim]
            feat = feat[0]  # Take first batch
        
        num_patches = feat.shape[0]
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size == num_patches:
            # Reshape to spatial grid
            feat = feat.reshape(patch_size, patch_size, -1)
            
            # Take first few channels
            feat = feat[..., :3].permute(2, 0, 1)
            
            # Normalize
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            
            # Resize to standard size
            feat = F.interpolate(feat.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            feat = feat[0]
        else:
            # Fallback visualization
            feat = torch.ones(3, 224, 224) * 0.5
    
    elif feature_type == "pooled" or feature_type == "token":
        # Visualize as color blocks
        if feat.dim() == 2:
            feat = feat[0]  # Take first batch
        
        # Create a color representation
        dim = min(feat.shape[0], max_features)
        colors = feat[:dim].reshape(-1, 1, 1)
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
        
        # Create grid of color blocks
        block_size = 224 // int(np.sqrt(dim))
        grid = torch.zeros(3, 224, 224)
        
        for i in range(dim):
            row = i // int(np.sqrt(dim))
            col = i % int(np.sqrt(dim))
            
            y_start = row * block_size
            x_start = col * block_size
            
            # Assign color based on feature value
            grid[0, y_start:y_start + block_size, x_start:x_start + block_size] = colors[i]
            grid[1, y_start:y_start + block_size, x_start:x_start + block_size] = colors[i] * 0.7
            grid[2, y_start:y_start + block_size, x_start:x_start + block_size] = colors[i] * 0.5
        
        feat = grid
    
    # Convert to ComfyUI format
    return feat.unsqueeze(0).permute(0, 2, 3, 1)


def create_comparison_grid(
    original: torch.Tensor,
    processed: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    attention: Optional[torch.Tensor] = None,
    labels: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Create a comparison grid showing original, processed, and optionally result + attention
    
    Args:
        original: Original input image
        processed: Processed/encoded image
        result: Optional generated result
        attention: Optional attention weights
        labels: Optional labels for images
    
    Returns:
        Comparison grid [B, H, W, C]
    """
    images = [original, processed]
    
    if result is not None:
        images.append(result)
    
    if attention is not None:
        # Create attention overlay
        attention_overlay = overlay_attention_on_image(original, attention, alpha=0.6)
        images.append(attention_overlay)
    
    # Determine grid size
    num_images = len(images)
    if num_images <= 2:
        grid_size = (1, 2)
    elif num_images <= 4:
        grid_size = (2, 2)
    else:
        grid_size = (2, 3)
    
    return create_debug_grid(images, labels, grid_size)


def add_text_overlay(
    image: torch.Tensor,
    text: str,
    position: Tuple[int, int] = (10, 10),
    font_size: int = 12,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    Add text overlay to an image (placeholder - actual text rendering would require PIL)
    
    For now, this just adds a colored box where text would be
    """
    if image.dim() == 4:
        image = image[0]
    
    # Clone to avoid modifying original
    image = image.clone()
    
    # Add a simple colored rectangle as placeholder for text
    x, y = position
    h, w = font_size, len(text) * font_size // 2
    
    C, H, W = image.shape
    
    # Ensure rectangle fits
    h = min(h, H - y)
    w = min(w, W - x)
    
    if h > 0 and w > 0:
        for c in range(C):
            image[c, y:y+h, x:x+w] = color[c] * 0.8
    
    return image