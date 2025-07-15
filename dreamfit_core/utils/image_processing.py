"""
Image processing utilities for DreamFit
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional


def preprocess_garment_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    target_size: int = 224,
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Preprocess garment image for Anything-Dressing Encoder
    
    Args:
        image: Input image (tensor, numpy array, or PIL Image)
        target_size: Target size for the image (default: 224)
        normalize: Whether to normalize the image
        device: Device to place the tensor on
        
    Returns:
        Preprocessed image tensor [B, C, H, W]
    """
    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    if isinstance(image, np.ndarray):
        # Handle different input formats
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # Add batch dimension
        
        # Convert from HWC to CHW if needed
        if image.shape[-1] == 3:
            image = image.transpose(0, 3, 1, 2)
        
        # Convert to float tensor
        image = torch.from_numpy(image).float()
        
        # Normalize to [0, 1] if needed
        if image.max() > 1:
            image = image / 255.0
    
    elif isinstance(image, torch.Tensor):
        # Ensure 4D tensor
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # Ensure float
        image = image.float()
        
        # Normalize to [0, 1] if needed
        if image.max() > 1:
            image = image / 255.0
    
    # Resize to target size
    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(
            image,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    # Normalize using ImageNet statistics
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if device:
            mean = mean.to(device)
            std = std.to(device)
            
        image = (image - mean) / std
    
    # Move to device if specified
    if device:
        image = image.to(device)
    
    return image


def preprocess_pose_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    target_size: int = 224,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Preprocess pose/model image for DreamFit
    
    Args:
        image: Input pose image
        target_size: Target size for the image
        device: Device to place the tensor on
        
    Returns:
        Preprocessed pose image tensor [B, C, H, W]
    """
    # Use same preprocessing as garment images
    return preprocess_garment_image(image, target_size, normalize=True, device=device)


def extract_garment_mask(
    image: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.9,
    erode_dilate: bool = True
) -> torch.Tensor:
    """
    Extract a simple mask for the garment region
    
    Args:
        image: Input garment image
        threshold: Threshold for background detection
        erode_dilate: Whether to apply morphological operations
        
    Returns:
        Binary mask tensor [B, 1, H, W]
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    # Convert to grayscale
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    else:
        gray = image.mean(dim=1)
    
    # Create mask (assuming white/light background)
    mask = gray < threshold
    mask = mask.unsqueeze(1).float()
    
    if erode_dilate:
        # Simple morphological operations
        kernel_size = 3
        padding = kernel_size // 2
        
        # Create a simple averaging kernel for dilation/erosion
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        
        # Erode
        mask = F.conv2d(mask, kernel, padding=padding)
        mask = (mask > 0.5).float()
        
        # Dilate
        mask = F.conv2d(mask, kernel, padding=padding)
        mask = (mask > 0.3).float()
    
    return mask


def resize_and_pad_image(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    pad_value: float = 1.0
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Resize image maintaining aspect ratio and pad to target size
    
    Args:
        image: Input image tensor [B, C, H, W]
        target_size: Target (height, width)
        pad_value: Value to use for padding
        
    Returns:
        Tuple of (padded_image, padding_info)
    """
    _, _, h, w = image.shape
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = F.interpolate(
        image,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    
    # Pad image
    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=pad_value
    )
    
    padding_info = (pad_left, pad_right, pad_top, pad_bottom)
    
    return padded, padding_info


def composite_garment_on_background(
    garment: torch.Tensor,
    background: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Composite garment image on a background
    
    Args:
        garment: Garment image tensor [B, C, H, W]
        background: Background image tensor (white if None)
        mask: Garment mask (will be computed if None)
        
    Returns:
        Composited image tensor
    """
    if background is None:
        # Create white background
        background = torch.ones_like(garment)
    
    if mask is None:
        # Extract mask from garment
        mask = extract_garment_mask(garment)
    
    # Ensure mask is properly shaped
    if mask.shape[1] == 1 and garment.shape[1] == 3:
        mask = mask.repeat(1, 3, 1, 1)
    
    # Composite
    result = garment * mask + background * (1 - mask)
    
    return result