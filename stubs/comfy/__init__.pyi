# Type stubs for ComfyUI
from typing import Any, Dict, List, Tuple, Optional, Union
import torch

class sample:
    @staticmethod
    def fix_empty_latent_channels(model: Any, latent: torch.Tensor) -> torch.Tensor: ...
    
    @staticmethod
    def prepare_noise(latent: torch.Tensor, seed: int, batch_inds: Optional[List[int]] = None) -> torch.Tensor: ...
    
    @staticmethod
    def sample(
        model: Any,
        noise: torch.Tensor,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: List[Tuple[torch.Tensor, Dict[str, Any]]],
        negative: List[Tuple[torch.Tensor, Dict[str, Any]]],
        latent_image: torch.Tensor,
        denoise: float = 1.0,
        seed: int = 0,
        noise_mask: Optional[torch.Tensor] = None,
        callback: Optional[Any] = None,
        disable_pbar: bool = False,
        **kwargs
    ) -> torch.Tensor: ...

class samplers:
    class KSampler:
        SAMPLERS: List[str]
        SCHEDULERS: List[str]

class model_management:
    @staticmethod
    def get_torch_device() -> torch.device: ...
    
    @staticmethod
    def intermediate_device() -> torch.device: ...
    
    @staticmethod
    def unet_dtype() -> torch.dtype: ...