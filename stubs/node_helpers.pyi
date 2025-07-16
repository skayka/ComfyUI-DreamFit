# Type stubs for node_helpers
from typing import Any, Dict, List, Tuple
import torch

def conditioning_set_values(
    conditioning: List[Tuple[torch.Tensor, Dict[str, Any]]],
    values: Dict[str, Any]
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]: ...