import torch

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == targets).float() / len(targets)

def compute_owcls(A_t: float, A_zs: float, BWT: float) -> float:
    return 0.5 * A_t + 0.3 * A_zs + 0.2 * BWT
