import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: implement y = x W^T + b using PyTorch ops
    mlt = x @ W.T # (N,in_f) @ (in_f,out_f) = (N,out_f)
    return  mlt + b.unsqueeze(0)
