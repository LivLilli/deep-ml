import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    mx = torch.maximum(logits, torch.zeros_like(logits))
    prd = logits * targets
    lg = torch.log(1+torch.exp(-torch.abs(logits)))
    return torch.round(torch.mean(mx - prd + lg), decimals=4).item()
