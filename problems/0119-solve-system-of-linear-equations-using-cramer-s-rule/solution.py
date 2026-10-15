import torch

def cramers_rule(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve system of linear equations Ax = b using Cramer's Rule.
    Returns solution vector x or -1 if no unique solution exists.
    """
    A_t = torch.as_tensor(A)
    b_t = torch.as_tensor(b)
    det_A = torch.det(A_t)
    
    if det_A == 0:
        return -1
    
    x = torch.zeros(A_t.size(0))
    for i in range(A_t.size(0)):
        A_new = torch.clone(A_t)
        A_new[:,i] = b_t
        x[i] = torch.det(A_new)/det_A
    return x
    