import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)    

    if a_t.size(1) != b_t.size(0):
        return -1

    val = torch.zeros(a_t.size(0), b_t.size(1))

    for r in range(a_t.size(0)):
        for c in range(b_t.size(1)):
            for i in range(a_t.size(1)):
                val[r, c] += a_t[r, :][i]*b_t[:, c][i]   
    return val
