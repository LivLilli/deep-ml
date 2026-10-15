import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)

	# invertible: det != 0
	if not (torch.det(T_t) != 0 or torch.det(S_t) != 0):
		return -1
	
	return torch.matmul(torch.matmul(torch.inverse(T_t), A_t), S_t)
    
