import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    matrix_t = torch.as_tensor(matrix)
    trace = torch.trace(matrix_t)
	det = torch.det(matrix_t)
	delta = (trace**2) - 4*det

	e1 = (trace + torch.sqrt(delta))/2
	e2 = (trace - torch.sqrt(delta))/2

    eigenvalues, _ = torch.sort(torch.as_tensor([e1, e2]), descending=True)
    return eigenvalues
