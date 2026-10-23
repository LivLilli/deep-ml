import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    # Your implementation here
	X = torch.as_tensor(vectors, dtype=float)
	n = len(vectors[0])
	mean = torch.mean(X, 1).reshape(len(X),1)
	diff = X-mean
	return torch.matmul(diff, torch.transpose(diff, 0, 1))*(1/(n-1))
