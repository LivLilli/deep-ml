import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    X = torch.as_tensor(data, dtype=torch.float)
    m = torch.mean(X, 0)
    s = torch.std(X, 0, correction=0)
    mn = torch.min(X, 0).values
    mx = torch.max(X, 0).values
    standardized_data = (X-m)/(s)
    normalized_data = (X-mn)/(mx-mn)
    return torch.round(standardized_data, decimals=4), normalized_data
    #return torch.round(standardized_data, decimals = 4), normalized_data
