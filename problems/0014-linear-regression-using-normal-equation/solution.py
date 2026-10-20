import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    
    # X_t_transpose 
    X_trans = torch.transpose(X_t, dim0=0, dim1=1)
    
    # mat_mul Xt X
    X_mul = torch.matmul(X_trans, X_t)
    
    # inverse
    X_inv = torch.linalg.inv(X_mul)
    
    # mat_mul inv trans
    X_mul2 = torch.matmul(X_inv, X_trans)
    
    # mat_mul2 and vect
    
    theta = torch.matmul(X_mul2, y_t)
    
    return theta
