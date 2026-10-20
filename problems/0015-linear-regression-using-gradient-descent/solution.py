import torch

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros(n, 1,)
    
    for i in range(iterations):
        
        X_trans = torch.transpose(X_t, 0, 1)
        preds = torch.matmul(X_t, theta)
        error = preds - y_t
        gradient = (1/m)*torch.matmul(X_trans, error)
        theta = theta - (alpha * gradient)
        

    return torch.flatten(theta)
