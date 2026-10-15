import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=float)
    b_t = torch.as_tensor(b, dtype=float)
    x = torch.zeros(A_t.size(1), dtype=float)
    
    for time in range(n):
        x_new = torch.zeros_like(x)
        print(x_new)
        for i in range(A_t.size(1)):
            el1 = 1/A_t[i,i]
            el2 = 0
            for j in range(A_t.size(1)):
                if i != j:
                    el2 += A_t[i,j]*x[j]
            x_new[i] = el1 * (b[i] - el2)
        x = x_new 
    return torch.round(x, decimals=4)
