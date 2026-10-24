import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    grad_input = grad_output @ W
    grad_w = grad_output.T @ x
    grad_b = grad_output.T @ torch.ones(x.shape[0],1)

    return grad_input, grad_w, grad_b.squeeze(1)

