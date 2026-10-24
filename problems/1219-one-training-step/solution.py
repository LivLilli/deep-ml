import torch

def train_step(model, x, y, optimizer, loss_fn):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # TODO: zero_grad -> forward -> loss -> backward -> step; return float loss
    optimizer.zero_grad() # re-initialize the gradients to zero
    frw = model(x) # forward pass
    loss = loss_fn(frw, y) # compute the loss   
    loss.backward() # compute gradients
    optimizer.step() #update weights and bias
    return loss.item() # return the loss as a float
