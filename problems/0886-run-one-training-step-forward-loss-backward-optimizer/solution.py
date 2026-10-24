import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    sgd = torch.optim.SGD(model.parameters(), lr=lr)
    sgd.zero_grad()
    frwd = model(x)
    # loss = (frwd - y)**2 # mean squared error loss
    loss = torch.nn.functional.mse_loss(frwd, y)
    loss.backward()
    sgd.step()
    return loss.item()
