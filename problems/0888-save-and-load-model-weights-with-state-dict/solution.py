import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    buffer = io.BytesIO() # save to buffer, not on disk
    torch.save(src.state_dict(), buffer)
    buffer.seek(0) # reset buffer position to the beginning
    dst.load_state_dict(torch.load(buffer, weights_only=True)) # load weights from buffer and apply to dst