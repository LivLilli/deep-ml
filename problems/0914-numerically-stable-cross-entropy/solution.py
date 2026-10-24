import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    mx = logits.max(dim=-1, keepdim=True).values
    lg = torch.logsumexp(logits -mx, dim=-1, keepdim=True)
    targ_onehot =torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
    logit_term = torch.sum(logits * targ_onehot, dim=-1, keepdim=True)
    # return torch.round(torch.mean(-logit_term + mx + lg), decimals=4)
    return torch.mean(-logit_term + mx + lg)


