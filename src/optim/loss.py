import torch


def loss_func_with_clutter(obj_s, clu_s, device):
    return torch.ones(1, device=device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))
