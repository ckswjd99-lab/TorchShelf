import torch
from torch.nn.utils import prune

def undo_pruning(model):
    for module in model.modules():
        if prune.is_pruned(module) and hasattr(module, 'weight'):
            module.weight_mask = torch.ones_like(module.weight)
            prune.remove(module, 'weight')

def get_layer_sparsity(model):
    sum_list_all = 0
    zero_sum_all = 0
    sparsity_ckpt = {}

    for name, m in model.named_modules():

        if prune.is_pruned(m) and hasattr(m, 'weight'):
            sum_list_all = sum_list_all + float(m.weight.nelement())
            zero_sum_all = zero_sum_all + float(torch.sum(m.weight == 0))

            sum_list = float(m.weight.nelement())
            zero_sum = float(torch.sum(m.weight == 0))

            layer_sparsity_rate = zero_sum/sum_list
            sparsity_ckpt[name] = layer_sparsity_rate

    return sparsity_ckpt
