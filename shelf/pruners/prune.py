import torch
from torch.nn.utils import prune
from torch.nn.utils.prune import global_unstructured as torch_global_unstructured


def global_unstructured_L1(model, importance_scores, prune_ratio):
    modules = tuple( (module, 'weight') for module in model.modules() if hasattr(module, 'weight') )
    pruning_method = prune.L1Unstructured
    importance_scores = { (module, 'weight'): score for module, score in zip(modules, importance_scores.items()) if hasattr(module, 'weight')}

    torch_global_unstructured(modules, pruning_method, importance_scores, amount=prune_ratio)


def layerwise_unstructured_L1(model, importance_scores, prune_ratios):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_ratios[name], importance_scores=importance_scores)