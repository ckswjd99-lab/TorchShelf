import torch
import torch.nn as nn
import hashlib

def model_hash(model: nn.Module):
    hash_str = ''
    for name, param in model.named_parameters():
        hash_str += str(param.data.sum().item())

    return hashlib.md5(hash_str.encode()).hexdigest()