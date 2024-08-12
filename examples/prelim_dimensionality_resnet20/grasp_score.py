import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd.functional import hvp

import math, os

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer
from shelf.pruners.scoring import get_grasp_score, get_hgp_score

DEVICE = 'cuda'
SMOOTHING = 1e-3
EPOCH = 3

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)


model_init_path = './saves/set1_default/initial_weights.pth'
target_model_paths = [
    './saves/set1_default/epoch_1.pth',
    './saves/set1_default/epoch_2.pth',
    './saves/set1_default/epoch_3.pth',
    './saves/set1_default/epoch_4.pth',
    './saves/set1_default/epoch_5.pth',
    './saves/set1_default/epoch_6.pth',
    './saves/set1_default/epoch_7.pth',
    './saves/set1_default/epoch_8.pth',
    './saves/set1_default/epoch_9.pth',
    './saves/set1_default/epoch_10.pth',
    './saves/set1_default/epoch_20.pth',
    './saves/set1_default/epoch_30.pth',
    './saves/set1_default/epoch_40.pth',
    './saves/set1_default/epoch_50.pth',
    './saves/set1_default/epoch_60.pth',
    './saves/set1_default/epoch_70.pth',
    './saves/set1_default/epoch_80.pth',
    './saves/set1_default/epoch_90.pth',
    './saves/set1_default/epoch_100.pth',
]


model_temp = MLPMixer().to(DEVICE)
model_init = MLPMixer().to(DEVICE)
model_init.load_state_dict(torch.load(model_init_path))
model_init_weights = {pname: param.clone() for pname, param in model_init.state_dict().items()}

num_params = sum(p.numel() for p in model_init.parameters())
print(f"Num params: {num_params}")

criterion = nn.CrossEntropyLoss().to(DEVICE)

## Calculate Hg value ##
hg_value_sum = {pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()}
for input, label in tqdm(train_loader, leave=False):
    input, label = input.to(DEVICE), label.to(DEVICE)

    hg_value = get_hgp_score(input, label, model_init)
    for pname, value in zip(model_init.state_dict().keys(), hg_value):
        hg_value_sum[pname] += value

for pname in model_init.state_dict().keys():
    hg_value_sum[pname] /= len(train_loader)

hg_value_flat = torch.cat([param.view(-1) for param in hg_value_sum.values()])
hg_value_norm = torch.norm(hg_value_flat, p=2)

## Calculate GraSP score of each parameter ##
parameter_flat = torch.cat([param.view(-1) for param in model_init.parameters()])
parameter_norm = torch.norm(parameter_flat, p=2)

grasp_score = -parameter_flat * hg_value_flat
grasp_norm = torch.norm(grasp_score, p=2)

print(f"PD-GraSP Min: {grasp_score.min().item()}, Max: {grasp_score.max().item()}")

## Calculate GraSP score of ideal direction that maximizes GraSP score ##
idim1_flat = parameter_norm * hg_value_flat.clone() + hg_value_norm * parameter_flat.clone()
idim1_norm = torch.norm(idim1_flat, p=2)
idim1_flat /= idim1_norm
param_idim1_dot = torch.dot(parameter_flat, idim1_flat)

ideal_grasp_score1 = -torch.dot(param_idim1_dot * idim1_flat, hg_value_flat)
print(f"Ideal1 PD-GraSP Score: {ideal_grasp_score1}")

idim2_flat = - parameter_norm * hg_value_flat.clone() + hg_value_norm * parameter_flat.clone()
idim2_norm = torch.norm(idim2_flat, p=2)
idim2_flat /= idim2_norm
param_idim2_dot = torch.dot(parameter_flat, idim2_flat)

ideal_grasp_score2 = -torch.dot(param_idim2_dot * idim2_flat, hg_value_flat)
print(f"Ideal1 PD-GraSP Score: {ideal_grasp_score2}")

## Calculate GraSP score of the ultimate directions ##
for target_model_path in target_model_paths:
    model_name_short = target_model_path.split('/')[-1][:-4]
    print(f"Processing {model_name_short}")

    model_temp.load_state_dict(torch.load(target_model_path))

    model_temp_state_dict = dict(model_temp.state_dict())

    model_temp_weights = {pname: param.clone() for pname, param in model_temp_state_dict.items()}
    # weights_difference = {pname: model_temp_weights[pname] for pname in model_temp_weights}
    weights_difference = {pname: model_temp_weights[pname] - model_init_weights[pname] for pname in model_init_weights}
    # weights_difference = {pname: torch.randn_like(model2_weights[pname]) for pname in model1_weights}
    weights_difference_flat = torch.cat([param.view(-1) for param in weights_difference.values()])
    wdiff_norm = torch.norm(weights_difference_flat, p=2)
    if wdiff_norm == 0:
        print(f"\tWeight Difference Norm is 0. Skipping {model_name_short}")
        continue
    num_params = sum(p.numel() for p in model_init.parameters())

    opt_dim_dict = {pname: diff / wdiff_norm for pname, diff in weights_difference.items()}
    opt_dim_dict_flat = torch.cat([param.view(-1) for param in opt_dim_dict.values()])

    opt_dim_dict_norm = torch.norm(opt_dim_dict_flat, p=2)
    print(f"\tOptimal Direction Norm: {opt_dim_dict_norm}")

    test_pnoises = {pname: SMOOTHING * torch.randn_like(param) for pname, param in model_init.named_parameters()}
    using_pnoises = {pname: SMOOTHING * math.sqrt(num_params) * opt_dim_dict[pname] for pname, param in model_init.named_parameters()}

    param_dim_dot = torch.dot(parameter_flat, opt_dim_dict_flat)
    hg_dim_dot = torch.dot(hg_value_flat, opt_dim_dict_flat)
    grasp_dim_dot = torch.dot(grasp_score, opt_dim_dict_flat)
    idim1_dim_dot = torch.dot(idim1_flat, opt_dim_dict_flat)
    idim2_dim_dot = torch.dot(idim2_flat, opt_dim_dict_flat)
    print(f"\tParam-Direction Dot: {param_dim_dot}, Param-Direction CosSim: {param_dim_dot / parameter_norm}")
    print(f"\tHg-Direction Dot: {hg_dim_dot}, Hg-Direction CosSim: {hg_dim_dot / hg_value_norm}")
    print(f"\tGrasp-Direction Dot: {grasp_dim_dot}, Grasp-Direction CosSim: {grasp_dim_dot / grasp_norm}")
    print(f"\tIdeal1-Direction Dot: {idim1_dim_dot}, Ideal1-Direction CosSim: {idim1_dim_dot / idim1_norm}")
    print(f"\tIdeal2-Direction Dot: {idim2_dim_dot}, Ideal2-Direction CosSim: {idim2_dim_dot / idim1_norm}")

    pnoises_flat = torch.cat([param.view(-1) for param in test_pnoises.values()])
    using_pnoises_flat = torch.cat([param.view(-1) for param in using_pnoises.values()])

    ultimate_direction_score = -torch.dot(param_dim_dot * opt_dim_dict_flat, hg_value_flat)
    print(f"\tGraSP Score of the Selected Direction: {ultimate_direction_score}")