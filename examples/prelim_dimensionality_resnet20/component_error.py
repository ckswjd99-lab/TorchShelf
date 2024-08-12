import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from tqdm import tqdm
import os

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.models.resnet.etc import resnet20
from shelf.pruners.scoring import get_hgp_score, get_hgp_score, get_abs_gradient_score


EPOCHS = 200
DEVICE = 'cuda'
PRUNE_RATE = 0.9
# FINAL_WEIGHT_PATH = './saves/final_weights_vloss0.323.pth'
FINAL_WEIGHT_PATH = './saves/epoch_5.pth'

train_loader, val_loader = get_CIFAR10_dataset(root='../data', augmentation=False)

model_init = resnet20().to(DEVICE)
model_init.load_state_dict(torch.load('./saves/initial_weights.pth'))

model = resnet20().to(DEVICE)
model.load_state_dict(torch.load(FINAL_WEIGHT_PATH))
model.eval()
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

num_params = sum(p.numel() for p in model.parameters())


for name, module in model_init.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)

weight_diff_dict = {
    pname: (param - param_init).clone()
    for (pname, param_init), (pname, param) in zip(model_init.named_parameters(), model.named_parameters())
    if 'weight_orig' in pname
}


nprune_num_groups = {
    pname: int((1-PRUNE_RATE) * param.numel())
    for pname, param in model.named_parameters() 
    if 'weight_orig' in pname
}

nprune_group_dict = {
    # pname: torch.cat([torch.arange(nprune_num_groups[pname]), -torch.ones(param.numel() - nprune_num_groups[pname])])[torch.randperm(param.numel())].view(param.shape)
    pname: torch.randint_like(param, 0, nprune_num_groups[pname])
    for pname, param in model.named_parameters()
    if 'weight_orig' in pname
}

grad_score_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}
hgp_score_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}

for input, label in tqdm(train_loader, leave=False):
    input, label = input.to(DEVICE), label.to(DEVICE)

    hvp_score = get_hgp_score(input, label, model)
    for pname, value in zip(dict(model.named_parameters()).keys(), hvp_score):
        hgp_score_dict[pname] += value
    
    grad = gradient_fo(input, label, model, criterion=criterion)
    for pname, value in grad.items():
        grad_score_dict[pname] += value

for pname, value in hgp_score_dict.items():
    hgp_score_dict[pname] /= len(train_loader)
    grad_score_dict[pname] /= len(train_loader)

params_flat = torch.cat([param.view(-1) for param in model.parameters()])
params_norm = torch.norm(params_flat, p=2)

hvp_score_flat = torch.cat([param.view(-1) for param in hgp_score_dict.values()])
hvp_score_norm = torch.norm(hvp_score_flat, p=2)

ideal_dimension_dict = {
    pname: params_norm * hvp_score + hvp_score_norm * param
    for (pname, param), hvp_score in zip(model.named_parameters(), hgp_score_dict.values())
    # pname: value
    # for pname, value in grad_score_dict.items()
}

nprune_dimension_dict = {
    pname: ideal_dimension_dict[pname]
    for pname, param in model.named_parameters()
    if 'weight_orig' in pname
}

print("Creating dimensions")
pbar = tqdm(nprune_dimension_dict.items(), leave=False)
for pname, dimension in pbar:
    for group_idx in range(nprune_num_groups[pname]):
        pbar.set_description(f"{pname} ({group_idx}/{nprune_num_groups[pname]})")
        dimension[nprune_group_dict[pname] == group_idx] /= dimension[nprune_group_dict[pname] == group_idx].view(-1).norm()

total_squared_error = 0
total_squared_diff = 0

for pname, param in weight_diff_dict.items():
    if 'weight_orig' in pname:
        # remove nprune_dimension_dict[pname] dimensions from the weight tensor
        param_flat = param.view(-1)

        pbar = tqdm(range(nprune_num_groups[pname]), leave=False)
        for group_idx in pbar:
            group_dim = nprune_dimension_dict[pname] * (nprune_group_dict[pname] == group_idx).float().to(DEVICE)

            weight_group_component = torch.dot(param_flat, group_dim.view(-1))
            pbar.set_description(f"Param {pname}: {torch.norm(param):7.4f}")
            
            param = param - weight_group_component * group_dim

        # print the error
        print(f"Error in {pname} (gnum: {nprune_num_groups[pname]}): {torch.norm(param).item():.4f}/{torch.norm(param_flat).item():.4f} ({torch.norm(param).item() / torch.norm(param_flat).item() * 100:.2f}%)")
        total_squared_error += torch.norm(param).item() ** 2
        total_squared_diff += torch.norm(param_flat).item() ** 2
    
print(f"Total squared error: {total_squared_error:.4f}/{total_squared_diff:.4f} ({total_squared_error / total_squared_diff * 100:.2f}%)")
print(f"as norm: {total_squared_error ** 0.5:.4f}/{total_squared_diff ** 0.5:.4f} ({total_squared_error ** 0.5 / total_squared_diff ** 0.5 * 100:.2f}%)")