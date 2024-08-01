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
from shelf.pruners.scoring import get_hvp_score, get_hvp_score, get_abs_gradient_score


EPOCHS = 200
DEVICE = 'cuda'
PRUNE_RATE = 0.9
FINAL_WEIGHT_PATH = './saves/final_weights_vloss0.323.pth'

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
    pname: (param - param_init).clone().view(-1)
    for (pname, param_init), (pname, param) in zip(model_init.named_parameters(), model.named_parameters())
}

# create multiple GraSP-OPT directions from mini-batches
num_mbatch = 391
hvp_dict_list = []

pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
for i, (input, target) in pbar:
    input, target = input.to(DEVICE), target.to(DEVICE)

    if i % num_mbatch == 0:
        hvp_dict_list.append({pname: torch.zeros_like(param) for pname, param in model.named_parameters()})
    
    hvp_score_list = get_hvp_score(input, target, model)
    grad_score_dict = gradient_fo(input, target, model, criterion)
    for (pname, hvp_opt_value), hvp_score in zip(hvp_dict_list[-1].items(), hvp_score_list):
        hvp_opt_value += hvp_score
    
    if i % num_mbatch == num_mbatch - 1:
        for hvp_val_dict in hvp_dict_list[-1].values():
            hvp_val_dict /= num_mbatch

hvp_dict_list = hvp_dict_list[:len(train_loader) // num_mbatch]

grasp_opt_dict_list = []

for hvp_val_dict in hvp_dict_list:
    hvp_norm = torch.norm(torch.cat([param.view(-1) for pname, param in hvp_val_dict.items()]), p=2)
    param_norm = torch.norm(torch.cat([param.view(-1) for pname, param in model.named_parameters()]), p=2)

    grasp_opt_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}
    hgp_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}
    for (pname, param), (pname, hvp_value) in zip(model.named_parameters(), hvp_val_dict.items()):
        if 'weight_orig' in pname:
            param_flat = param.view(-1)

            hvp_flat = hvp_value.view(-1)
            
            grasp_opt_dict[pname] = hvp_norm * param_flat + param_norm * hvp_flat
            grasp_opt_dict[pname] /= torch.norm(grasp_opt_dict[pname].view(-1), p=2)

            hgp_dict[pname] = hvp_flat / torch.norm(hvp_flat, p=2)
    
    grasp_opt_dict_list.append(grasp_opt_dict)
    grasp_opt_dict_list.append(hgp_dict)

    param_dict = {pname: param.clone() for pname, param in model.named_parameters()}
    for pname, param in param_dict.items():
        if 'weight_orig' in pname:
            param_dict[pname] /= torch.norm(param_dict[pname].view(-1), p=2)
            param_dict[pname] = param_dict[pname].view(-1)
    
    grasp_opt_dict_list.append(param_dict)


# remove opt direction components from weight diff
weight_diff_clone = {
    pname: (param - param_init).clone().view(-1)
    for (pname, param_init), (pname, param) in zip(model_init.named_parameters(), model.named_parameters())
    if 'weight_orig' in pname
}


for grasp_opt_dict in grasp_opt_dict_list:
    for pname, weight_diff in weight_diff_dict.items():
        if 'weight_orig' in pname:
            before_norm = weight_diff.norm()
            weight_diff -= torch.dot(weight_diff.view(-1), grasp_opt_dict[pname].view(-1)) * grasp_opt_dict[pname]

            print(f"{pname}: {before_norm:.4f} -> {weight_diff.norm():.4f}")

    error_norm = sum([weight_diff.norm() ** 2 for pname, weight_diff in weight_diff_dict.items() if 'weight_orig' in pname])
    orig_norm = sum([weight_diff.norm() ** 2 for pname, weight_diff in weight_diff_clone.items() if 'weight_orig' in pname])

    print(f"Total squared error: {error_norm:.4f} / {orig_norm:.4f} ({error_norm / orig_norm * 100 :.2f}%)")
    print(f"as norm: {error_norm ** 0.5:.4f} / {orig_norm ** 0.5:.4f} ({error_norm ** 0.5 / orig_norm ** 0.5 * 100 :.2f}%)")
    print()
    

    