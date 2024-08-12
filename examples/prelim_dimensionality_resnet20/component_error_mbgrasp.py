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

model_temp = resnet20().to(DEVICE)
early_path_list = [
    # './saves/epoch_1.pth',
    # './saves/epoch_2.pth',
    # './saves/epoch_3.pth',
    # './saves/epoch_4.pth',
    # './saves/epoch_5.pth',
    # './saves/epoch_6.pth',
    # './saves/epoch_7.pth',
    # './saves/epoch_8.pth',
    # './saves/epoch_9.pth',
    # './saves/epoch_10.pth',
]

model_final = resnet20().to(DEVICE)
model_final.load_state_dict(torch.load(FINAL_WEIGHT_PATH))
model_final.eval()
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model_final.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

num_params = sum(p.numel() for p in model_final.parameters())


for name, module in model_init.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)

for name, module in model_final.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)

weight_diff_dict = {
    pname: (param - param_init).clone().view(-1)
    for (pname, param_init), (pname, param) in zip(model_init.named_parameters(), model_final.named_parameters())
}

# create multiple GraSP-OPT directions from mini-batches
num_mbatch = 391
hvp_dict_list = []
grad_dict_list = []

pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
for i, (input, target) in pbar:
    input, target = input.to(DEVICE), target.to(DEVICE)

    if i % num_mbatch == 0:
        hvp_dict_list.append({pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()})
        grad_dict_list.append({pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()})
    
    hvp_score_list = get_hvp_score(input, target, model_init)
    grad_score_dict = gradient_fo(input, target, model_init, criterion)

    for (pname, hvp_opt_value), hvp_score in zip(hvp_dict_list[-1].items(), hvp_score_list):
        hvp_opt_value += hvp_score

    for (pname, grad_opt_value), grad_score in zip(grad_dict_list[-1].items(), grad_score_dict.values()):
        grad_opt_value += grad_score

    if i % num_mbatch == num_mbatch - 1:
        for hvp_val_dict in hvp_dict_list[-1].values():
            hvp_val_dict /= num_mbatch
        for grad_val_dict in grad_dict_list[-1].values():
            grad_val_dict /= num_mbatch

    break
    

hvp_dict_list = hvp_dict_list[:len(train_loader) // num_mbatch]
grad_dict_list = grad_dict_list[:len(train_loader) // num_mbatch]

grasp_opt_dict_list = []

for hvp_val_dict in hvp_dict_list:
    hvp_norm = torch.norm(torch.cat([param.view(-1) for pname, param in hvp_val_dict.items()]), p=2)
    param_norm = torch.norm(torch.cat([param.view(-1) for pname, param in model_init.named_parameters()]), p=2)

    grasp_opt_dict = {pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()}
    hgp_dict = {pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()}
    for (pname, param), (pname, hvp_value) in zip(model_init.named_parameters(), hvp_val_dict.items()):
        if 'weight_orig' in pname:
            param_flat = param.view(-1)

            hvp_flat = hvp_value.view(-1)
            
            grasp_opt_dict[pname] = hvp_norm * param_flat + param_norm * hvp_flat
            grasp_opt_dict[pname] /= torch.norm(grasp_opt_dict[pname].view(-1), p=2)

            hgp_dict[pname] = hvp_flat / torch.norm(hvp_flat, p=2)
    
    # grasp_opt_dict_list.append(grasp_opt_dict)
    # grasp_opt_dict_list.append(hgp_dict)

    param_dict = {pname: param.clone() for pname, param in model_init.named_parameters()}
    for pname, param in param_dict.items():
        if 'weight_orig' in pname:
            param_dict[pname] /= torch.norm(param_dict[pname].view(-1), p=2)
            param_dict[pname] = param_dict[pname].view(-1)
    
    grasp_opt_dict_list.append(param_dict)


for grad_dict in grad_dict_list:
    grad_opt_dict = {pname: torch.zeros_like(param) for pname, param in model_init.named_parameters()}
    for (pname, param), (pname, grad_value) in zip(model_init.named_parameters(), grad_dict.items()):
        if 'weight_orig' in pname:
            param_flat = param.view(-1)
            grad_flat = grad_value.view(-1)
            
            grad_opt_dict[pname] = grad_flat / torch.norm(grad_flat, p=2)
    
    # grasp_opt_dict_list.append(grad_opt_dict)

for early_path in early_path_list:
    model_temp.load_state_dict(torch.load(early_path))
            
    for name, module in model_temp.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.0)

    early_ticket = {pname: (param_early - param_init).clone().view(-1) for (pname, param_early), (pname, param_init) in zip(model_temp.named_parameters(), model_init.named_parameters())}
    for pname, param in early_ticket.items():
        if 'weight_orig' in pname:
            param /= torch.norm(param.view(-1), p=2)

    grasp_opt_dict_list.append(early_ticket)

    for name, module in model_temp.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, name='weight')

# remove opt direction components from weight diff
weight_diff_clone = {
    pname: (param - param_init).clone().view(-1)
    for (pname, param_init), (pname, param) in zip(model_init.named_parameters(), model_final.named_parameters())
    if 'weight_orig' in pname
}

# paramwise orthogonalize
for i_idx in range(len(grasp_opt_dict_list)):
    for j_idx in range(i_idx + 1, len(grasp_opt_dict_list)):
        for pname, opt_dict_i in grasp_opt_dict_list[i_idx].items():
            if 'weight_orig' in pname:
                opt_dict_j = grasp_opt_dict_list[j_idx][pname]
                opt_dict_j -= torch.dot(opt_dict_i.view(-1), opt_dict_j.view(-1)) * opt_dict_i / torch.norm(opt_dict_i.view(-1), p=2) 


for grasp_opt_dict in grasp_opt_dict_list:
    for pname, weight_diff in weight_diff_dict.items():
        if 'weight_orig' in pname:
            before_norm = weight_diff.norm()

            param_size = weight_diff.numel()
            num_group = int(param_size * (1 - PRUNE_RATE)) // len(grasp_opt_dict_list)
            # num_group = 1
            group_idx_mat = torch.randint_like(weight_diff, 0, num_group)

            for group_idx in range(num_group):
                group_mask = (group_idx_mat == group_idx).float()
                opt_mat = grasp_opt_dict[pname] * group_mask
                if torch.norm(opt_mat.view(-1), p=2) == 0:
                    continue
                opt_mat /= torch.norm(opt_mat.view(-1), p=2)
                weight_diff -= torch.dot(weight_diff.view(-1), opt_mat.view(-1)) * opt_mat

            print(f"{pname} (#group: {num_group}): {before_norm:.4f} -> {weight_diff.norm():.4f}")

    error_norm = sum([weight_diff.norm() ** 2 for pname, weight_diff in weight_diff_dict.items() if 'weight_orig' in pname])
    orig_norm = sum([weight_diff.norm() ** 2 for pname, weight_diff in weight_diff_clone.items() if 'weight_orig' in pname])

    print(f"Total squared error: {error_norm:.4f} / {orig_norm:.4f} ({error_norm / orig_norm * 100 :.2f}%)")
    print(f"as norm: {error_norm ** 0.5:.4f} / {orig_norm ** 0.5:.4f} ({error_norm ** 0.5 / orig_norm ** 0.5 * 100 :.2f}%)")
    print()
    

# create projected weight
for pname, param in model_final.named_parameters():
    if 'weight_orig' in pname:
        param.data = param.data - weight_diff_dict[pname].view(param.size())

# validate
val_acc, val_loss = validate(val_loader, model_final, criterion, 0)
print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
    