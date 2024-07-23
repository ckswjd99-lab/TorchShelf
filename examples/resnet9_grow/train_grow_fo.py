import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np

import matplotlib.pyplot as plt

from torch.nn.utils.prune import remove, l1_unstructured
from tqdm import tqdm
import os

from models import ResNet9, ResNet20

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.pruners.scoring import get_grasp_score, get_zo_grasp_score


EPOCHS = 100
DEVICE = 'cuda'


train_loader, val_loader = get_CIFAR10_dataset(augmentation=True)

model = ResNet9().to(DEVICE)
# model = ResNet20().to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_params = sum(p.numel() for p in model.parameters())

def get_random_score(model):
    conv_lin_modules = {mname: module for mname, module in model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)}
    return {mname: torch.randn_like(module.weight) for mname, module in conv_lin_modules.items()}


def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            remove(module, name='weight')

def undo_pruning(model):
    for name, buffer in model.named_buffers():
        if 'weight_mask' in name:
            buffer.fill_(1)
    
    remove_pruning(model)

def prune_model_with_score(model, pruning_rate, score_dict):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            l1_unstructured(module, name='weight', amount=pruning_rate, importance_scores=score_dict[name])

def extract_pruning_mask(model):
    buffers = dict(model.named_buffers())

    pruning_mask = {}
    for pname, param in model.named_parameters():
        if 'weight_orig' in pname:
            buffer_name = pname.replace('weight_orig', 'weight_mask')
            pruning_mask[buffer_name] = buffers[buffer_name].clone()

    return pruning_mask

def train(train_loader, model, criterion, optimizer, epoch, config, epoch_pbar=None, verbose=True):
    model.train()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    grad_abs_sum = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for pname, param in model.named_parameters():
            grad_abs_sum[pname] += torch.abs(param.grad)

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)
    
        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        if verbose:
            pbar.set_postfix(train_accuracy=accuracy, train_loss=avg_loss)
        
    accuracy = num_correct / num_data
    avg_loss = sum_loss / num_data

    grad_abs_avg = {pname: grad / num_data for pname, grad in grad_abs_sum.items()}
    config['grad_abs_avg'] = grad_abs_avg

    return accuracy, avg_loss



pruning_schedule = {
    # epoch 0~100: 0.99 -> 0.00
    epoch: 0.99 - epoch * 0.99 / EPOCHS
    for epoch in range(EPOCHS)
}
pruning_score_dict = get_random_score(model)
# save score dict
torch.save(pruning_score_dict, './saves/pruning_score_dict.pth')

prune_model_with_score(model, 1.0, pruning_score_dict)
prev_pruning_mask = extract_pruning_mask(model)

init_values = {pname: param.clone() for pname, param in model.named_parameters()}

undo_pruning(model)

# Train the model
best_val_acc = 0

for epoch in range(EPOCHS):
    # prune the model
    pruning_rate = pruning_schedule[epoch]
    prune_model_with_score(model, pruning_rate, pruning_score_dict)
    now_pruning_mask = extract_pruning_mask(model)
    only_now_pruning_mask = {pname: now_pruning_mask[pname] - prev_pruning_mask[pname] for pname in now_pruning_mask}

    # initialize the mask
    for mname in only_now_pruning_mask:
        pname = mname.replace('weight_mask', 'weight_orig')
        model.state_dict()[pname] += init_values[pname] * only_now_pruning_mask[mname]

    # train the model
    config = {}
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, config)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    prev_pruning_mask = now_pruning_mask

    # remove the mask
    remove_pruning(model)

    # logging
    alive_params = 0
    for param in model.parameters():
        alive_params += torch.sum(param != 0).item()

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        torch.save(model.state_dict(), './saves/best_model.pth')

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | PARAM {int(alive_params):10,d} ({alive_params/num_params*100:.4f}%)")

print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

os.rename('./saves/best_model.pth', f'./saves/best_model_grow_fo_vacc{best_val_acc*100:.2f}.pth')

# evaluate the pruning rate
pruning_rates = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
pruning_scores = []

for pruning_rate in pruning_rates:
    model.load_state_dict(torch.load(f'./saves/best_model_grow_fo_vacc{best_val_acc*100:.2f}.pth'))

    prune_model_with_score(model, pruning_rate, pruning_score_dict)
    val_acc, _ = validate(val_loader, model, criterion, 0)
    pruning_scores.append(val_acc)
    remove_pruning(model)

print(pruning_rates)
print(pruning_scores)