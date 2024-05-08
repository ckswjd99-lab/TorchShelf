import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

from functools import partial
import torch.func as fc
import torch.nn.functional as F

from import_shelf import shelf
from shelf.models.transformer import VisionTransformer
from shelf.models.mutable import MutableResNet18
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.zeroth_order import gradient_fo, gradient_fwd, group_by_gradient_exp, drop_momentum_by_score
from shelf.trainers.classic import train, validate

from tqdm import tqdm
import argparse

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)

args = parser.parse_args()


### HYPERPARAMS ###

BATCH_SIZE = 512
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 10

NUM_QUERY = 1
LR_MAX = 1e-2
LR_MIN = 1e-5
MOMENTUM = 0.9
# QUERY_BASE = 1.05
QUERY_BASE = 1.005
LEVEL_NOISE = 1e-4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH_MODEL = './saves/zoo_poc.pth'
PATH_MODEL_BASE = './saves/zoo_poc_baseline.pth'


### DATA LOADING ###

train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_loader.dataset.data[i])
    plt.title(classes[train_loader.dataset.targets[i]])
    plt.axis('off')


def functional_xent(
    params,
    buffers,
    names,
    model,
    x,
    t,
):
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return F.cross_entropy(y, t)

def calc_decay_rate(gradient_momentum):
    all_gradients = torch.cat([grad.flatten() for grad in gradient_momentum.values()]).abs()
    all_gradients = torch.sort(all_gradients, descending=True).values
    
    x = np.arange(all_gradients.size(0))
    y = np.log(all_gradients.cpu().numpy() + 1e-8)
    
    a, B_log = np.polyfit(x, y, 1)

    return -a

def group_by_given_logr(estimated_gradient, logr, descending=True):
    all_gradients = torch.cat([grad.flatten() for grad in estimated_gradient.values()]).abs()
    num_params = all_gradients.size(0)

    num_groups = int(np.log(np.exp(np.log(num_params) + logr) - num_params + 1) / logr)
    
    group_dict, _ = group_by_gradient_exp(estimated_gradient, num_groups, descending)

    return group_dict, num_groups


def train_zo(
        train_loader, model, criterion, optimizer, epoch,
        smoothing=1e-3, query=1, lr_auto=True, lr_max=1e-2, lr_min=1e-5, momentum=0.9,
        num_groups=1, group_dict=None, group_sizes=None, momentum_dict=None, decay_rate=None, r_per_decay=500, level_noise=0, warmup=False, num_drop=0,
        config=None, verbose=True
    ):
    model.eval()

    estimated_gradient = None

    # Prepare statistics
    num_data = 0
    num_correct = 0
    sum_loss = 0
    num_query = 0
    num_bp = 0
    
    sum_decay_rate = 0

    sum_cosine_sim = 0
    sum_magnitude = 0
    sum_mse = 0

    sum_group_diff = 0

    group_avg = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        group_avg[name] = 0

    # Prepare grouping
    if group_dict == None:
        input, label = next(iter(train_loader))
        input = input.cuda()
        label = label.cuda()

        # real_gradient = gradient_fo(input, label, model, criterion)
        real_gradient = {}
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            real_gradient[name] = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

        group_dict, group_sizes = group_by_gradient_exp(real_gradient, num_groups)

        estimated_gradient = real_gradient

    if momentum_dict == None:
        momentum_dict = {}
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            momentum_dict[name] = torch.randn_like(param.data)
    gradient_momentum = momentum_dict
        
    if decay_rate == None:
        if estimated_gradient == None:
            input, label = next(iter(train_loader))
            input = input.cuda()
            label = label.cuda()

            estimated_gradient = gradient_fo(input, label, model, criterion)

        decay_rate = calc_decay_rate(estimated_gradient)
    

    # Train the model
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    
    for i, (input, label) in enumerate(pbar):
        input = input.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        num_iter = 10
        itgge_num_groups = num_groups//num_iter

        # Gradient estimation
        real_gradient = gradient_fo(input, label, model, criterion)
        estimated_gradient = gradient_fwd(
            input, label, model, criterion,
            query=num_iter, type='itgge', momentum_dict=estimated_gradient, num_groups=itgge_num_groups,
            cheat_fo=True
        )
        num_query += query * num_groups

        # Cosine similarity
        all_real_gradients = torch.cat([grad.flatten() for grad in real_gradient.values()])
        all_estimated_gradients = torch.cat([grad.flatten() for grad in estimated_gradient.values()])
        # all_estimated_gradients = torch.cat([grad.flatten() for grad in estimated_gradient.values()])

        real_norm = all_real_gradients.norm()
        estimated_norm = all_estimated_gradients.norm()
        
        cosine_similarity = (all_real_gradients * all_estimated_gradients).sum() / (real_norm * estimated_norm)

        # Apply gradient
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            param.grad = estimated_gradient[name]
            # param.grad = estimated_gradient[name]

        # Update momentum
        
        # Calculate update direction of Adam and use it as momentum
        gradient_momentum = estimated_gradient

        # Update the model
        optimizer.step()

        # Statistics
        output = model(input)
        loss = criterion(output, label)

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)
    
        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        sum_cosine_sim += cosine_similarity.item()
        sum_magnitude += estimated_norm.item()/real_norm.item()
        sum_mse += (all_real_gradients - all_estimated_gradients).pow(2).sum().item()

        pbar.set_postfix(
            cossim=cosine_similarity.item(), mag=estimated_norm.item()/real_norm.item(), mse=(all_real_gradients - all_estimated_gradients).pow(2).sum().item(),
            tacc=accuracy, tloss=avg_loss
        ) if verbose else None

    accuracy = num_correct / num_data
    avg_loss = sum_loss / num_data
    cosine_similarity = sum_cosine_sim / len(train_loader)
    magnitude_ratio = sum_magnitude / len(train_loader)

    if config is not None:
        group_avg = {name: val / len(train_loader) for name, val in group_avg.items()}

    if config is not None:
        config['num_query'] = num_query
        config['group_dict'] = group_dict
        config['group_sizes'] = group_sizes
        config['cosine_similarity'] = cosine_similarity
        config['magnitude_ratio'] = magnitude_ratio
        config['mse'] = sum_mse / len(train_loader)
        config['momentum_dict'] = gradient_momentum
        config['num_bp'] = num_bp
        config['group_diff'] = sum_group_diff / len(train_loader)
        config['group_avg'] = group_avg

    return accuracy, avg_loss


### MODEL ###

# TinyViT
model = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=512,
    depth=4,
    heads=6,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1
).to(DEVICE)

# VeryTinyViT2 - 43.91% by FO
# model = VisionTransformer(
#     image_size=32,
#     patch_size=4,
#     num_classes=10,
#     dim=16,
#     depth=2,
#     heads=1,
#     mlp_dim=32,
#     dropout=0.1,
#     emb_dropout=0.1
# ).to(DEVICE)

start_epoch = 0
RESUME_PATH = args.resume
if RESUME_PATH is not None:
    model.load_state_dict(torch.load(RESUME_PATH))
    start_epoch = int(RESUME_PATH.split('_')[-1].split('.')[0][1:])

    print(f"Resuming from {RESUME_PATH}, epoch {start_epoch}")

print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_groups = int(np.log(num_params) / np.log(QUERY_BASE))
print(f"Number of param_tensors: {len(list(model.parameters()))}")
print(f"Number of parameters: {num_params}")
print(f"Number of groups: {num_groups}")
print()


### OTHERS ###

# EPOCHS = 2000
EPOCHS = 20000

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.5, 0.5))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

print(optimizer)
print(scheduler)
print(scheduler.state_dict())
print()


## TRAINING ##

group_dict = None
group_sizes = None
momentum_dict = None
decay_rate = None

lr_max = LR_MAX
lr = optimizer.param_groups[0]['lr']

level_noise = lr * 1e-2

for epoch in range(start_epoch, start_epoch + EPOCHS):
    start_time = time.time()

    config = {}

    train_acc, train_loss = train_zo(
        train_loader, model, criterion, optimizer, epoch,
        query=NUM_QUERY, lr_auto=False, lr_max=lr, lr_min=lr, momentum=MOMENTUM,
        num_groups=num_groups, group_dict=group_dict, group_sizes=group_sizes, momentum_dict=momentum_dict, decay_rate=decay_rate, level_noise=level_noise,
        config=config, verbose=True
    )
    # train_acc, train_loss = train(train_loader, model, nn.CrossEntropyLoss(), optimizer, epoch)

    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    group_dict = config['group_dict']
    group_sizes = config['group_sizes']
    momentum_dict = config['momentum_dict']
    num_bp = config['num_bp']
    group_diff = config['group_diff']
    lr = optimizer.param_groups[0]['lr']
    level_noise = lr * 1e-2

    print(
        f"Epoch {epoch+1:3d}/{start_epoch + EPOCHS}, "
        f"LR: {lr:.4e},"
        f"Cosine Sim: {config['cosine_similarity']:.4f}, "
        f"Magnitude Ratio: {config['magnitude_ratio']:.4f}, "
        f"Level Noise: {level_noise:.4e} | "
        f"Train Acc: {train_acc * 100:.2f}%, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Acc: {val_acc*100:.2f}%, "
        f"Val Loss: {val_loss:.4f} | "
        f"Num Query: {config['num_query']}, "
        f"Num BP: {num_bp}, "
        f"Time: {time.time() - start_time:.3f}s"
    )

    # scheduler.step()
    if epoch % 100 == 0:
        # torch.save(model.state_dict(), f"./saves/fwdgge_poc_e{epoch+1:03d}.pth")
        torch.save(model.state_dict(), f"./saves/fwdgge_tinyvit_e{epoch+1:05d}.pth")

torch.save(model.state_dict(), PATH_MODEL)