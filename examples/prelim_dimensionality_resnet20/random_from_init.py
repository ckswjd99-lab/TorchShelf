import torch
import torch.nn as nn

import torchsummary
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'
DISTANCE = 1e+7
STEPS = 10

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

model = MLPMixer().to(DEVICE)
model.load_state_dict(torch.load('./saves/set1_default/initial_weights.pth'))
num_params = sum(p.numel() for p in model.parameters())

criterion = nn.CrossEntropyLoss()

train_gradient = {pname: torch.zeros_like(param) for pname, param in model.state_dict().items()}
for input, label in tqdm(train_loader):
    input, label = input.to(DEVICE), label.to(DEVICE)
    loss = criterion(model(input), label)
    loss.backward()
    for pname, param in model.named_parameters():
        train_gradient[pname] += param.grad.data
    model.zero_grad()

train_gradient = {pname: grad / len(train_loader) for pname, grad in train_gradient.items()}

# pnoises = {pname: torch.randn_like(param) for pname, param in model.state_dict().items()}
pnoises = {pname: -grad for pname, grad in train_gradient.items()}

val_acc, val_loss_orig = validate(val_loader, model, criterion, 0)
for alpha in torch.linspace(0, 1, STEPS):

    for pname, param in model.named_parameters():
        param.data += pnoises[pname] * alpha * DISTANCE / num_params
    
    val_acc, val_loss_perb = validate(val_loader, model, criterion, 0)

    for pname, param in model.named_parameters():
        param.data -= pnoises[pname] * alpha * DISTANCE / num_params

    print(f"Alpha: {alpha:.4f}, Loss Orig: {val_loss_orig:.4f}, Loss Perturbed: {val_loss_perb:.4f}")