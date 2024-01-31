import torch
import torch.nn as nn
import torch.nn.functional as F

from import_shelf import shelf
from shelf.models.mutable import MutableResNet18
from shelf.dataloaders.cifar import get_CIFAR10_dataset

train_loader, val_loader = get_CIFAR10_dataset(batch_size=128)

sample_input = next(iter(train_loader))[0]

model = MutableResNet18(input_size=32, num_output=10, input_channel=3, shrink_ratio=8/512)
model = model.cuda()
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'>> Number of parameters: {num_params}')
model(sample_input.cuda())
print('inferences well')


model.grow_tobe(9/512)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f'>> Grown to: {num_params}')
model(sample_input.cuda())
print('inferences well')

model.grow_tobe(10/512)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f'>> Grown to: {num_params}')
model(sample_input.cuda())
print('inferences well')
