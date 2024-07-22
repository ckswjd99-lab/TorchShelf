import torch
import torch.nn as nn
import torchsummary

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

model = MLPMixer(hidden_s=64, hidden_c=512, num_layers=8).to(DEVICE)
torchsummary.summary(model, (3, 32, 32))
