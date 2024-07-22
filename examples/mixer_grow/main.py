import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np

import matplotlib.pyplot as plt

from torch.nn.utils.prune import remove, l1_unstructured
from tqdm import tqdm
import torchsummary

from model import MLPMixerLora, LinearLora, Linear

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.pruners.scoring import get_grasp_score, get_zo_grasp_score


EPOCHS = 300
DEVICE = 'cuda'


train_loader, val_loader = get_CIFAR10_dataset(root='../data')

model = MLPMixerLora(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=64, 
    hidden_c=512, 
    num_layers=8, 
    subdim_ratio=0.1,
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)

# simple mlp model
# model = nn.Sequential(
#     nn.Flatten(),
#     LinearLora(3*32*32, 512, 64),
#     nn.ReLU(),
#     LinearLora(512, 256, 64),
#     nn.ReLU(),
#     LinearLora(256, 10, 64)
# ).to(DEVICE)

torchsummary.summary(model, (3, 32, 32))



criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# Train the model
for epoch in range(EPOCHS):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    for module in model.modules():
        if isinstance(module, LinearLora):
            module.flush()

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}%")
