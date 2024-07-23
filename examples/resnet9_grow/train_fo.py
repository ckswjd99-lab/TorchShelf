import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np

import matplotlib.pyplot as plt

from models import ResNet9

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate


EPOCHS = 100
DEVICE = 'cuda'


train_loader, val_loader = get_CIFAR10_dataset(augmentation=True)

model = ResNet9().to(DEVICE)
# model = ResNet20().to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_params = sum(p.numel() for p in model.parameters())

# Train the model
best_val_acc = 0

for epoch in range(EPOCHS):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% |" + (" *" if is_best else ""))


print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
