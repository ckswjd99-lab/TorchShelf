import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import numpy as np
import os
import time

from import_shelf import shelf
from shelf.models.transformer import VisionTransformer
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers import train_zo_paramwise_autolr, validate

os.environ["CUDA_VISIBLE_DEVICES"]="MIG-60fed909-9539-55f4-9bab-e99df995d4a0"


### HYPERPARAMS ###

EPOCHS = 200
BATCH_SIZE = 512
LR_MAX = 2e-1
SMOOTHING = 1e-3
QUERY_RATIO = 0.01

DEPTH = 4
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM_HIDDEN = 16
NUM_CLASSES = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH_MODEL = './saves/train_zo_tinyvit/model.pth'


### DATA LOADING ###

train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


### MODEL ###

model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    dim=DIM_HIDDEN,
    depth=DEPTH,
    heads=1,
    dim_head=16,
    mlp_dim=32,
    dropout=0.1,
    emb_dropout=0.1,
).to(DEVICE)

print(model)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_query = int(QUERY_RATIO * num_params)

print(f"Number of parameters: {num_params}")
print(f"Number of queries: {num_query}")
    
    
### OTHERS ###

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX)

### TRAINING ###

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_config = {}
    train_acc, train_loss = train_zo_paramwise_autolr(
        train_loader, model, criterion, optimizer, epoch,
        max_lr=LR_MAX, smoothing=SMOOTHING, query=num_query, confidence=1.0,
        config=train_config
    )
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    lr_avg = train_config['avg_lr']
    lr_std = train_config['std_lr']

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS}, "
        f"LR: {lr_avg:.6f}±{lr_std:.6f} | "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc * 100:.2f}%, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc*100:.2f}% | "
        f"Time: {time.time() - start_time:.3f}s"
    )

torch.save(model.state_dict(), PATH_MODEL)