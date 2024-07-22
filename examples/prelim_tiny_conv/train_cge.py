from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.zeroth_order import gradient_fwd, functional_xent, group_by_gradient_exp
from shelf.trainers.classic import train, validate

from models import ConvNet770, ConvNet3250, ConvNet6694
from tqdm import tqdm

import torch
import torch.nn as nn
import warmup_scheduler

# Hyperparams
EPOCHS = 50
DEVICE = 'cuda'

# Load CIFAR-10 dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=128, augmentation=False)

# Load models
model770 = ConvNet770().to(DEVICE)
model4010 = ConvNet3250().to(DEVICE)
model6236 = ConvNet6694().to(DEVICE)

# Train model770
model = model770
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=base_scheduler)

momentum_dict = {pname: torch.randn_like(param) for pname, param in model.named_parameters()}
num_groups = int(num_params * 0.1)

for epoch in range(EPOCHS):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    scheduler.step()

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc*100:.2f}%, V Loss: {val_loss:.4f}, V Acc: {val_acc*100:.2f}%")