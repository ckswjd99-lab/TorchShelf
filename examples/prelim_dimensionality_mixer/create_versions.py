import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

import matplotlib.pyplot as plt

import warmup_scheduler


from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer


EPOCH = 100
DEVICE = 'cuda'


# Load the CIFAR10 dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=128)

# Create the model
model = MLPMixer(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=64,
    hidden_c=512, 
    num_layers=8, 
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params:,d}")

# if initial weights can be loaded, load from ./saves folder.
# else, save it in ./saves folder.
if os.path.exists('./saves/initial_weights.pth'):
    model.load_state_dict(torch.load('./saves/initial_weights.pth'))
else:
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    torch.save(model.state_dict(), './saves/initial_weights.pth')

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5, betas=(0.9, 0.99))
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)
# base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=3e-2 **(1/EPOCH))
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_scheduler)


# Logs
for epoch in range(EPOCH):
    t_loss_sum = 0
    t_acc_sum = 0
    num_steps = 0

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    scheduler.step()

    print(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc * 100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")

    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), f'./saves/epoch_{epoch + 1}.pth')
        # print(f"Model saved at ./saves/epoch_{epoch + 1}.pth")
    
torch.save(model.state_dict(), f'./saves/final_weights_vloss{val_loss:.3f}.pth')