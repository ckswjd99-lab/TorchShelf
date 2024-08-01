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
from shelf.models.resnet.etc import resnet20


EPOCH = 200
DEVICE = 'cuda'


# Load the CIFAR10 dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=128, root='../data')

# Create the model
model = resnet20().to(DEVICE)
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

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


# Logs
best_vacc = 0
for epoch in range(EPOCH):
    t_loss_sum = 0
    t_acc_sum = 0
    num_steps = 0

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    scheduler.step()

    is_best = False
    if val_acc > best_vacc:
        best_vacc = val_acc
        is_best = True

    print(
        f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc * 100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f} | "
        + ("*" if is_best else "")
    )

    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), f'./saves/epoch_{epoch + 1}.pth')
        # print(f"Model saved at ./saves/epoch_{epoch + 1}.pth")
    
torch.save(model.state_dict(), f'./saves/final_weights_vloss{val_loss:.3f}.pth')

print(f"Best validation accuracy: {best_vacc * 100:.2f}")