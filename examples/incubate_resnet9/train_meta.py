import torch
import torch.nn as nn
import torchsummary

from models import ResNet9_Meta

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate

EPOCH = 100
DEVICE = 'cuda'

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)
model = ResNet9_Meta().to(DEVICE)
torchsummary.summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

for epoch in range(EPOCH):
    t_loss_sum = 0
    t_acc_sum = 0
    num_steps = 0

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc * 100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")


torch.save(model.state_dict(), f'./saves/meta_vloss{val_loss:.3f}.pth')