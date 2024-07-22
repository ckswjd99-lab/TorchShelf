from import_shelf import shelf
from shelf.models.resnet.etc import resnet20
from shelf.trainers import train, validate
from shelf.dataloaders import get_CIFAR10_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import warmup_scheduler
import os

# hyperparameters
ModelClass = resnet20

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DEVICE = 'cuda'

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE, augmentation=False)

print(f'========== From Scratch: {ModelClass.__name__} ==========')

# model, criterion, optimizer
model_resnet = ModelClass().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=base_scheduler)

# save initial weights
if os.path.exists('./saves/resnet20/initial_model.pth'):
    model_resnet.load_state_dict(torch.load('./saves/resnet20/initial_model.pth'))
else:
    torch.save(model_resnet.state_dict(), './saves/resnet20/initial_model.pth')

best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = optimizer.param_groups[0]['lr']

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_resnet, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_resnet, criterion, epoch)

    # print training/validation statistics
    print(
        'Epoch: {0}/{1}\t'
        'LR: {lr:.6f}\t'
        'Train Accuracy {train_acc:.3f}\t'
        'Train Loss {train_loss:.3f}\t'
        'Val Accuracy {val_acc:.3f}\t'
        'Val Loss {val_loss:.3f}'
        .format(
            epoch + 1, EPOCHS, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )
    scheduler.step()

    # record best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_resnet.state_dict(), './saves/resnet20/best_model_running.pth')
        print(f'>> Model saved (Val Acc: {best_val_acc:.3f})')

for epoch in range(20):
    train_acc, train_loss = train(train_loader, model_resnet, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model_resnet, criterion, epoch)

    print(
        'Epoch: {0}/{1}\t'
        'LR: {lr:.6f}\t'
        'Train Accuracy {train_acc:.3f}\t'
        'Train Loss {train_loss:.3f}\t'
        'Val Accuracy {val_acc:.3f}\t'
        'Val Loss {val_loss:.3f}'
        .format(
            epoch + 1, EPOCHS, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

    # record best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_resnet.state_dict(), './saves/resnet20/best_model_running.pth')
        print(f'>> Model saved (Val Acc: {best_val_acc:.3f})')

# load the best model
model_resnet.load_state_dict(torch.load('./saves/resnet20/best_model_running.pth'))
val_acc, val_loss = validate(val_loader, model_resnet, criterion, epoch)
print(f'Best Validation Accuracy: {val_acc:.3f}')
torch.save(model_resnet.state_dict(), f'./saves/resnet20/best_model_vacc{val_acc}.pth')
os.remove('./saves/resnet20/best_model_running.pth')