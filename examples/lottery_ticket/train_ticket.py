from import_shelf import shelf
from shelf.models.resnet.etc import resnet20
from shelf.trainers import train, validate
from shelf.dataloaders import get_CIFAR10_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.utils.prune as prune

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
TICKET_MODEL = './saves/ticket_prate0.2.pth'
PRUNING_RATE = TICKET_MODEL.split('prate')[1].split('.pth')[0]

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=BATCH_SIZE, augmentation=False)

# zero prune the model and load the ticket
model_ticket = ModelClass().to(DEVICE)
for mname, module in model_ticket.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.0)

model_ticket.load_state_dict(torch.load(TICKET_MODEL))

print(f"Pruning Rate: {PRUNING_RATE}")

# train the ticket model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ticket.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=base_scheduler)

best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = optimizer.param_groups[0]['lr']

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_ticket, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_ticket, criterion, epoch)

    scheduler.step()

    # print training/validation statistics
    print(
        f"EPOCH: {epoch + 1:3d}/{EPOCHS:3d}, LR: {epoch_lr:.4e} | T Acc {train_acc*100:5.2f}%, T Loss {train_loss:.4f}, V Acc {val_acc*100:5.2f}%, V Loss {val_loss:.4f}"
        + (" | *" if val_acc > best_val_acc else " |")
    )

    # update the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_ticket.state_dict(), './saves/best_ticket_running.pth')

for epoch in range(20):
    train_acc, train_loss = train(train_loader, model_ticket, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model_ticket, criterion, epoch)

    print(
        f"EPOCH: {epoch + 1:3d}/{20:3d}, LR: {epoch_lr:.4e} | T Acc {train_acc*100:5.2f}%, T Loss {train_loss:.4f}, V Acc {val_acc*100:5.2f}%, V Loss {val_loss:.4f}"
        + (" | *" if val_acc > best_val_acc else " |")
    )

    # update the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_ticket.state_dict(), './saves/best_ticket_running.pth')

# load the best model
model_ticket.load_state_dict(torch.load('./saves/best_ticket_running.pth'))
val_acc, val_loss = validate(val_loader, model_ticket, criterion, epoch)
print(f"Best model validation accuracy: {val_acc:.4f}")

# save the best model
torch.save(model_ticket.state_dict(), f'./saves/best_ticket_prate{PRUNING_RATE}_vacc{val_acc:.4f}.pth')
os.remove('./saves/best_ticket_running.pth')