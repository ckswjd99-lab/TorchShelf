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
INIT_MODEL = './saves/initial_model.pth'
SOURCE_TICKET = './saves/ticket_prate0.6862.pth'
SOURCE_PRUNE_RATE = SOURCE_TICKET.split('prate')[1].split('.pth')[0]
PRUNE_RATE = 0.1

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=BATCH_SIZE, augmentation=False)

model_ticket = ModelClass().to(DEVICE)
for mname, module in model_ticket.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.0)
model_ticket.load_state_dict(torch.load(SOURCE_TICKET))

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

    # remember best accuracy and save checkpoint
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

    # remember best accuracy and save checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_ticket.state_dict(), './saves/best_ticket_running.pth')

# load the best model
model_ticket.load_state_dict(torch.load('./saves/best_ticket_running.pth'))
val_acc, val_loss = validate(val_loader, model_ticket, criterion, epoch)
print(f"Best model validation accuracy: {val_acc:.4f}")
torch.save(model_ticket.state_dict(), f'./saves/best_model_prate{float(SOURCE_PRUNE_RATE):.4f}_vacc{val_acc:.4f}.pth')
os.remove('./saves/best_ticket_running.pth')

for mname, module in model_ticket.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=PRUNE_RATE)

model_init = ModelClass().to(DEVICE)
model_init.load_state_dict(torch.load(INIT_MODEL))
for mname, module in model_init.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.0)

for (mname, module), (mname_init, module_init) in zip(model_ticket.named_modules(), model_init.named_modules()):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        module_init.weight_mask = module.weight_mask.clone()

torch.save(model_init.state_dict(), f'./saves/ticket_prate{1 - (1 - PRUNE_RATE) * (1 - float(SOURCE_PRUNE_RATE)):.4f}.pth')