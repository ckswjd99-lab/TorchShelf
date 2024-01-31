from import_shelf import shelf
from shelf.models.transformer import VisionTransformer
from shelf.trainers import adjust_learning_rate, train, validate
from shelf.dataloaders import get_CIFAR10_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

# hyperparameters

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1

NUM_CLASSES = 10

PRETRAIN_EPOCH = 50
TRANSFER_EPOCH = 5

DEVICE = 'cuda'

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

print(f'========== From Scratch: ViT ==========')

# model, criterion, optimizer
model = VisionTransformer(image_size=32, patch_size=4, num_classes=NUM_CLASSES, dim=768, depth=12, heads=12, mlp_dim=768*4, dropout=0.0)
model = model.cuda()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'>> Number of parameters: {num_params}')

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = scheduler.get_last_lr()[0]

    # train for one epoch
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

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

    # record best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')

