from import_shelf import shelf
from shelf.models.vgg import VGG11, VGG11_bn, VGG13, VGG13_bn, VGG16, VGG16_bn, VGG19, VGG19_bn
from shelf.trainers import adjust_learning_rate, train, validate
from shelf.dataloaders import get_CIFAR100_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torchvision.models as models

# hyperparameters
ModelClass = VGG19_bn

EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 100

PRETRAIN_EPOCH = 50
TRANSFER_EPOCH = 5

DEVICE = 'cuda'

# load dataset
train_loader, val_loader = get_CIFAR100_dataset(batch_size=BATCH_SIZE)

print(f'========== From Scratch: {ModelClass.__name__} ==========')

# model, criterion, optimizer
model_vgg = ModelClass(input_size=32, num_output=NUM_CLASSES)
model_vgg = model_vgg.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_vgg.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer, LEARNING_RATE, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_vgg, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_vgg, criterion, epoch)

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