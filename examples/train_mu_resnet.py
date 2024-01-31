from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, train_zo_rge, train_zo_cge, validate
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset
from shelf.mutators import mutate_linear_kaiming, mutate_conv2d_kaiming, mutate_batchnorm2d_identity
from shelf.models.mutable import MutableResNet18

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data


# hyperparameters
ModelClass = MutableResNet18

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SMOOTHING = 5e-3
MOMENTUM = 0.0
DAMPENING = 0
WEIGHT_DECAY = 5e-4
NESTEROV = False
NUM_QUERY = 16


now_growth = 32
TOTAL_GROWTH = 32
GROW_FREQ = 3


NUM_CLASSES = 10

DEVICE = 'cuda'

# model, criterion, optimizer
# model_vgg = ModelClass(input_size=28, input_channel=1, num_output=NUM_CLASSES)
model_vgg = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES, shrink_ratio=now_growth/TOTAL_GROWTH)
model_vgg = model_vgg.cuda()
num_params = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
print(model_vgg)
print(f'>> Number of parameters: {num_params}')

print('Hyperparameters:')
print(f'>> EPOCHS: {EPOCHS}')
print(f'>> BATCH_SIZE: {BATCH_SIZE}')
print(f'>> LEARNING_RATE: {LEARNING_RATE}')
print(f'>> SMOOTHING: {SMOOTHING}')
print(f'>> MOMENTUM: {MOMENTUM}')
print(f'>> DAMPENING: {DAMPENING}')
print(f'>> WEIGHT_DECAY: {WEIGHT_DECAY}')
print(f'>> NESTEROV: {NESTEROV}')
print(f'>> NUM_QUERY: {NUM_QUERY}')
print(f'>> NUM_CLASSES: {NUM_CLASSES}')
print(f'>> GROW_FREQ: {GROW_FREQ}')
print(f'>> DEVICE: {DEVICE}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_vgg.parameters(), LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
# optimizer = torch.optim.Adam(model_vgg.parameters(), LR_PERTURB, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_acc = 0

# load dataset
# train_loader, val_loader = get_MNIST_dataset(batch_size=BATCH_SIZE)
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)


print(f'========== Train with ZO: {ModelClass.__name__} ==========')

for epoch in range(EPOCHS):
    epoch_lr = scheduler.get_last_lr()[0]

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_vgg, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_vgg, criterion, epoch)

    # step scheduler
    scheduler.step()

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

    if epoch % GROW_FREQ == GROW_FREQ - 1 and now_growth < TOTAL_GROWTH:
        now_growth += 1
        model_vgg.grow_tobe(now_growth / TOTAL_GROWTH)

        optimizer = torch.optim.SGD(model_vgg.parameters(), LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        for _ in range(epoch + 1):
            scheduler.step()

        num_params = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
        print(f'grown to {num_params} parameters ({now_growth}/{TOTAL_GROWTH})')

    

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')

