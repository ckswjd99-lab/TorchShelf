from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, train_zeroth_order, validate
from shelf.dataloaders import get_MNIST_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, input_size=28, input_channel=1, num_output=10):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.num_output = num_output
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * input_size * input_channel, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_output)
        )

    
    def forward(self, x):
        x = self.features(x)
        return x


# hyperparameters
ModelClass = MyModel

EPOCHS_PRETRAIN = 0
LR_PRETRAIN = 0.05

EPOCHS = 50
BATCH_SIZE = 128
LR_PERTURB = 1e-3
PERTURB_EPS = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 10

DEVICE = 'cuda'

# model, criterion, optimizer
# model_vgg = ModelClass()
model_vgg = ModelClass(input_size=28, input_channel=1, num_output=NUM_CLASSES)
model_vgg = model_vgg.cuda()
num_params = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
print(f'>> Number of parameters: {num_params}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_vgg.parameters(), LR_PRETRAIN, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

best_val_acc = 0

# load dataset
train_loader, val_loader = get_MNIST_dataset(batch_size=BATCH_SIZE)


# pretrain
print(f'========== Pretrain with E2EBP: {ModelClass.__name__} ==========')

for epoch in range(EPOCHS_PRETRAIN):
    epoch_lr = adjust_learning_rate(optimizer, LR_PRETRAIN, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

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


print(f'========== Train with ZO: {ModelClass.__name__} ==========')


for epoch in range(EPOCHS_PRETRAIN, EPOCHS):
    # epoch_lr = adjust_learning_rate(None, LR_PERTURB, epoch, 70, 0.5, minimum_lr=1e-4)
    epoch_lr = LR_PERTURB

    # train for one epoch
    train_acc, train_loss = train_zeroth_order(train_loader, model_vgg, criterion, epoch, learning_rate=epoch_lr, weight_decay=WEIGHT_DECAY, epsilon=PERTURB_EPS, momentum=MOMENTUM)

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