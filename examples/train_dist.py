from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, validate, train_dist
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset, get_CIFAR10_dataset_dist
from shelf.models.resnet import ResNet152

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torch.distributed as dist
import torch.multiprocessing as mp

import os


def print_root(msg):
    if dist.get_rank() == 0:
        print(msg)

def print_rank(msg):
    print(f'Rank {dist.get_rank()}: {msg}')


class MyModel(nn.Module):
    def __init__(self, input_size=28, input_channel=1, num_output=10):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.num_output = num_output
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channel, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * (self.input_size // 2) * (self.input_size // 2), self.num_output)
        )

    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# hyperparameters
ModelClass = MyModel

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SMOOTHING = 5e-3
MOMENTUM = 0.0
DAMPENING = 0
WEIGHT_DECAY = 5e-4
NESTEROV = False

NUM_CLASSES = 10

DEVICE = 'cuda'
NUM_PROCESS = 8

# model, criterion, optimizer

print('Hyperparameters:')
print(f'>> EPOCHS: {EPOCHS}')
print(f'>> BATCH_SIZE: {BATCH_SIZE}')
print(f'>> LEARNING_RATE: {LEARNING_RATE}')
print(f'>> SMOOTHING: {SMOOTHING}')
print(f'>> MOMENTUM: {MOMENTUM}')
print(f'>> DAMPENING: {DAMPENING}')
print(f'>> WEIGHT_DECAY: {WEIGHT_DECAY}')
print(f'>> NESTEROV: {NESTEROV}')
print(f'>> NUM_CLASSES: {NUM_CLASSES}')
print(f'>> DEVICE: {DEVICE}')
print(f'>> NUM_WORKERS: {NUM_PROCESS}')


def run(rank, size):
    model = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES)
    model = model.cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_root(model)
    print_root(f'>> Number of parameters: {num_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0

    # load dataset
    _, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)
    train_loader, _ = get_CIFAR10_dataset_dist(batch_size=BATCH_SIZE, num_process=NUM_PROCESS)


    print_root(f'========== Train with FO: {ModelClass.__name__} ==========')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    for epoch in range(EPOCHS):
        epoch_lr = scheduler.get_last_lr()[0]

        # train for one epoch
        train_acc, train_loss = train_dist(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        # step scheduler
        scheduler.step()

        # print training/validation statistics
        print_root(
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
    print_root(f'>> Best Validation Accuracy {best_val_acc:.3f}')


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    

if __name__ == '__main__':
    size = NUM_PROCESS
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        print(f'Process {rank} started')

    for rank in range(size):
        p.join()
        print(f'Process {rank} joined')
    
    print('All processes joined')
    