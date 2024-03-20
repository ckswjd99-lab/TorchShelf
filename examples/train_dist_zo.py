from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, validate, train_dist_zo
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset
from shelf.models.transformer import VisionTransformer

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


# hyperparameters
EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 1e-4

SMOOTHING = 1e-3
NUM_QUERY = 4

IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM_HIDDEN = 512
DEPTH = 4
NUM_HEADS = 6
DIM_MLP = 256
DROPOUT = 0.1
EMB_DROPOUT = 0.1

NUM_CLASSES = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 1
PATH_MODEL = './saves/train_dist/model.pth'


def run(rank, size):
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM_HIDDEN,
        depth=DEPTH,
        heads=NUM_HEADS,
        mlp_dim=DIM_MLP,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT
    )
    model = model.cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_root(model)
    print_root(f'>> Number of parameters: {num_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0

    # load dataset
    train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)


    print_root(f'========== Train with FO: {VisionTransformer.__name__} ==========')

    for epoch in range(EPOCHS):
        
        epoch_lr = scheduler.get_last_lr()[0]

        # train for one epoch
        train_acc, train_loss = train_dist_zo(
            train_loader, model, criterion, optimizer, epoch,
            smoothing=SMOOTHING, query=NUM_QUERY//NUM_WORKERS, lr_auto=True, lr_max=LEARNING_RATE, lr_min=1e-5, ge_type='rge'
        )

        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, verbose=(rank == 0))

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
    
    print('Hyperparameters:')
    print(f'>> EPOCHS: {EPOCHS}')
    print(f'>> BATCH_SIZE: {BATCH_SIZE}')
    print(f'>> LEARNING_RATE: {LEARNING_RATE}')
    print(f'>> SMOOTHING: {SMOOTHING}')
    print(f'>> NUM_QUERY: {NUM_QUERY}')
    print(f'>> IMAGE_SIZE: {IMAGE_SIZE}')
    print(f'>> PATCH_SIZE: {PATCH_SIZE}')
    print(f'>> DIM_HIDDEN: {DIM_HIDDEN}')
    print(f'>> DEPTH: {DEPTH}')
    print(f'>> NUM_HEADS: {NUM_HEADS}')
    print(f'>> DIM_MLP: {DIM_MLP}')
    print(f'>> DROPOUT: {DROPOUT}')
    print(f'>> EMB_DROPOUT: {EMB_DROPOUT}')
    print(f'>> NUM_CLASSES: {NUM_CLASSES}')
    print(f'>> DEVICE: {DEVICE}')
    print(f'>> NUM_WORKERS: {NUM_WORKERS}')
    print(f'>> PATH_MODEL: {PATH_MODEL}')
    
    size = NUM_WORKERS
    
    processes = []
    
    mp.set_start_method("spawn")
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
        print(f'Process {rank} started')

    for rank in range(size):
        p = processes[rank]
        p.join()
        print(f'Process {rank} joined')
    
    print('All processes joined')
    