from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, train_zo_rge, train_zo_rge_autolr, train_zo_cge, validate
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset
from shelf.models.mutable import MutableResNet18

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import numpy as np
import math
import os
import time


# hyperparameters
ModelClass = MutableResNet18

EPOCHS = 50000
BATCH_SIZE = 128
LEARNING_RATE = 'auto'
LR_CONFIDENCE = 1.0
LR_MAX = 2e-1
SMOOTHING = 5e-3
MOMENTUM = 0.0
DAMPENING = 0.0
WEIGHT_DECAY = 5e-4
NESTEROV = False

QUERY_RATIO = 0.5
QUERY_GROWTH = 1.0

HYPARAM_UPDATE_FREQ = 200
TOLERANCE = 1

now_growth = 8
TOTAL_GROWTH = 512

ACC_MILESTONES = [
    *np.linspace(0.1000, 0.4114, 8),
    *np.linspace(0.4114, 0.5329, 16 - 8),
    *np.linspace(0.5329, 0.6483, 32 - 16),
    *np.linspace(0.6483, 0.7529, 64 - 32),
    *np.linspace(0.7529, 0.8201, 128 - 64),
    *np.linspace(0.8201, 0.8617, 256 - 128),
    *np.linspace(0.8617, 0.8919, 512 - 256),
]


NUM_CLASSES = 10

DEVICE = 'cuda'

# model, criterion, optimizer
model_resnet = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES, shrink_ratio=now_growth/TOTAL_GROWTH)
model_resnet = model_resnet.cuda()
num_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
print(model_resnet)
print(f'>> Number of parameters: {num_params} (growth: {now_growth}/{TOTAL_GROWTH})')

print('Hyperparameters:')
print(f'>> EPOCHS: {EPOCHS}')
print(f'>> BATCH_SIZE: {BATCH_SIZE}')
print(f'>> LEARNING_RATE: {LEARNING_RATE}')
print(f'>> LR_CONFIDENCE: {LR_CONFIDENCE}')
print(f'>> LR_MAX: {LR_MAX}')
print(f'>> SMOOTHING: {SMOOTHING}')
print(f'>> MOMENTUM: {MOMENTUM}')
print(f'>> DAMPENING: {DAMPENING}')
print(f'>> WEIGHT_DECAY: {WEIGHT_DECAY}')
print(f'>> NESTEROV: {NESTEROV}')
print(f'>> QUERY_GROWTH: {QUERY_GROWTH}')
print(f'>> HYPARAM_UPDATE_FREQ: {HYPARAM_UPDATE_FREQ}')
print(f'>> TOLERANCE: {TOLERANCE}')
print(f'>> NUM_CLASSES: {NUM_CLASSES}')
print(f'>> DEVICE: {DEVICE}')

criterion = nn.CrossEntropyLoss()

last_best_val_acc = 0
best_val_acc = 0

last_best_val_loss = 1e99
best_val_loss = 1e99

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

# logs
log = []


print(f'========== Train with ZO: {ModelClass.__name__} ==========')

for now_growth in range(8, TOTAL_GROWTH):

    model_resnet.grow_tobe(now_growth / TOTAL_GROWTH)
    num_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)

    now_num_query = int(num_params * QUERY_RATIO)
    print(f'>> Query: {now_num_query}')

    now_confience = LR_CONFIDENCE

    optimizer = torch.optim.SGD(model_resnet.parameters(), 1e-3, momentum=MOMENTUM, dampening=DAMPENING, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)

    print(f'>> Growth: {now_growth}/{TOTAL_GROWTH} ({num_params} parameters)')
    log.append(f'>> Growth: {now_growth}/{TOTAL_GROWTH}')

    # train until milestone
    target_acc = ACC_MILESTONES[now_growth] * 0.9
    print(f'>> Target Accuracy: {target_acc:.3f}')
    log.append(f'>> Target Accuracy: {target_acc:.3f}')

    epoch = 1
    run = True
    while best_val_acc < target_acc and run:
        start_time = time.time()

        train_config = {}
        train_acc, train_loss = train_zo_rge_autolr(
            train_loader, model_resnet, criterion, optimizer, epoch-1, 
            max_lr=LR_MAX, smoothing=SMOOTHING, query=now_num_query, confidence=now_confience, clip_loss_diff=1e-1,
            config=train_config
        )

        val_acc, val_loss = validate(val_loader, model_resnet, criterion, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        now_lr_avg = train_config['avg_lr']
        now_lr_std = train_config['std_lr']

        logstring = 'Epoch: {0}\t' \
                    'LR: {lr:.6f}±{lr_std:.6f}\t' \
                    'Confidence: {confidence:.6f}\t' \
                    'Train Accuracy {train_acc:.3f}\t' \
                    'Train Loss {train_loss:.3f}\t' \
                    'Val Accuracy {val_acc:.3f}\t' \
                    'Val Loss {val_loss:.3f}\t' \
                    'Time Elap {time:.3f}'.format(
            epoch, 
            lr=now_lr_avg, lr_std=now_lr_std, confidence=now_confience, 
            train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss,
            time=time.time() - start_time
        )


        print(logstring)
        log.append(logstring)
        
        if epoch % HYPARAM_UPDATE_FREQ == 0:
            now_num_query = int(now_num_query * QUERY_GROWTH)
            print(f'>> Query increased to: {now_num_query}')
            log.append(f'>> Query increased to: {now_num_query}')

        if epoch > HYPARAM_UPDATE_FREQ * TOLERANCE:
            print('>> Unable to reach target accuracy, continue to grow.')
            log.append('>> Unable to reach target accuracy, continue to grow.')
            run = False
        
        epoch += 1

    print(f'>> Reached Accuracy: {best_val_acc:.3f}')
    log.append(f'>> Reached Accuracy: {best_val_acc:.3f}')

    if not os.path.exists('./saves/train_muzo_resnet'):
        os.makedirs('./saves/train_muzo_resnet')

    torch.save(
        {
            'model': model_resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'now_growth': now_growth,
            'log': log,
        },
        f'./saves/train_muzo_resnet/acc{best_val_acc*100:.2f}_loss{best_val_loss:.3f}_growth{now_growth}.pth'
    )



# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')

