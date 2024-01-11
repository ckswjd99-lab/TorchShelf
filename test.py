import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models

from tqdm import tqdm
import os

from shelf.utils import adjust_learning_rate, train, validate
from shelf.dataloaders import get_CIFAR10_dataset

# hyperparameters
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

PRETRAIN_EPOCH = 50
TRANSFER_EPOCH = 5

DEVICE = 'cuda'
MODEL_SAVE_DIR = './save'
MODEL_PATH_RESNET50_PT = MODEL_SAVE_DIR + '/resnet50_e' + str(PRETRAIN_EPOCH) + '.pth'
MODEL_PATH_RESNET101_LL = './save/resnet101_ll.pth'
MODEL_PATH_RESNET101_FT = './save/resnet101_ft.pth'

if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)


print('========== From Scratch: ResNet50 ==========')
writer_resnet50_fs = SummaryWriter()

# model, criterion, optimizer
model_resnet50 = models.resnet50(weights=None)
model_resnet50.fc = nn.Linear(2048, 10)
model_resnet50 = model_resnet50.cuda()

criterion_resnet50 = nn.CrossEntropyLoss()

optimizer_resnet50 = torch.optim.SGD(model_resnet50.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer_resnet50, LEARNING_RATE, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_resnet50, criterion_resnet50, optimizer_resnet50, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_resnet50, criterion_resnet50, epoch)

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

    # write to tensorboard
    writer_resnet50_fs.add_scalar('train/accuracy', train_acc, epoch)
    writer_resnet50_fs.add_scalar('train/loss', train_loss, epoch)
    writer_resnet50_fs.add_scalar('val/accuracy', val_acc, epoch)
    writer_resnet50_fs.add_scalar('val/loss', val_loss, epoch)

    # record best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    # save model
    if (epoch + 1) == PRETRAIN_EPOCH:
        torch.save(model_resnet50.state_dict(), MODEL_SAVE_DIR + '/resnet50_e' + str(PRETRAIN_EPOCH) + '.pth')

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')


print('========== From Scratch: ResNet101 ==========')
writer_resnet101_fs = SummaryWriter()

# model, criterion, optimizer
model_resnet101 = models.resnet101(weights=None)
model_resnet101.fc = nn.Linear(2048, 10)
model_resnet101 = model_resnet101.cuda()

criterion_resnet101 = nn.CrossEntropyLoss()

optimizer_resnet101 = torch.optim.SGD(model_resnet101.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# train model
best_val_acc = 0

for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer_resnet101, LEARNING_RATE, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_resnet101, criterion_resnet101, optimizer_resnet101, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_resnet101, criterion_resnet101, epoch)

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

    # write to tensorboard
    writer_resnet101_fs.add_scalar('train/accuracy', train_acc, epoch)
    writer_resnet101_fs.add_scalar('train/loss', train_loss, epoch)
    writer_resnet101_fs.add_scalar('val/accuracy', val_acc, epoch)
    writer_resnet101_fs.add_scalar('val/loss', val_loss, epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')


print('========== reKD: ResNet50 -> ResNet101 ==========')
writer_cell_division = SummaryWriter()

# model
model_resnet50_pt = models.resnet50(weights=None).cuda()
model_resnet50.load_state_dict(torch.load(MODEL_PATH_RESNET50_PT))
model_resnet101 = models.resnet101(weights=None).cuda()

# copy weights
model_resnet101.conv1 = model_resnet50.conv1
model_resnet101.bn1 = model_resnet50.bn1
model_resnet101.relu = model_resnet50.relu
model_resnet101.maxpool = model_resnet50.maxpool
model_resnet101.layer1 = model_resnet50.layer1
model_resnet101.layer2 = model_resnet50.layer2
model_resnet101.layer3 = model_resnet50.layer3
model_resnet101.fc = model_resnet50.fc

submodule_resnet50 = model_resnet50.layer4
submodule_resnet101 = model_resnet101.layer4

# criterion, optimizer
criterion_resnet101_kd = nn.MSELoss()
criterion_resnet101_val = nn.CrossEntropyLoss()
optimizer_resnet101 = torch.optim.SGD(submodule_resnet101.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

transfer_epoch = 5
for epoch in range(PRETRAIN_EPOCH, PRETRAIN_EPOCH + TRANSFER_EPOCH):
    epoch_lr = adjust_learning_rate(optimizer_resnet50, LEARNING_RATE, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

    # train for one epoch
    submodule_resnet101.train()
    submodule_resnet50.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        hidden = model_resnet50.conv1(input)
        hidden = model_resnet50.bn1(hidden)
        hidden = model_resnet50.relu(hidden)
        hidden = model_resnet50.maxpool(hidden)
        hidden = model_resnet50.layer1(hidden)
        hidden = model_resnet50.layer2(hidden)
        hidden_in = model_resnet50.layer3(hidden)
        hidden_out = submodule_resnet50(hidden_in)

        output = submodule_resnet101(hidden_in.detach())
        target = hidden_out.detach()

        loss = criterion_resnet101_kd(output, target)

        optimizer_resnet101.zero_grad()
        loss.backward()
        optimizer_resnet101.step()

        output = model_resnet101(input)

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)

        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        pbar.set_postfix(train_accuracy=accuracy, train_loss=avg_loss)
    
    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_resnet101, criterion_resnet101_val, epoch)

    # print training/validation statistics
    print(
        'Epoch: {0}/{1}\t'
        'LR: {lr:.6f}\t'
        'Train Accuracy {train_acc:.3f}\t'
        'Train Loss {train_loss:.3f}\t'
        'Val Accuracy {val_acc:.3f}\t'
        'Val Loss {val_loss:.3f}'
        .format(
            epoch + 1, PRETRAIN_EPOCH + TRANSFER_EPOCH, lr=epoch_lr, train_acc=accuracy, train_loss=avg_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

    # write to tensorboard
    writer_cell_division.add_scalar('train/accuracy', accuracy, epoch)
    writer_cell_division.add_scalar('train/loss', avg_loss, epoch)
    writer_cell_division.add_scalar('val/accuracy', val_acc, epoch)
    writer_cell_division.add_scalar('val/loss', val_loss, epoch)



print('========== Fine-Tune: ResNet101 ==========')

# optimizer
optimizer_resnet101_ft = torch.optim.SGD(model_resnet101.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# train model
for epoch in range(PRETRAIN_EPOCH + TRANSFER_EPOCH, EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer_resnet101_ft, LEARNING_RATE, epoch, 5, 0.2 ** (1/10), minimum_lr=0.0008)

    # train for one epoch
    train_acc, train_loss = train(train_loader, model_resnet101, criterion_resnet101_val, optimizer_resnet101_ft, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model_resnet101, criterion_resnet101_val, epoch)

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

    # write to tensorboard
    writer_cell_division.add_scalar('train/accuracy', train_acc, epoch)
    writer_cell_division.add_scalar('train/loss', train_loss, epoch)
    writer_cell_division.add_scalar('val/accuracy', val_acc, epoch)
    writer_cell_division.add_scalar('val/loss', val_loss, epoch)

# test model
test_acc, test_loss = validate(val_loader, model_resnet101, criterion_resnet101_val, epoch, verbose=False)
print(f'>> Test Accuracy {test_acc:.3f}\tTest Loss {test_loss:.3f}')
print('')

# save model
torch.save(model_resnet101.state_dict(), MODEL_PATH_RESNET101_FT)
