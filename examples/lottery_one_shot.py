import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from import_shelf import shelf
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset
from shelf.trainers import train, validate, adjust_learning_rate
from shelf.models.vgg import VGG6_custom
from shelf.models.resnet import ResNet34
    

# ModelClass = MyModel
ModelClass = ResNet34

# hyperparams
EPOCHS_PRETRAIN = 10
EPOCHS = 100
LR = 0.05
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SURVIVE_RATIO = 0.005

NUM_CLASSES = 10

DEVICE = 'cuda'

# data
train_loader, val_loader = get_CIFAR10_dataset(root='./data', batch_size=BATCH_SIZE)

# model, criterion, optimizer
model = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES)
model = model.cuda()

model_init = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES)
model_init = model_init.cuda()
model_init.load_state_dict(model.state_dict())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# find lottery ticket
print(f'========== Find Lottery Ticket: {ModelClass.__name__} ==========')

for epoch in range(EPOCHS_PRETRAIN):
    epoch_lr = adjust_learning_rate(optimizer, LR, epoch, epoch_freq=30)

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(
        'Epoch: {0}/{1}\t'
        'LR: {lr:.6f}\t'
        'Train Accuracy {train_acc:.3f}\t'
        'Train Loss {train_loss:.3f}\t'
        'Val Accuracy {val_acc:.3f}\t'
        'Val Loss {val_loss:.3f}'
        .format(
            epoch + 1, EPOCHS_PRETRAIN, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

# prune model
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=(1-SURVIVE_RATIO))

# print original number of parameters and pruned number of parameters
num_params = 0
num_pruned_params = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        num_params += module.weight.numel()
        num_pruned_params += module.weight.nelement() - module.weight.nonzero().size(0)

print(f'>> Pruned {num_pruned_params} parameters, originally {num_params}, now {num_params - num_pruned_params} ({(num_params - num_pruned_params) / num_params * 100:.2f}%)')

# copy masks to model_init
num_params_init = 0
num_pruned_params_init = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        module_init = dict(model_init.named_modules())[name]
        prune.custom_from_mask(module_init, name='weight', mask=module.weight_mask)

for name, module_init in model_init.named_modules():
    if isinstance(module_init, nn.Conv2d) or isinstance(module_init, nn.Linear):
        num_params_init += module_init.weight.numel()
        num_pruned_params_init += module_init.weight.nelement() - module_init.weight.nonzero().size(0)

print(f'>> Pruned {num_pruned_params_init} parameters, originally {num_params_init}, now {num_params_init - num_pruned_params_init} ({(num_params_init - num_pruned_params_init) / num_params_init * 100:.2f}%)')



# train pruned model
print(f'========== Train Lottery Ticket: {ModelClass.__name__} ==========')

optimizer = optim.SGD(model_init.parameters(), LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# check initial accuracy
val_acc, val_loss = validate(val_loader, model_init, criterion, 0)
print(f'>> Initial accuracy: {val_acc:.3f}')

for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer, LR, epoch, epoch_freq=30)

    train_acc, train_loss = train(train_loader, model_init, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model_init, criterion, epoch)

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

