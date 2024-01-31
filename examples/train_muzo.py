from import_shelf import shelf
from shelf.trainers import adjust_learning_rate, train, train_zo_rge, train_zo_cge, validate
from shelf.dataloaders import get_MNIST_dataset, get_CIFAR10_dataset
from shelf.models.mutable import mutate_linear_kaiming, mutate_conv2d_kaiming, mutate_batchnorm2d_identity

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data


class MyModel(nn.Module):
    def __init__(self, input_size=28, input_channel=1, num_output=10):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.num_output = num_output

        self.inter_feature = 4

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channel, self.inter_feature, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_feature),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.inter_feature * (self.input_size // 2) * (self.input_size // 2), self.num_output)
        )

        self.growth = 0

    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def grow(self):
        self.growth += 1
        self.features[0] = mutate_conv2d_kaiming(self.features[0], self.input_channel, self.inter_feature + 1 * self.growth)
        self.features[1] = mutate_batchnorm2d_identity(self.features[1], self.inter_feature + 1 * self.growth)
        self.classifier[0] = mutate_linear_kaiming(
            self.classifier[0], 
            (self.inter_feature + 1 * self.growth) * (self.input_size // 2) * (self.input_size // 2), 
            self.num_output
        )


# hyperparameters
ModelClass = MyModel

EPOCHS = 5000
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SMOOTHING = 5e-3
MOMENTUM = 0.0
DAMPENING = 0
WEIGHT_DECAY = 5e-4
NESTEROV = False
NUM_QUERY = 1024
GROW_FREQ = 10

NUM_CLASSES = 10

DEVICE = 'cuda'

# model, criterion, optimizer
# model_vgg = ModelClass(input_size=28, input_channel=1, num_output=NUM_CLASSES)
model_vgg = ModelClass(input_size=32, input_channel=3, num_output=NUM_CLASSES)
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
    train_acc, train_loss = train_zo_rge(train_loader, model_vgg, criterion, optimizer, epoch, smoothing=SMOOTHING, query=NUM_QUERY)

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

    if epoch % GROW_FREQ == GROW_FREQ - 1:
        model_vgg.grow()
        model_vgg = model_vgg.cuda()

        optimizer = torch.optim.SGD(model_vgg.parameters(), LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        for _ in range(epoch + 1):
            scheduler.step()

        num_params = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
        print(f'grown to {num_params} parameters')

    

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')
print('')


print(f'========== Train with FO: {ModelClass.__name__} ==========')

POST_EPOCH = 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=POST_EPOCH, eta_min=1e-5)

for epoch in range(POST_EPOCH):
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
            epoch + 1, POST_EPOCH, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

    # record best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc

# result
print(f'>> Best Validation Accuracy {best_val_acc:.3f}')