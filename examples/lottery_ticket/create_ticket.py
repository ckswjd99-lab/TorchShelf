from import_shelf import shelf
from shelf.models.resnet.etc import resnet20
from shelf.trainers import train, validate
from shelf.dataloaders import get_CIFAR10_dataset

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.utils.prune as prune

# hyperparameters
ModelClass = resnet20

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DEVICE = 'cuda'
INIT_MODEL = './saves/initial_model.pth'
BEST_MODEL = './saves/best_model_vacc0.9087.pth'
PRUNE_RATE = 0.1

# load dataset
train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=BATCH_SIZE, augmentation=False)

model_best = ModelClass().to(DEVICE)
model_best.load_state_dict(torch.load(BEST_MODEL))

model_init = ModelClass().to(DEVICE)
model_init.load_state_dict(torch.load(INIT_MODEL))

# l1 prune the best model
for mname, module in model_best.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=PRUNE_RATE)

# zero prune the init model
for mname, module in model_init.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.0)

# copy the masks from the best model to the init model
for (mname_best, module_best), (mname_init, module_init) in zip(model_best.named_modules(), model_init.named_modules()):
    if isinstance(module_best, nn.Linear) or isinstance(module_best, nn.Conv1d) or isinstance(module_best, nn.Conv2d):
        module_init.weight_mask = module_best.weight_mask.clone()

# save the pruned model
torch.save(model_init.state_dict(), f'./saves/ticket_prate{PRUNE_RATE}.pth')