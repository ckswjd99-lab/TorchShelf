import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from tqdm import tqdm
import os

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.resnet.etc import resnet20
from shelf.pruners.scoring import get_grasp_score

DEVICE = 'cuda'
NUM_CLASSES = 10

model_init_path = './saves/initial_weights.pth'

model = resnet20().to(DEVICE)


ntk_opt_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}
model.eval()
model.load_state_dict(torch.load(model_init_path))
criterion = nn.CrossEntropyLoss().to(DEVICE)

train_loader, val_loader = get_CIFAR10_dataset(root='../data', augmentation=False)

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)


def get_outputwise_direction(input, label, model, num_output=10):
    directions = [{pname: torch.zeros_like(param) for pname, param in model.named_parameters()} for _ in range(num_output)]

    output = model(input)
    for data_idx in tqdm(range(input.size(0)), leave=False):
        for output_idx in range(num_output):
            model.zero_grad()

            output[data_idx, output_idx].backward(retain_graph=True)
            for pname, param in model.named_parameters():
                if output_idx == label[data_idx]:
                    directions[output_idx][pname] += param.grad
                else:
                    directions[output_idx][pname] -= param.grad

    for output_idx in range(num_output):
        for pname in directions[output_idx]:
            directions[output_idx][pname] /= input.size(0)

    return directions


outputwise_directions = [{pname: torch.zeros_like(param) for pname, param in model.named_parameters()} for _ in range(NUM_CLASSES)]


num_steps = 1

for input, label in tqdm(train_loader):
    input, label = input.to(DEVICE), label.to(DEVICE)

    outputwise_directions_temp = get_outputwise_direction(input, label, model)

    for output_idx in range(NUM_CLASSES):
        for pname in outputwise_directions[output_idx]:
            outputwise_directions[output_idx][pname] += outputwise_directions_temp[output_idx][pname]

    num_steps -= 1
    if num_steps == 0:
        break



for output_idx in range(NUM_CLASSES):
    for pname in outputwise_directions[output_idx]:
        outputwise_directions[output_idx][pname] /= len(train_loader)

# save outputwise_directions
torch.save(outputwise_directions, './saves/init_outputwise_directions.pth')
