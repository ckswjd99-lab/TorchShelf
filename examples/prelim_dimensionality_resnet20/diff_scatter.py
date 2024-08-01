import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'

model1 = MLPMixer().to(DEVICE)
model2 = MLPMixer().to(DEVICE)

model_path_1 = './saves/set1_default/initial_weights.pth'
model_path_2 = './saves/set1_default/epoch_100.pth'

model1.load_state_dict(torch.load(model_path_1))
model2.load_state_dict(torch.load(model_path_2))

model1_state_dict = dict(model1.state_dict())
model2_state_dict = dict(model2.state_dict())

model1_weights = {pname: param.view(-1).cpu() for pname, param in model1_state_dict.items() if 'num_batches_tracked' not in pname}
model2_weights = {pname: param.view(-1).cpu() for pname, param in model2_state_dict.items() if 'num_batches_tracked' not in pname}

# plot the movement
# x is the initial weight, y is the final weight
pbar = tqdm(model1_weights.keys())
for pname in pbar:
    plt.figure(figsize=(8, 8))
    pbar.set_description(pname)
    plt.scatter(model1_weights[pname], model2_weights[pname], label=pname, s=1, alpha=0.5)
    # hline and vline at 0
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.xlabel('Initial weight')
    plt.ylabel('Final weight')
    plt.title('Weight movement')
    plt.savefig('./logs/difference_scatter/{}.png'.format(pname))