import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.resnet.etc import resnet20

DEVICE = 'cuda'

model1 = resnet20().to(DEVICE)
model2 = resnet20().to(DEVICE)

model_path_1 = './saves/initial_weights.pth'
model_path_2 = './saves/epoch_200.pth'

model1.load_state_dict(torch.load(model_path_1))
model2.load_state_dict(torch.load(model_path_2))

model1_state_dict = dict(model1.state_dict())
model2_state_dict = dict(model2.state_dict())

model1_weights = {pname: param.clone() for pname, param in model1_state_dict.items() if 'num_batches_tracked' not in pname}
model2_weights = {pname: param.clone() for pname, param in model2_state_dict.items() if 'num_batches_tracked' not in pname}
weights_difference = {pname: model2_weights[pname] - model1_weights[pname] for pname in model1_weights}
weights_difference_flat = torch.cat([param.view(-1) for param in weights_difference.values()]).abs()

weights_ratio = {pname: model2_weights[pname] / model1_weights[pname] for pname in model1_weights}
weights_ratio_flat = torch.cat([param.view(-1) for param in weights_ratio.values()])

# for pname in model1_weights:
#     print(f'{pname} ({model1_weights[pname].numel()}) : {weights_difference[pname].abs().float().mean()}')

# plot the difference
plt.figure()
plt.scatter(range(len(weights_difference_flat)), weights_difference_flat.cpu().numpy(), s=1)
plt.xlabel('Parameter Index')
plt.ylabel('Difference')
plt.yscale('log')
plt.title('Difference')
plt.savefig(f'./logs/difference_{model_path_1.split("/")[-1][:-4]}_{model_path_2.split("/")[-1][:-4]}.png')
plt.close()

# plot the distribution
weights_difference_sorted = torch.sort(weights_difference_flat, descending=True).values
plt.figure()
plt.plot(weights_difference_sorted.cpu().numpy())
plt.xlabel('Parameter Index')
plt.ylabel('Difference')
plt.yscale('log')
plt.title('Difference Distribution')
plt.savefig(f'./logs/difference_distribution_{model_path_1.split("/")[-1][:-4]}_{model_path_2.split("/")[-1][:-4]}.png')
plt.close()

# plot the ratio
weights_ratio_sorted = torch.sort(weights_ratio_flat, descending=True).values
plt.figure()
plt.plot(weights_ratio_sorted.cpu().numpy())
# horizontal line at 1, -1
plt.axhline(1, color='red', linestyle='--')
plt.axhline(-1, color='red', linestyle='--')
plt.xlabel('Parameter Index')
plt.ylabel('Ratio')
plt.ylim(-10, 10)
plt.title('Ratio Distribution')
plt.savefig(f'./logs/ratio_distribution_{model_path_1.split("/")[-1][:-4]}_{model_path_2.split("/")[-1][:-4]}.png')
plt.close()