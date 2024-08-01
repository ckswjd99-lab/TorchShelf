import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'


train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

model1 = MLPMixer().to(DEVICE)
model2 = MLPMixer().to(DEVICE)

model_path_1 = './saves/set1/initial_weights.pth'
model_path_2 = './saves/set1/epoch_100.pth'

model1.load_state_dict(torch.load(model_path_1))
model2.load_state_dict(torch.load(model_path_2))

model1_state_dict = dict(model1.state_dict())
model2_state_dict = dict(model2.state_dict())

model1_weights = {pname: param.clone() for pname, param in model1_state_dict.items()}
model2_weights = {pname: param.clone() for pname, param in model2_state_dict.items()}
weights_difference = {pname: model2_weights[pname] - model1_weights[pname] for pname in model1_weights}


model1.eval()
model2.eval()

# Compute the line between the two models
num_points = 100
losses = []
for alpha in torch.linspace(0, 1, num_points):
    new_weights = {pname: model1_weights[pname] + alpha * weights_difference[pname] for pname in model1_weights}
    model1.load_state_dict(new_weights)
    model1.eval()

    val_acc, val_loss = validate(train_loader, model1, nn.CrossEntropyLoss(), 0)
    print(f'Alpha: {alpha:.4f}, Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}')

    losses.append(val_loss)

# Plot the line between the two models
plt.figure()
plt.plot(torch.linspace(0, 1, num_points), losses)
plt.xlabel('Alpha')
plt.ylabel('Validation Loss')
plt.title('Line Between Two Models')
plt.savefig(f'./logs/line_between_{model_path_1.split("/")[-1][:-4]}_{model_path_2.split("/")[-1][:-4]}.png')
