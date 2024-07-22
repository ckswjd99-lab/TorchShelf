import torch
import torch.nn as nn

import torchsummary
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)


model_init = MLPMixer().to(DEVICE)
model_init.load_state_dict(torch.load('./saves/set1_default/initial_weights.pth'))

model = MLPMixer().to(DEVICE)
model.load_state_dict(torch.load('./saves/set1_default/final_weights_vloss0.788.pth'))

weights_diff = {pname: model_init.state_dict()[pname] - model.state_dict()[pname] for pname in model_init.state_dict()}
norm_diff = torch.norm(torch.cat([param.view(-1) for param in weights_diff.values()]).abs(), p=2)


criterion = nn.CrossEntropyLoss()

distance_from_optimal = 1e+2

loss_orig_sum = 0
loss_perturbed_sum = 0

with torch.no_grad():
    for input, label in train_loader:
        input, label = input.to(DEVICE), label.to(DEVICE)
        loss_orig = criterion(model(input), label)

        pnoises = {pname: torch.randn_like(param) for pname, param in model.state_dict().items()}
        pnorm = torch.norm(torch.cat([param.view(-1) for param in pnoises.values()]).abs(), p=2)
        pnoises = {pname: pnoise / pnorm * distance_from_optimal for pname, pnoise in pnoises.items()}

        for pname, param in model.named_parameters():
            param.data += pnoises[pname]
        
        loss_perturbed = criterion(model(input), label)

        weights_diff = {pname: model_init.state_dict()[pname] - model.state_dict()[pname] for pname in model_init.state_dict()}
        norm_diff = torch.norm(torch.cat([param.view(-1) for param in weights_diff.values()]).abs(), p=2)

        print(f"Original Loss: {loss_orig:.4f}, Perturbed Loss: {loss_perturbed:.4f}, Distance: {norm_diff:.4f}")

        for pname, param in model.named_parameters():
            param.data -= pnoises[pname]

        loss_orig_sum += loss_orig
        loss_perturbed_sum += loss_perturbed

# pring avg
loss_orig_avg = loss_orig_sum / len(train_loader)
loss_perturbed_avg = loss_perturbed_sum / len(train_loader)
print(f"Norm of the difference from init to final: {norm_diff:.4f}")
print(f"Average Original Loss: {loss_orig_avg:.4f}, Average Perturbed Loss: {loss_perturbed_avg:.4f}, Increase: {loss_perturbed_avg - loss_orig_avg:.4f} ({(loss_perturbed_avg - loss_orig_avg) / loss_orig_avg * 100:.2f}%)")
