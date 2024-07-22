import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'
FIG_FOLDER = './logs/cossim/'

train_loader, val_loader = get_CIFAR10_dataset(batch_size=1024)

final_weight_path = './saves/set1/final_weights_vloss0.381.pth'

final_model = MLPMixer().to(DEVICE)

final_model.load_state_dict(torch.load(final_weight_path))

final_model_state_dict = dict(final_model.state_dict())
final_model_weights = {pname: param.clone() for pname, param in final_model_state_dict.items() if 'num_batches_tracked' not in pname}
final_model_weights_flat = torch.cat([param.view(-1) for param in final_model_weights.values()])

num_trials = 100
smoothing = 1e-3
vthv_values = []

criterion = nn.CrossEntropyLoss()

loss_orig = validate(val_loader, final_model, criterion, 0)[1]
with torch.no_grad():
    for i in tqdm(range(num_trials)):

        noise = {pname: torch.randn_like(param) for pname, param in final_model_weights.items()}
        
        for pname, param in final_model.named_parameters():
            param.data += smoothing * noise[pname]

        loss_pos = validate(val_loader, final_model, criterion, 0)[1]

        for pname, param in final_model.named_parameters():
            param.data -= 2 * smoothing * noise[pname]

        loss_neg = validate(val_loader, final_model, criterion, 0)[1]

        vthv = (loss_pos + loss_neg - 2 * loss_orig) / (smoothing ** 2)
        vthv_values.append(vthv)

    num_concave = sum([vthv < 0 for vthv in vthv_values])
    num_convex = sum([vthv > 0 for vthv in vthv_values])
    num_linear = sum([vthv == 0 for vthv in vthv_values])

    print(f'Concave: {num_concave}, Convex: {num_convex}, Linear: {num_linear}, Total {len(vthv_values)}')
