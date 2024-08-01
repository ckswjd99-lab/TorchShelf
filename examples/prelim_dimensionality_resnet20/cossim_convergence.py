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

model_paths = [f'./saves/set1_default/epoch_{i}.pth' for i in range(1, 100)]
initial_weight_path = './saves/set1_default/initial_weights.pth'
final_weight_path = './saves/set1_default/epoch_100.pth'

models = [MLPMixer().to(DEVICE) for _ in model_paths]
initial_model = MLPMixer().to(DEVICE)
final_model = MLPMixer().to(DEVICE)

for model, model_path in zip(models, model_paths):
    model.load_state_dict(torch.load(model_path))
initial_model.load_state_dict(torch.load(initial_weight_path))
final_model.load_state_dict(torch.load(final_weight_path))

model_state_dicts = [dict(model.state_dict()) for model in models]
model_weights = [{pname: param.clone() for pname, param in model_state_dict.items() if 'num_batches_tracked' not in pname} for model_state_dict in model_state_dicts]
model_weights_flat = [torch.cat([param.view(-1) for param in model_weight.values()]) for model_weight in model_weights]

initial_model_state_dict = dict(initial_model.state_dict())
initial_model_weights = {pname: param.clone() for pname, param in initial_model_state_dict.items() if 'num_batches_tracked' not in pname}
initial_model_weights_flat = torch.cat([param.view(-1) for param in initial_model_weights.values()])

final_model_state_dict = dict(final_model.state_dict())
final_model_weights = {pname: param.clone() for pname, param in final_model_state_dict.items() if 'num_batches_tracked' not in pname}
final_model_weights_flat = torch.cat([param.view(-1) for param in final_model_weights.values()])

distance_init_final = torch.norm(final_model_weights_flat - initial_model_weights_flat, p=2).item()

cos_sims_init = [
    torch.nn.functional.cosine_similarity(
        final_model_weights_flat - initial_model_weights_flat,
        model_weight_flat - initial_model_weights_flat, dim=0
    ).item()
    for model_weight_flat in model_weights_flat
]
cos_sims_init = [0] + cos_sims_init + [1]

cos_sims_final = [
    torch.nn.functional.cosine_similarity(
        initial_model_weights_flat - final_model_weights_flat,
        model_weight_flat - final_model_weights_flat, dim=0
    ).item()
    for model_weight_flat in model_weights_flat
]
cos_sims_final = [1] + cos_sims_final + [0]

cos_sims_training = [
    torch.nn.functional.cosine_similarity(
        model_weight_flat - initial_model_weights_flat,
        final_model_weights_flat - model_weight_flat, dim=0
    ).item()
    for model_weight_flat in model_weights_flat
]
cos_sims_training = [0] + cos_sims_training + [0]

diff_norms_from_final = [
    torch.norm(
        final_model_weights_flat - model_weight_flat, p=2
    ).item()
    for model_weight_flat in model_weights_flat
]
diff_norms_from_final = [distance_init_final] + diff_norms_from_final + [0]

diff_norms_from_init = [
    torch.norm(
        initial_model_weights_flat - model_weight_flat, p=2
    ).item()
    for model_weight_flat in model_weights_flat
]
diff_norms_from_init = [0] + diff_norms_from_init + [distance_init_final]

trajectory = [
    (diff_norm_from_init * cos_sim, diff_norm_from_init * math.sqrt(1 - min(cos_sim ** 2, 1)))
    for cos_sim, diff_norm_from_init in zip(cos_sims_init, diff_norms_from_init)
]
trajectory = [(0, 0)] + trajectory + [(torch.norm(final_model_weights_flat - initial_model_weights_flat, p=2).item(), 0)]

print(f"Cossim of epoch 21: {cos_sims_init[21]}")

# plot the cosine similarity (from initial)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 102), cos_sims_init)
plt.xlabel('Epoch')
plt.ylabel('Cosine Similarity (from Initial)')
plt.title('Cosine Similarity (from Initial)')
plt.savefig(f'{FIG_FOLDER}cosine_similarity_initial.png')
plt.close()

# plot the cosine similarity (from final)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 102), cos_sims_final)
plt.xlabel('Epoch')
plt.ylabel('Cosine Similarity (from Final)')
plt.title('Cosine Similarity (from Final)')
plt.savefig(f'{FIG_FOLDER}cosine_similarity_final.png')
plt.close()

# plot the cosine similarity (training)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 102), cos_sims_training)
plt.xlabel('Epoch')
plt.ylabel('Cosine Similarity (Training)')
plt.title('Cosine Similarity (Training)')
plt.savefig(f'{FIG_FOLDER}cosine_similarity_training.png')
plt.close()

# plot the trajectory
plt.figure(figsize=(10, 10))
plt.plot([t[0] for t in trajectory], [t[1] for t in trajectory], marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
plt.xlabel('Disposition from Initial to Final')
plt.ylabel('Orthogonal Displacement')
plt.xlim(-0.1 * distance_init_final, distance_init_final * 1.1)
plt.ylim(-0.1 * distance_init_final, distance_init_final * 1.1)
plt.title('Trajectory')
plt.savefig(f'{FIG_FOLDER}trajectory.png')
plt.close()