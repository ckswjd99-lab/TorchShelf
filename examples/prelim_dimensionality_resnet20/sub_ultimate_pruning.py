import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import warmup_scheduler

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'
SMOOTHING = 1e-3
EPOCH = 5
NUM_GROUPS = 20

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

reference_models = [MLPMixer().to(DEVICE) for _ in range(NUM_GROUPS+1)]
model_paths = ['./saves/set1/initial_weights.pth'] + [f'./saves/set1/epoch_{i * int(200/NUM_GROUPS)}.pth' for i in range(1, NUM_GROUPS + 1)] + ['./saves/set1/final_weights_vloss.0.788.pth']
for model, path in zip(reference_models, model_paths):
    model.load_state_dict(torch.load(path))
model_state_dicts = [dict(model.state_dict()) for model in reference_models]
model_weights = [{pname: param.clone() for pname, param in model_state_dict.items()} for model_state_dict in model_state_dicts]

weight_differences = [{pname: model_weights[i][pname] - model_weights[i - 1][pname] for pname in model_weights[i - 1]} for i in range(1, NUM_GROUPS)]
wdiff_norms = [torch.norm(torch.cat([param.view(-1) for param in weight_diff.values()]).abs(), p=2) for weight_diff in weight_differences]

opt_dim_dicts = [{pname: diff / wdiff_norm for pname, diff in weight_diff.items()} for weight_diff, wdiff_norm in zip(weight_differences, wdiff_norms)]

# using models
print("Using models:")
for model in model_paths:
    print(model)
print()

# check final accuracy and loss
val_acc, val_loss = validate(val_loader, reference_models[-1], nn.CrossEntropyLoss(), 0)
print(f"Final Accuracy: {val_acc * 100:.2f}%, Final Loss: {val_loss:.4f}")

model = MLPMixer().to(DEVICE)
model.load_state_dict(model_state_dicts[0])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e+1, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-4)
# scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_scheduler)

train_loss_log = []
train_acc_log = []

with torch.no_grad():
    for epoch in range(EPOCH):
        train_loss_sum = 0
        train_acc_sum = 0
        num_steps = 0

        pbar = tqdm(train_loader, leave=False)

        for input, label in pbar:
            input, label = input.to(DEVICE), label.to(DEVICE)
            num_steps += 1
            
            loss_orig = criterion(model(input), label)
            optimizer.zero_grad()

            estimated_gradient = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}

            for opt_dim_dict in opt_dim_dicts:
                for pname, param in model.named_parameters():
                    param.data += SMOOTHING * opt_dim_dict[pname].data
                loss_perb = criterion(model(input), label)
                for pname, param in model.named_parameters():
                    param.data -= SMOOTHING * opt_dim_dict[pname].data
                    
                    estimated_gradient[pname] += (loss_perb - loss_orig) / SMOOTHING * opt_dim_dict[pname]
            
            for pname, param in model.named_parameters():
                param.grad = estimated_gradient[pname]
                
            optimizer.step()

            train_loss = loss_orig.item()
            train_acc = (model(input).argmax(dim=1) == label).float().mean().item()
            
            train_loss_sum += train_loss
            train_acc_sum += train_acc
            
            train_loss_log.append(train_loss)
            train_acc_log.append(train_acc)
            
            pbar.set_description(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T LOSS: {train_loss_sum / num_steps:.4f}, T ACC: {train_acc_sum / num_steps * 100:.2f}%")

        val_acc, val_loss = validate(val_loader, model, criterion, epoch)
        print(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T LOSS: {train_loss_sum / num_steps:.4f}, T ACC: {train_acc_sum / num_steps * 100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc * 100:.2f}%")


        # scheduler.step()


# plot the training loss
plt.figure()
plt.plot(train_loss_log)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title(f'Training Loss of Sub-Ultimate Pruning (#Groups={NUM_GROUPS})')
plt.savefig(f'./logs/sub_ultimate_pruning_tloss.png')
plt.close()

# plot the training accuracy
plt.figure()
plt.plot(train_acc_log)
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title(f'Training Accuracy of Sub-Ultimate Pruning (#Groups={NUM_GROUPS})')
plt.savefig(f'./logs/sub_ultimate_pruning_tacc.png')
plt.close()