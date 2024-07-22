import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'
SMOOTHING = 1e-3
EPOCH = 3

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

model1 = MLPMixer().to(DEVICE)
model2 = MLPMixer().to(DEVICE)

model_path_1 = './saves/set1_default/initial_weights.pth'
model_path_2 = './saves/set1_default/epoch_100.pth'

model1.load_state_dict(torch.load(model_path_1))
model2.load_state_dict(torch.load(model_path_2))

model1_state_dict = dict(model1.state_dict())
model2_state_dict = dict(model2.state_dict())

model1_weights = {pname: param.clone() for pname, param in model1_state_dict.items()}
model2_weights = {pname: param.clone() for pname, param in model2_state_dict.items()}
weights_difference = {pname: model2_weights[pname] - model1_weights[pname] for pname in model1_weights}
weights_difference_flat = torch.cat([param.view(-1) for param in weights_difference.values()]).abs()
wdiff_norm = torch.norm(weights_difference_flat, p=2)
num_params = sum(p.numel() for p in model1.parameters())

opt_dim_dict = {pname: diff / wdiff_norm for pname, diff in weights_difference.items()}


model = MLPMixer().to(DEVICE)
model.load_state_dict(model1_state_dict)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss_log = []
train_acc_log = []

with torch.no_grad():
    for epoch in range(EPOCH):
        train_loss_avg = 0
        train_acc_avg = 0
        num_steps = 0

        pbar = tqdm(train_loader, leave=False)

        for input, label in pbar:
            input, label = input.to(DEVICE), label.to(DEVICE)
            
            loss_orig = criterion(model(input), label)

            for pname, param in model.named_parameters():
                param.data += opt_dim_dict[pname].data

            loss_perb = criterion(model(input), label)

            for pname, param in model.named_parameters():
                param.data -= opt_dim_dict[pname].data

            optimizer.zero_grad()

            jvp_value = (loss_perb - loss_orig) / SMOOTHING

            for pname, param in model.named_parameters():
                param.grad = jvp_value * opt_dim_dict[pname]
            
            optimizer.step()

            train_loss = loss_orig.item()
            train_acc = (model(input).argmax(dim=1) == label).float().mean().item()
            
            if train_loss_avg == 0:
                train_loss_avg = train_loss
            else:
                train_loss_avg = 0.9 * train_loss_avg + 0.1 * train_loss
            
            if train_acc_avg == 0:
                train_acc_avg = train_acc
            else:
                train_acc_avg = 0.9 * train_acc_avg + 0.1 * train_acc
            
            train_loss_log.append(train_loss)
            train_acc_log.append(train_acc)

            pbar.set_description(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T LOSS: {train_loss_avg:.4f}, T ACC: {train_acc_avg * 100:.2f}%")

        val_acc, val_loss = validate(val_loader, model, criterion, epoch)
        print(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T LOSS: {train_loss_avg:.4f}, T ACC: {train_acc_avg * 100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc * 100:.2f}%")


# plot the training loss
plt.figure()
plt.plot(train_loss_log)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss of Ultimate Pruning')
plt.savefig(f'./logs/ultimate_pruning_tloss.png')
plt.close()

# plot the training accuracy
plt.figure()
plt.plot(train_acc_log)
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Training Accuracy of Ultimate Pruning')
plt.savefig(f'./logs/ultimate_pruning_tacc.png')
plt.close()