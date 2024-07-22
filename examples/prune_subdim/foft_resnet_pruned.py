import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch.nn.utils.prune import remove, l1_unstructured
from tqdm import tqdm
import torchsummary

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.pruners.scoring import get_grasp_score_dict, get_zo_grasp_score
from shelf.models.mlp_mixer import MLPMixer
from shelf.models.resnet.resnet9_cifar import ResNet9


EPOCHS = 100
DEVICE = 'cuda'
PRUNING_RATE = 0.9
MASK_UPDATE_EPOCH = 20
SMOOTHING = 1e-3
QUERY=1

# PRETRAINED_PATH = './saves/resnet9_pr0.90_bestvacc82.73.pth'
PRETRAINED_PATH = './saves/resnet9_pr0.90_sdim0.90_bestvacc86.88.pth'

train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=256)

model = ResNet9().to(DEVICE)

# prune model
# grasp_score = get_grasp_score_dict(train_loader, model)
grasp_score = get_zo_grasp_score(train_loader, model)
# grasp_score = {}
for mname, module in model.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.0)

model.load_state_dict(torch.load(PRETRAINED_PATH))
torchsummary.summary(model, (3, 32, 32))

val_acc, val_loss = validate(val_loader, model, nn.CrossEntropyLoss().to(DEVICE), 0)
print(f"Pretrained Validation Accuracy: {val_acc*100:.2f}%, Loss: {val_loss:.4f}")

# remove pruning and initialize zero areas
for mname, module in model.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        mask = dict(model.named_buffers())[mname + '.weight_mask']
        adding_area = (mask.data == 0).float()
        # initialize zero areas
        module.weight.data += torch.randn_like(module.weight.data) * 1e-5 * adding_area
        mask.data.fill_(1)

        prune.remove(module, 'weight')

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01 ** (1/EPOCHS))

best_acc = val_acc

# Train the model
model.train()
for epoch in range(EPOCHS):
    train_acc_sum = 0
    train_loss_sum = 0
    num_steps = 0

    dprune_masks = {}
    num_alive_grads = None

    tloss_checkpoint = 1e99
    train_acc_avg = 1e98

    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for i, (input, label) in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)
        num_steps += 1

        optimizer.zero_grad()

        loss = criterion(model(input), label)
        loss.backward()

        # count alive gradients
        if num_alive_grads is None:
            num_alive_grads = 0
            num_total_grads = 0
            for pname, param in model.named_parameters():
                if param.grad is not None:
                    num_alive_grads += torch.sum(param.grad != 0).item()
                    num_total_grads += param.grad.nelement()

        optimizer.step()

        output = model(input)
        loss = criterion(output, label)

        train_acc_sum += (output.argmax(1) == label).float().mean().item()
        train_loss_sum += loss.item()

        train_acc_avg = train_acc_sum / num_steps

        lr = optimizer.param_groups[0]['lr']

        pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d}, LR {lr:.4e} | T LOSS: {train_loss_sum/num_steps:.4f}, T ACC: {train_acc_sum/num_steps*100:.2f}% | ALIVE: {num_alive_grads:6,d}")

    
    train_acc = train_acc_sum / num_steps
    train_loss = train_loss_sum / num_steps

    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d}, LR {lr:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | ALIVE: {num_alive_grads:6,d}")

    if val_acc > best_acc:
        best_acc = val_acc

    scheduler.step()

print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

