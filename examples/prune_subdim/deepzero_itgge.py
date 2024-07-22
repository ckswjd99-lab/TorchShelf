import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch.nn.utils.prune import remove, l1_unstructured
from tqdm import tqdm
import torchsummary
import warmup_scheduler

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.pruners.scoring import get_grasp_score_dict, get_zo_grasp_score
from shelf.models.resnet.etc import resnet20


EPOCHS = 50
DEVICE = 'cuda'
PRUNING_RATE = 0.9
SUBDIM_RATE = 0.0
MASK_UPDATE_EPOCH = 20

train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=128, augmentation=False)

model = resnet20().to(DEVICE)

# prune model
grasp_score = get_zo_grasp_score(train_loader, model, query=196)

importance_scores = {pname: -score for pname, score in grasp_score.items()}
impt_score_flat = torch.cat([score.flatten() for score in importance_scores.values()])
threshold = torch.kthvalue(impt_score_flat, int(len(impt_score_flat) * PRUNING_RATE)).values.item()
layerwise_pruning_rate = {pname: ((param < threshold).float().sum() / param.numel()).item() for pname, param in importance_scores.items()}

torchsummary.summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4, nesterov=True)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=base_scheduler)

best_acc = 0

# Train the model
for epoch in range(EPOCHS):
    train_acc_sum = 0
    train_loss_sum = 0
    num_steps = 0

    dprune_masks = {pname: (torch.rand_like(param) > layerwise_pruning_rate[pname]).float() for pname, param in model.named_parameters()}

    num_alive_grads = None
    
    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for i, (input, label) in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)
        num_steps += 1

        optimizer.zero_grad()

        real_gradient = gradient_fo(input, label, model, criterion)

        for pname, param in model.named_parameters():
            param.grad = real_gradient[pname] * dprune_masks[pname]

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

        pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss_sum/num_steps:.4f}, T ACC: {train_acc_sum/num_steps*100:.2f}% | ALIVE: {num_alive_grads:6,d}")

    
    train_acc = train_acc_sum / num_steps
    train_loss = train_loss_sum / num_steps

    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | ALIVE: {num_alive_grads:6,d}")

    if best_acc < val_acc:
        best_acc = val_acc

        torch.save(model.state_dict(), f'./saves/resnet20_pr{PRUNING_RATE:.2f}_bestvacc.pth')
    
    scheduler.step()


print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

