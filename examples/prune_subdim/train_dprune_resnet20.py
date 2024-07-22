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
from shelf.models.resnet.etc import resnet20


EPOCHS = 200
DEVICE = 'cuda'
PRUNING_RATE = 0.9
SUBDIM_RATE = 0.0
MASK_UPDATE_EPOCH = 20

train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=128)

model = resnet20().to(DEVICE)

# prune model
# grasp_score = get_grasp_score_dict(train_loader, model)
grasp_score = get_zo_grasp_score(train_loader, model, query=196)
# grasp_score = {}

importance_scores = {pname: -score for pname, score in grasp_score.items()}
impt_score_flat = torch.cat([score.flatten() for score in importance_scores.values()])
threshold = torch.kthvalue(impt_score_flat, int(len(impt_score_flat) * PRUNING_RATE)).values.item()
layerwise_pruning_rate = {pname: ((param < threshold).float().sum() / param.numel()).item() for pname, param in importance_scores.items()}

# for mname, module in model.named_modules():
#     if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
#         if mname + '.weight' in grasp_score:
#             prate = layerwise_pruning_rate[mname + '.weight']
#             prune.l1_unstructured(module, name='weight', amount=prate, importance_scores=-grasp_score[mname + '.weight'])
#         else:
#             prate = PRUNING_RATE
#             prune.l1_unstructured(module, name='weight', amount=prate)
        
#         print(f"{mname} | Pruning Rate: {prate:.2f}")

torchsummary.summary(model, (3, 32, 32))

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 210])

for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-1*0.1


best_acc = 0

# Train the model
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
        # update dprune masks
        if i % MASK_UPDATE_EPOCH == 0:
            if tloss_checkpoint > train_acc_avg:
                tloss_checkpoint = train_acc_avg
                for mname, module in model.named_modules():
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                        # prune_mask_weight = dict(model.named_buffers())[mname + '.weight_mask']
                        dprune_masks[mname + '.weight'] = (torch.rand_like(module.weight) > layerwise_pruning_rate[mname])
            num_alive_grads = None

        input, label = input.to(DEVICE), label.to(DEVICE)
        num_steps += 1

        optimizer.zero_grad()

        real_gradient = gradient_fo(input, label, model, criterion)

        for pname, param in model.named_parameters():
            if pname.replace('_orig', '') in dprune_masks:
                weight_name = pname.replace('_orig', '')
                param.grad = real_gradient[pname] * dprune_masks[weight_name]
            else:
                param.grad = real_gradient[pname]

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


print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

