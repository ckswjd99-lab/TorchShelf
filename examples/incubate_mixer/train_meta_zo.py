import torch
import torch.nn as nn

import torchsummary
import warmup_scheduler

from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.models.mlp_mixer import MLPMixer
from shelf.pruners.scoring import get_zo_grasp_score

DEVICE = 'cuda'
EPOCH = 500
SPARSITY = 0.05
SMOOTHING = 1e-3

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)

model_meta = MLPMixer(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=32,
    hidden_c=64, 
    num_layers=8, 
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)
torchsummary.summary(model_meta, (3, 32, 32))

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model_meta.parameters(), lr=1e-3, weight_decay=5e-5, betas=(0.9, 0.99))
# base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)
# scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_scheduler)

pnames = [pname for pname, _ in model_meta.named_parameters()]

for epoch in range(EPOCH):
    grasp_score_dict = get_zo_grasp_score(train_loader, model_meta, query=32)

    importance_score = [-gs for gs in grasp_score_dict.values()]
    impor_score_dict = {pname: impscore for pname, impscore in zip(pnames, importance_score)}
    impor_score_flat = torch.cat([impscore.flatten() for impscore in importance_score])
    impor_score_flat = torch.sort(impor_score_flat, descending=True).values
    threshold = impor_score_flat[int(impor_score_flat.size(0) * SPARSITY)]

    # create mask for each layer
    cge_masks = {pname: (impscore > threshold).float() for pname, impscore in zip(pnames, importance_score)}
    alive_count = {pname: mask.sum().item() for pname, mask in cge_masks.items()}

    for mask in cge_masks.values():
        # random shuffle
        mask_flat = mask.view(-1)
        mask_flat = mask_flat[torch.randperm(mask_flat.size(0))]

    train_acc_sum = 0
    train_loss_sum = 0
    num_steps = 0

    pbar = tqdm(train_loader, leave=False)
    for input, label in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)
        num_steps += 1
        
        real_gradient = gradient_fo(input, label, model_meta, criterion)
        estimated_gradient = {pname: real_gradient[pname] * mask for pname, mask in cge_masks.items()}
        
        for pname, param in model_meta.named_parameters():
            param.grad = estimated_gradient[pname]
        
        optimizer.step()

        loss = criterion(model_meta(input), label)

        train_loss_sum += loss.item()
        train_acc_sum += (model_meta(input).argmax(dim=-1) == label).float().mean().item()

        pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCH:3d} | Train Loss: {train_loss_sum / num_steps:.4f}, Train Acc: {train_acc_sum / num_steps * 100:.2f}%")
    
    train_loss = train_loss_sum / num_steps
    train_acc = train_acc_sum / num_steps

    val_acc, val_loss = validate(val_loader, model_meta, criterion, epoch)
    print(f"Epoch {epoch+1:3d}/{EPOCH:3d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

    # scheduler.step()

torch.save(model_meta.state_dict(), './saves/model_meta_zo.pth')