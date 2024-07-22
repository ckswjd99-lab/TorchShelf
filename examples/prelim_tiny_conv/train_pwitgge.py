from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.zeroth_order import gradient_fo, functional_xent, group_by_gradient_exp, gradient_estimate_pwitgge
from shelf.trainers.classic import validate

from models import ConvNet770, ConvNet3250, ConvNet6694
from tqdm import tqdm

import torch
import torch.nn as nn
import warmup_scheduler

# Hyperparams
EPOCHS = 50
DEVICE = 'cuda'


# Load CIFAR-10 dataset
train_loader, val_loader = get_CIFAR10_dataset(batch_size=128, augmentation=False)

# Load models
model770 = ConvNet770().to(DEVICE)
model4010 = ConvNet3250().to(DEVICE)
model6236 = ConvNet6694().to(DEVICE)

# Train model770
model = model770
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

momentum_dict = {pname: torch.randn_like(param) for pname, param in model.named_parameters()}
num_groups = int(num_params * 0.2)
print(f"Number of groups: {num_groups}")

for epoch in range(EPOCHS):
    train_acc_sum = 0
    train_loss_sum = 0
    cossim_sum = 0

    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for i, (input, label) in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)

        # Zero the gradients
        model770.zero_grad()

        real_gradient = gradient_fo(input, label, model, criterion)

        config = {}
        estimated_gradient = gradient_estimate_pwitgge(input, label, model, criterion, real_gradient=real_gradient, init_momentum=momentum_dict, config=config)

        real_grad_norm = sum(real_gradient[pname].norm()**2 for pname in real_gradient)**0.5
        est_grad_norm = sum(estimated_gradient[pname].norm()**2 for pname in estimated_gradient)**0.5
        
        for pname, param in model.named_parameters():
            param.grad = estimated_gradient[pname] * 1e7
            momentum_dict[pname] = momentum_dict[pname] * 0.9 + estimated_gradient[pname] * 0.1
        
        
        optimizer.step()

        loss = criterion(model(input), label)
        acc = (model(input).argmax(dim=1) == label).float().mean()

        train_loss_sum += loss.item()
        train_acc_sum += acc.item()
        cossim_sum += config['cossim']

        pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T Loss: {train_loss_sum/(i+1):.4f}, T Acc: {train_acc_sum/(i+1)*100:.2f}, cossim {cossim_sum/(i+1):.4f}, est norm {est_grad_norm:.4e} real norm {real_grad_norm:.4e}")
        # pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T Loss: {train_loss_sum/(i+1):.4f}, T Acc: {train_acc_sum/(i+1)*100:.2f}")

        # scheduler.step()
    
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T Loss: {train_loss_sum/(i+1):.4f}, T Acc: {train_acc_sum/(i+1)*100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc*100:.2f}% | Cossim: {cossim_sum/(i+1):.4f}")
