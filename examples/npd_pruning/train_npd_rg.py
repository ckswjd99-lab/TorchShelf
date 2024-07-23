import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from tqdm import tqdm
import os

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.resnet.etc import resnet20


EPOCHS = 100
DEVICE = 'cuda'
PRUNE_RATE = 0.9

train_loader, val_loader = get_CIFAR10_dataset(root='../data', augmentation=False)

model = resnet20().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)

num_params = sum(p.numel() for p in model.parameters())

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.0)

if os.path.exists("./saves/nprune_package.pt"):
    nprune_package = torch.load("./saves/nprune_package.pt")
    nprune_num_groups = nprune_package['nprune_num_groups']
    nprune_group_dict = nprune_package['nprune_group_dict']
    nprune_dimension_dict = nprune_package['nprune_dimension_dict']
else:
    nprune_num_groups = {
        pname: int((1-PRUNE_RATE) * param.numel())
        for pname, param in model.named_parameters() 
        if 'weight_orig' in pname
    }

    nprune_group_dict = {
        pname: torch.randint_like(param, 0, nprune_num_groups[pname])
        for pname, param in model.named_parameters()
        if 'weight_orig' in pname
    }

    nprune_dimension_dict = {
        pname: torch.randn_like(param)
        for pname, param in model.named_parameters()
        if 'weight_orig' in pname
    }

    print("Creating dimensions")
    pbar = tqdm(nprune_dimension_dict.items(), leave=False)
    for pname, dimension in pbar:
        for group_idx in range(nprune_num_groups[pname]):
            pbar.set_description(f"{pname} ({group_idx}/{nprune_num_groups[pname]})")
            dimension[nprune_group_dict[pname] == group_idx] /= dimension[nprune_group_dict[pname] == group_idx].view(-1).norm()

    nprune_package = {
        "nprune_num_groups": nprune_num_groups,
        "nprune_group_dict": nprune_group_dict,
        "nprune_dimension_dict": nprune_dimension_dict
    }

    torch.save(nprune_package, "./saves/nprune_package.pt")

num_alives = 0

for pname, param in model.named_parameters():
    if 'weight_orig' in pname:
        num_alives += nprune_num_groups[pname]
    else:
        num_alives += param.numel()

print(f"Number of alive parameters: {num_alives}/{num_params} ({num_alives / num_params * 100:.2f}%)")


# Train the model
best_val_acc = 0

for epoch in range(EPOCHS):
    
    train_acc_sum = 0
    train_loss_sum = 0
    
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (inputs, targets) in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for pname, param in model.named_parameters():
            if 'weight_orig' in pname:
                pgrad = param.grad.clone()
                pgrad_flat = pgrad.view(-1)

                group_dims = [(nprune_dimension_dict[pname] * (nprune_group_dict[pname] == group_idx).float()).view(-1) for group_idx in range(nprune_num_groups[pname])]
                gdim_mats = torch.stack(group_dims, dim=0)

                jvp_values = gdim_mats @ pgrad_flat

                pgrad_dimmed = jvp_values @ gdim_mats

                pgrad = pgrad_dimmed.view(param.shape)
                param.grad = pgrad

        optimizer.step()
        
        train_acc_sum += (outputs.argmax(1) == targets).float().mean().item()
        train_loss_sum += loss.item()

        train_acc = train_acc_sum / (i+1)
        train_loss = train_loss_sum / (i+1)

        pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%")


    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% |" + (" *" if is_best else ""))


print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
