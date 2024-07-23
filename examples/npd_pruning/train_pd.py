import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


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
        prune.l1_unstructured(module, name='weight', amount=PRUNE_RATE)

pruned_params = [param for pname, param in model.named_parameters() if 'weight_orig' in pname]
pruned_masks = [mask for pname, mask in model.named_buffers() if 'weight_mask' in pname]

num_pruned_params = sum((pmask == 0).sum().item() for pmask in pruned_masks)

print(f"Number of alive parameters: {num_params - num_pruned_params}/{num_params} ({(num_params - num_pruned_params) / num_params * 100:.2f}%)")

# Train the model
best_val_acc = 0

for epoch in range(EPOCHS):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc

    print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% |" + (" *" if is_best else ""))


print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
