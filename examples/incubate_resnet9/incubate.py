import torch
import torch.nn as nn
import torchsummary

from tqdm import tqdm

from models import ResNet9, ResNet9_Meta, ResNet9_Incubating

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate

EPOCH = 1
FT_EPOCH = 20
DEVICE = 'cuda'

META_MODEL_PATH = './saves/meta_vloss0.527.pth'


## CHECK MODELS ##

train_loader, val_loader = get_CIFAR10_dataset(batch_size=256)
model_meta = ResNet9_Meta().to(DEVICE)
model_meta.load_state_dict(torch.load(META_MODEL_PATH))
torchsummary.summary(model_meta, (3, 32, 32))

model_incubating = ResNet9_Incubating().to(DEVICE)
model_incubating.set_mode('full_inflate')
torchsummary.summary(model_incubating, (3, 32, 32))

model_incubating.set_mode('meta')
model_incubating.load_state_dict(torch.load(META_MODEL_PATH))

val_acc, val_loss = validate(val_loader, model_incubating, nn.CrossEntropyLoss(), 0)
print(f"Meta V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")
print()


## TRAIN ##
target_modules = ['block1', 'block2', 'block3', 'block4', 'block5']
# target_modules = ['block1', 'block2']

for target_module in target_modules:
    model_incubating.set_mode('meta')
    model_incubating.load_state_dict(torch.load(META_MODEL_PATH))

    size_target_meta = sum(p.numel() for p in getattr(model_incubating, target_module).parameters())

    model_incubating.set_mode(target_module)
    module_incubating = getattr(model_incubating, target_module)

    size_target_incubating = sum(p.numel() for p in module_incubating.parameters())

    print(f"Inflating {target_module} ({size_target_meta} -> {size_target_incubating})")
    val_acc, val_loss = validate(val_loader, model_incubating, nn.CrossEntropyLoss(), 0)
    print(f"Mode Init V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(module_incubating.parameters(), lr=1e-3)

    for epoch in range(EPOCH):

        train_acc = 0
        train_loss = 0

        pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
        for i, (input, label) in pbar:

            input, label = input.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            output = model_incubating(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            acc = (output.argmax(1) == label).float().mean()

            train_loss += loss.item()
            train_acc += acc.item()

            pbar.set_description(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T LOSS: {train_loss / (i + 1):.4f}, T ACC: {train_acc * 100 / (i + 1):.2f}%")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)

        print(f"Epoch {epoch + 1:3d}/{EPOCH:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc * 100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")

    val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)

    print(f"Final V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")
    print()


print("Testing Meta Again")
model_incubating.set_mode('meta')
val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)
print(f"Final V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")
print()


## VALIDATE ##

for target_module in target_modules:
    print(f"Testing {target_module}")
    model_incubating.set_mode(target_module)
    val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)

    print(f"Final V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")

model_incubating.set_mode('full_inflate')
val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)

print(f"Full Inflated V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")


## Fine-Tune
print("Fine-Tuning")
model_incubating.set_mode('full_inflate')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_incubating.parameters(), lr=1e-3)

for epoch in range(FT_EPOCH):
    train_acc, train_loss = train(train_loader, model_incubating, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model_incubating, criterion, epoch)

    print(f"Epoch {epoch + 1:3d}/{FT_EPOCH:3d} | T Loss: {train_loss:.4f}, T Acc: {train_acc * 100:.2f}, V Loss: {val_loss:.4f}, V Acc: {val_acc * 100:.2f}")
