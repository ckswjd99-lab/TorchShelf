import torch
import torch.nn as nn

import torchsummary
import warmup_scheduler

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.models.mlp_mixer import MLPMixer

DEVICE = 'cuda'
EPOCH_INC = 20
EPOCH_FT = 100

META_PATH = './saves/model_meta.pth'

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
model_meta.load_state_dict(torch.load(META_PATH))
for param in model_meta.parameters():
    param.requires_grad = False

model = MLPMixer(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=64,
    hidden_c=512, 
    num_layers=8, 
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)
torchsummary.summary(model, (3, 32, 32))
for param in model.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# validate meta model
val_acc, val_loss = validate(val_loader, model_meta, criterion, 0)
print(f"Meta Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

for i, mixer_layer in enumerate(model.mixer_layers):
    print(f"Incubating Layer {i+1:3d}")

    model_meta.load_state_dict(torch.load(META_PATH))

    orig_layer = model_meta.mixer_layers[i]
    model_meta.mixer_layers[i] = mixer_layer

    for param in model_meta.mixer_layers[i].parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model_meta.mixer_layers[i].parameters(), lr=1e-3, weight_decay=5e-5, betas=(0.9, 0.99))

    for epoch in range(EPOCH_INC):
        train_acc, train_loss = train(train_loader, model_meta, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model_meta, criterion, epoch)
        print(f"Epoch {epoch+1:3d}/{EPOCH_INC:3d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")
    
    model_meta.mixer_layers[i] = orig_layer

    for param in model_meta.mixer_layers[i].parameters():
        param.requires_grad = False

model.patch_emb = model_meta.patch_emb
model.cls_token = model_meta.cls_token
model.ln = model_meta.ln
model.clf = model_meta.clf

for param in model.parameters():
    param.requires_grad = True

print("Re-evaluating the incubated modules")
for i, mixer_layer in enumerate(model.mixer_layers):
    orig_layer = model_meta.mixer_layers[i]
    model_meta.mixer_layers[i] = mixer_layer

    val_acc, val_loss = validate(val_loader, model_meta, criterion, 0)

    model_meta.mixer_layers[i] = orig_layer

    print(f"Layer {i+1:3d} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

val_acc, val_loss = validate(val_loader, model, criterion, 0)
print(f"Incubated Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

torch.save(model.state_dict(), './saves/model_incubated.pth')

print("Fine-tuning the incubated model")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5, betas=(0.9, 0.99))
for epoch in range(EPOCH_FT):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f"Epoch {epoch+1:3d}/{EPOCH_FT:3d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

torch.save(model.state_dict(), './saves/model_finetuned.pth')