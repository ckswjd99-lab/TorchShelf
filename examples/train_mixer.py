import numpy as np
import torch
import torchsummary
import warmup_scheduler

from tqdm import tqdm

from import_shelf import shelf
from shelf.models.mlp_mixer import MLPMixer
from shelf.dataloaders import get_CIFAR10_dataset
from shelf.trainers.classic import validate


# Hyperparameters
EPOCHS = 300
BATCH_SIZE = 128
CUTMIX_PROB = 0.5
CUTMIX_BETA = 1.0
LABEL_SMOOTHING = 0.1
LR_INIT = 1e-3
LR_LAST = 1e-6
WARMUP = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CIFAR-10 dataset
train_loader, test_loader = get_CIFAR10_dataset(augmentation=False)

# Define the model
model = MLPMixer(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=512, 
    hidden_c=64, 
    num_layers=8, 
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)

print(model)
torchsummary.summary(model, (3, 32, 32))


# Util funcs
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Train the model

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=5e-5, betas=(0.9, 0.99))
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_LAST)
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=WARMUP, after_scheduler=base_scheduler)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

for epoch in range(EPOCHS):
    train_loss_sum = 0
    train_correct_sum = 0
    train_step_sum = 0

    model.train()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1:3d}/{EPOCHS:3d}', leave=False)
    for input, label in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()

        r = np.random.rand(1)
        if CUTMIX_BETA > 0 and r < CUTMIX_PROB:
            lam = np.random.beta(CUTMIX_BETA, CUTMIX_BETA)
            rand_index = torch.randperm(input.size(0)).to(DEVICE)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_correct_sum += (output.argmax(dim=-1) == label).sum().item()
        train_step_sum += 1

        pbar.set_postfix(
            loss=f'{train_loss_sum / train_step_sum:.3f}',
            acc=f'{train_correct_sum / (train_step_sum * BATCH_SIZE):.3f}'
        )

    val_acc, val_loss = validate(test_loader, model, criterion, epoch)

    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    print(f"EPOCH {epoch:3d}/{EPOCHS:3d} , LR: {lr:.5e} | Train Loss: {train_loss_sum / train_step_sum:.4f}, Train Acc: {train_correct_sum / (train_step_sum * BATCH_SIZE) * 100:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}")
