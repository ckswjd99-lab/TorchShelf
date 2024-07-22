import torch
from tqdm import tqdm

from models import ResNet9

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate


## HYPERPARAMS ##
SMOOTHING = 1e-3
EPOCHS = 100
LR_MAX = 1e-5
LR_MIN = 1e-7
MOMENTUM = 0.9

train_loader, val_loader = get_CIFAR10_dataset()

model = ResNet9().to('cuda')
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
momentum_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

# Load the model
# model.load_state_dict(torch.load('./saves/resnet9_vloss0.34_vacc0.89.pth'))
model.load_state_dict(torch.load('./saves/resnet9_vloss0.34_vacc0.90.pth'))


# Fine-tune the model
model.train()
with torch.no_grad():
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, leave=False)

        train_loss = None
        train_acc = None
        lr_sum = 0

        for input, label in pbar:
            input, label = input.to('cuda'), label.to('cuda')
            
            pnames = [name for name, param in model.named_parameters()]
            params = [param for name, param in model.named_parameters()]
            pnoise = [torch.randn_like(param) for param in params]

            loss_orig = criterion(model(input), label)

            for param, noise in zip(params, pnoise):
                param.data += SMOOTHING * noise
            
            loss_pos = criterion(model(input), label)

            for param, noise in zip(params, pnoise):
                param.data -= 2 * SMOOTHING * noise

            loss_neg = criterion(model(input), label)

            for param, noise in zip(params, pnoise):
                param.data += SMOOTHING * noise

            jvp_value = (loss_pos - loss_neg) / (2 * SMOOTHING)
            # vthv_value = (loss_pos - 2 * loss_orig + loss_neg) / (SMOOTHING ** 2)

            # if vthv_value < 0:
            #     scaler_abs = LR_MIN
            #     scaler_sign = torch.sign(jvp_value)
            #     scaler = scaler_sign * scaler_abs
            # else:
            #     scaler = jvp_value / vthv_value
            #     scaler_abs = abs(scaler)
            #     scaler_abs = max(min(scaler_abs, LR_MAX), LR_MIN)
            #     scaler_sign = torch.sign(scaler)
            #     scaler = scaler_sign * scaler_abs

            # lr_sum += scaler_abs


            # estimated_gradient = [scaler * noise for noise in pnoise]
            estimated_gradient = [LR_MAX * jvp_value * noise for noise in pnoise]

            for momentum, grad in zip(momentum_dict.values(), estimated_gradient):
                momentum.data = MOMENTUM * momentum + (1 - MOMENTUM) * grad
            
            for param, momentum in zip(params, momentum_dict.values()):
                param.data -= momentum
            
            loss_new = criterion(model(input), label)


            opt_loss = criterion(model(input), label).item()
            opt_acc = (model(input).argmax(dim=1) == label).float().mean().item()

            if train_loss == None:
                train_loss = opt_loss
            else:
                train_loss = 0.99 * train_loss + 0.01 * opt_loss
            
            if train_acc == None:
                train_acc = opt_acc
            else:
                train_acc = 0.99 * train_acc + 0.01 * opt_acc

            pbar.set_description(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%")

        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        lr_avg = lr_sum / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{EPOCHS:3d} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | LR: {lr_avg:.4e}")