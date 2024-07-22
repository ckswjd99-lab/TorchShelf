import torch
import os
import time
import torch.nn.utils.prune as prune
import warmup_scheduler

from tqdm import tqdm

from import_shelf import shelf
from shelf.models.transformer import VisionTransformer
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fo
from shelf.pruners.scoring import get_grasp_score
from shelf.pruners import global_unstructured_L1, undo_pruning

## HYPERPARAMS ##
EPOCHS = 500
BATCH_SIZE = 128

DIM_MODEL = 256
DEPTH = 4
HEADS = 6
MLP_DIM = 256
DROPOUT = 0.1
EMB_DROPOUT = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

## DATA LOADING ##
train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## MODEL ##
model = VisionTransformer(
    image_size=32, 
    patch_size=4, 
    num_classes=10, 
    dim=DIM_MODEL,
    depth=DEPTH,
    heads=HEADS,
    mlp_dim=MLP_DIM,
    dropout=DROPOUT,
    emb_dropout=EMB_DROPOUT
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

print(model)
print(f"Model has {num_params:,} parameters")

# save initial model weights
if not os.path.exists('saves/ablation_tinyvit'):
    os.makedirs('saves/ablation_tinyvit')
if not os.path.exists('saves/ablation_tinyvit/initial_model.pth'):
    torch.save(model.state_dict(), 'saves/ablation_tinyvit/initial_model.pth')
else:
    model.load_state_dict(torch.load('saves/ablation_tinyvit/initial_model.pth'))

## OTHERS ##
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(0.001) ** (1 / EPOCHS))
scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=10, after_scheduler=base_scheduler)


## TRAINING ##
print("1. Baseline: FO")

if not os.path.exists('saves/ablation_tinyvit/fo_baseline.pth') or True:
    print("| EPOCH |     LR     | T LOSS |  T ACC  | V LOSS |  V ACC  |  ETA  |")
    print("|-------|------------|--------|---------|--------|---------|-------|")

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        lr = optimizer.param_groups[0]['lr']

        eta = int(time.time() - start_time)

        print(f"|  {epoch+1:3d}  | {lr:.4e} | {train_loss:.4f} |  {train_acc*100:2.2f}% | {val_loss:.4f} |  {val_acc*100:2.2f}% | {eta:5d} |")

        scheduler.step()

    # torch.save(model.state_dict(), 'saves/ablation_tinyvit/fo_baseline.pth')
    print("")
else:
    model.load_state_dict(torch.load('saves/ablation_tinyvit/fo_baseline.pth'))
    val_acc, val_loss = validate(val_loader, model, criterion, 0)

    print(f"Already Trained")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print("")

exit(0)

print("2. Baseline: Sparse Weight (1%)")

# initialize
model.load_state_dict(torch.load('saves/ablation_tinyvit/initial_model.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(0.01) ** (1 / EPOCHS))

x_temp, t_temp = next(iter(train_loader))
x_temp, t_temp = x_temp.to(DEVICE), t_temp.to(DEVICE)
param_names = [ name for name, _ in model.named_parameters() ]

grasp_score = get_grasp_score(x_temp, t_temp, model)
grasp_score_dict = {k: v for k, v in zip(param_names, grasp_score)}

for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        prune.l1_unstructured(module, name='weight', amount=0.99, importance_scores=grasp_score_dict[name + '.weight'])

if not os.path.exists('saves/ablation_tinyvit/sparse_weight_baseline.pth'):
    print("| EPOCH |     LR     | T LOSS |  T ACC  | V LOSS |  V ACC  |  ETA  |")
    print("|-------|------------|--------|---------|--------|---------|-------|")

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        lr = optimizer.param_groups[0]['lr']

        eta = int(time.time() - start_time)

        print(f"|  {epoch+1:3d}  | {lr:.4e} | {train_loss:.4f} |  {train_acc*100:.2f}% | {val_loss:.4f} |  {val_acc*100:.2f}% | {eta:5d} |")

        scheduler.step()

    for module in model.modules():
        if hasattr(module, 'weight'):
            prune.remove(module, 'weight')

    torch.save(model.state_dict(), 'saves/ablation_tinyvit/sparse_weight_baseline.pth')
    print("")
else:
    for module in model.modules():
        if hasattr(module, 'weight'):
            prune.remove(module, 'weight')
            
    model.load_state_dict(torch.load('saves/ablation_tinyvit/sparse_weight_baseline.pth'))
    val_acc, val_loss = validate(val_loader, model, criterion, 0)

    print(f"Already Trained")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print("")


print("3. Baseline: Sparse Gradient (1%)")

# initialize
model.load_state_dict(torch.load('saves/ablation_tinyvit/initial_model.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(0.01) ** (1 / EPOCHS))

if not os.path.exists('saves/ablation_tinyvit/sparse_gradient_baseline.pth'):
    print("| EPOCH |     LR     | T LOSS |  T ACC  | V LOSS |  V ACC  |  ETA  |")
    print("|-------|------------|--------|---------|--------|---------|-------|")

    for epoch in range(EPOCHS):
        start_time = time.time()

        pbar = tqdm(train_loader, leave=False)

        total_loss = 0
        total_correct = 0
        total_step = 0

        for input, target in pbar:
            input, target = input.to(DEVICE), target.to(DEVICE)

            real_gradient = gradient_fo(input, target, model, criterion)
            grasp_score = get_grasp_score(input, target, model)

            for (name, param), gs in zip(model.named_parameters(), grasp_score):
                mask = (gs < torch.quantile(gs, 0.01)).float()
                real_gradient[name] *= mask
            
            optimizer.zero_grad()
            
            for name, param in model.named_parameters():
                param.grad = real_gradient[name]
            
            optimizer.step()

            output = model(input)
            loss = criterion(output, target).item()
            accuracy = (output.argmax(1) == target).float().mean().item()

            total_loss += loss
            total_correct += accuracy
            total_step += 1

            pbar.set_postfix(train_accuracy=total_correct/total_step, train_loss=total_loss/total_step)

        val_acc, val_loss = validate(train_loader, model, criterion, epoch)

        lr = optimizer.param_groups[0]['lr']

        train_loss = total_loss / total_step
        train_acc = total_correct / total_step

        eta = int(time.time() - start_time)

        print(f"|  {epoch+1:3d}  | {lr:.4e} | {train_loss:.4f} |  {train_acc*100:.2f}% | {val_loss:.4f} |  {val_acc*100:.2f}% | {eta:5d} |")

        scheduler.step()

    torch.save(model.state_dict(), 'saves/ablation_tinyvit/sparse_gradient_baseline.pth')
    print("")
else:
    model.load_state_dict(torch.load('saves/ablation_tinyvit/sparse_gradient_baseline.pth'))
    val_acc, val_loss = validate(val_loader, model, criterion, 0)

    print(f"Already Trained")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print("")
