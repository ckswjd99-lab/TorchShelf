from models import ConvNet_710, ConvNet_3250, ConvNet_6694, ConvNet_9640, ConvNet_12282

import torch
import torch.nn as nn
import time

import argparse

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import train_zo


## ARGPARSE ##

parser = argparse.ArgumentParser("Trainer for ablation study of various training methods")
parser.add_argument("--model", type=str, default="ConvNet_710", choices=["ConvNet_710", "ConvNet_3250", "ConvNet_6694", "ConvNet_9640", "ConvNet_12282"], help="Model to train")
parser.add_argument("--method", type=str, default="classic", choices=["classic", "cge", "rge"], help="Training method to use")
parser.add_argument("--epoch", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

args = parser.parse_args()

print(f"Model: {args.model}")
print(f"Method: {args.method}")
print(f"Epoch: {args.epoch}")
print(f"Learning rate: {args.lr}")


## MAIN ##

train_loader, val_loader = get_CIFAR10_dataset()

# Select model
if args.model == "ConvNet_710":
    model = ConvNet_710().to('cuda')
elif args.model == "ConvNet_3250":
    model = ConvNet_3250().to('cuda')
elif args.model == "ConvNet_6694":
    model = ConvNet_6694().to('cuda')
elif args.model == "ConvNet_9640":
    model = ConvNet_9640().to('cuda')
elif args.model == "ConvNet_12282":
    model = ConvNet_12282().to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of parameters: {num_params}")

# Train the model
for epoch in range(args.epoch):
    epoch_start = time.time()

    if args.method == "classic":
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    elif args.method == "cge":
        train_acc, train_loss = train_zo(train_loader, model, criterion, optimizer, epoch, ge_type='cge')
    elif args.method == "rge":
        train_acc, train_loss = train_zo(train_loader, model, criterion, optimizer, epoch, ge_type='rge', query=num_params)
    
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    epoch_end = time.time()

    print(f"Epoch {epoch+1:3d}/{args.epoch:3d} | Train Acc: {train_acc*100:.2f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}, Val Loss: {val_loss:.4f} | ETA: {epoch_end-epoch_start:5.2f}s")

torch.save(model.state_dict(), f"./saves/{args.model}_{args.method}_e{args.epoch}_vacc{val_acc:.4f}.pt")
print(f"Model saved as {args.model}_{args.method}_e{args.epoch}_vacc{val_acc:.4f}.pt")