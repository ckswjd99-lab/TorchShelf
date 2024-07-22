"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.func as fc
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

from import_shelf import shelf
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.zeroth_order import gradient_fwd
from shelf.trainers.classic import validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def functional_xent(params, buffers, names, model, x, t):
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return F.cross_entropy(y, t)

def get_CIFAR10_dataset(root='./data', batch_size=128):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader


def run(rank, size):
    model = ToyModel().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_loader, val_loader = get_CIFAR10_dataset()

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        train_loader = tqdm(train_loader, leave=False)

    for epoch in range(10):
        # Synchronize the model parameters across processes
        for param in model.parameters():
            dist.broadcast(param.data, 0)
        
        # Train the model
        model.train()
        for input, label in train_loader:
            input, label = input.to(DEVICE), label.to(DEVICE)
            estimated_gradient = gradient_fwd(input, label, model, functional_xent, query=128, type='rge')

            # allreduce the estimated gradient
            for k, v in estimated_gradient.items():
                # dist.all_reduce(v, op=dist.ReduceOp.SUM)
                v /= size
            
            # update the model parameters
            for pname, param in model.named_parameters():
                param.grad = estimated_gradient[pname]
            
            optimizer.step()
            optimizer.zero_grad()

            # loss and accuracy
            loss = F.cross_entropy(model(input), label)
            acc = (model(input).argmax(dim=1) == label).float().mean().item()
            if rank == 0:
                train_loader.set_description(f"Epoch {epoch} | Loss: {loss.item():6.4f}, Accuracy: {acc*100:.2f}%")
        
        # Validate the model
        model.eval()

        val_acc, val_loss = validate(val_loader, model, F.cross_entropy, epoch)
        
        if rank == 0:
            print(f"Epoch {epoch}, Rank {rank}, Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")            


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()