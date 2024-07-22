import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .augmentation import CIFAR10Policy

def get_CIFAR10_dataset(root='./data', batch_size=128, augmentation=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(fillcolor=(125, 122, 113)) if augmentation else transforms.Compose([]),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader

def get_CIFAR100_dataset(root='./data', batch_size=128):
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(fillcolor=(125, 122, 113)),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader
