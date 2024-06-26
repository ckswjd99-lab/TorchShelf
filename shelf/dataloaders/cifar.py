import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .augmentation import CIFAR10Policy

def get_CIFAR10_dataset(root='./data', batch_size=128, hard_aug=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if hard_aug:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(fillcolor=(125, 122, 113)),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
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

def get_CIFAR100_dataset(root='./data', batch_size=128, hard_aug=False):
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    if hard_aug:
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
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
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
