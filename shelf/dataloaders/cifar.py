import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist

from ._distributed import DataPartitioner

def get_CIFAR10_dataset(root='./data', batch_size=128):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader

def get_CIFAR100_dataset(root='./data', batch_size=128):
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader

def get_CIFAR10_dataset_dist(root='./data', batch_size=128, num_process=1):
    size = num_process
    worker_batch_size = batch_size // size
    partition_sizes = [1.0 / size for _ in range(size)]
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    
    val_dataset = datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    
    train_partition = DataPartitioner(train_dataset, partition_sizes)
    train_partition = train_partition.use(dist.get_rank())

    train_loader = torch.utils.data.DataLoader(
        train_partition,
        batch_size=worker_batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )

    val_partition = DataPartitioner(val_dataset, partition_sizes)
    val_partition = val_partition.use(dist.get_rank())

    val_loader = torch.utils.data.DataLoader(
        val_partition,
        batch_size=worker_batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader