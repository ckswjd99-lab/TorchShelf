import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_MNIST_dataset(root='./data', batch_size=128):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader
