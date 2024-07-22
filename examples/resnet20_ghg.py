import torch
import torch.func as fc
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.func import jvp, grad

import numpy as np
import math
import time

from tqdm import tqdm
import matplotlib.pyplot as plt

from import_shelf import shelf
from shelf.models.resnet.etc import resnet20
from shelf.models.resnet import ResNet18
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.pruners.scoring import vthvp

## HYPERPARAMS ##

EPOCHS = 10
BATCH_SIZE = 128
NUM_QUERY = 2000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


### DATA LOADING ###

train_loader, val_loader = get_CIFAR10_dataset(batch_size=BATCH_SIZE)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_loader.dataset.data[i])
    plt.title(classes[train_loader.dataset.targets[i]])
    plt.axis('off')


## MODEL ##
model = resnet20().to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

print(model)
print(f"Model has {num_params} parameters")

pnames = list(model.state_dict().keys())
params = list(model.parameters())


## OTHERS ##
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def functional_xent(params, buffers, names, model, x, t):
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return F.cross_entropy(y, t)


## TRAINING ##
torch.no_grad()
for epoch in range(EPOCHS):

    pbar = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        get_loss_with_params = partial(functional_xent, buffers={}, names=list(model.state_dict().keys()), model=model, x=inputs, t=targets)

        estimated_gradient = list(torch.zeros_like(p) for p in params)
        for _ in range(NUM_QUERY):
            tangent = list(torch.rand_like(p) for p in params)
            jvp_val = jvp(get_loss_with_params, (params,), (tangent,))[1]

            estimated_gradient = [eg + jvp_val * tan for eg, tan in zip(estimated_gradient, tangent)]
        
        estimated_gradient = [eg / NUM_QUERY for eg in estimated_gradient]

        tht = max(tht, 1e-6)

        for param, tan in zip(params, estimated_gradient):
            new_param = jvp_val * tan / tht
            param.data = param.data - new_param

        output = model(inputs, targets)
        loss = criterion(output, targets)

        accuracy = (output.argmax(1) == targets).float().mean()
        
        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item()} | Accuracy: {accuracy.item()}")