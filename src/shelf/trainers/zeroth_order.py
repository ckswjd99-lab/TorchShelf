import torch
from tqdm import tqdm
import numpy as np


def train_zeroth_order(train_loader, model, criterion, epoch, learning_rate=1e-7, weight_decay=5e-4, epsilon=1e-3, verbose=True, momentum=1.0):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    max_grad = 0

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        perturb_seed = np.random.randint(10000000)

        _zo_perturb(model, perturb_seed, scaling_factor = 1 * epsilon)
        output1 = model(input)
        loss1 = criterion(output1, label).item()

        _zo_perturb(model, perturb_seed, scaling_factor = -2 * epsilon)
        output2 = model(input)
        loss2 = criterion(output2, label).item()

        _zo_perturb(model, perturb_seed, scaling_factor = 1 * epsilon)

        projected_gradient = (loss1 - loss2) / (2 * epsilon)

        momentum_buffer = _zo_update(model, perturb_seed, projected_gradient, learning_rate, weight_decay, momentum_buffer, momentum)

        output = model(input)
        loss = criterion(output, label)

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)
    
        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        if verbose:
            pbar.set_postfix(train_accuracy=accuracy, train_loss=avg_loss)
        
    accuracy = num_correct / num_data
    avg_loss = sum_loss / num_data

    return accuracy, avg_loss

def _zo_perturb(model, perturb_seed, scaling_factor=1.0):
    torch.manual_seed(perturb_seed)
    
    for name, param in model.named_parameters():
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data += z * scaling_factor

def _zo_update(model, perturb_seed, projected_gradient, learning_rate, weight_decay, momentum_buffer, momentum):
    torch.manual_seed(perturb_seed)

    for name, param in model.named_parameters():
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        estimated_gradient = projected_gradient * z
        momentum_buffer[name] = momentum * momentum_buffer[name] + (momentum) * estimated_gradient

        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - learning_rate * (estimated_gradient + weight_decay * param.data)
        else:
            param.data = param.data - learning_rate * (estimated_gradient)
        
    return momentum_buffer
        