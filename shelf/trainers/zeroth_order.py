import torch
from tqdm import tqdm
import numpy as np


def train_zo_rge(train_loader, model, criterion, optimizer, epoch, smoothing=1e-3, query=1, verbose=True):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        estimated_gradient = gradient_estimate_randvec(input, label, model, criterion, query=query, smoothing=smoothing)

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            param.grad = estimated_gradient[name]
        optimizer.step()

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


def train_zo_cge(train_loader, model, criterion, optimizer, epoch, smoothing=1e-3, verbose=True):
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

        estimated_gradient = gradient_estimate_coordwise(input, label, model, criterion, smoothing=smoothing)

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            param.grad = estimated_gradient[name]
        optimizer.step()

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


@torch.no_grad()
def gradient_estimate_randvec(input, label, model, criterion, query=1, smoothing=1e-3):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)

    for name, param in model.named_parameters():
        averaged_gradient[name] = torch.zeros_like(param.data)

    for _ in range(query):
        estimated_gradient = {}


        for name, param in model.named_parameters():
            estimated_gradient[name] = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data += estimated_gradient[name] * smoothing

        loss_perturbed = criterion(model(input), label)

        for name, param in model.named_parameters():
            param.data -= estimated_gradient[name] * smoothing

        loss_difference = (loss_perturbed - loss_original) / smoothing

        for param_name in estimated_gradient.keys():
            estimated_gradient[param_name] *= loss_difference

        for param_name in estimated_gradient.keys():
            averaged_gradient[param_name] += estimated_gradient[param_name]

    for param_name in averaged_gradient.keys():
        averaged_gradient[param_name] /= query
    
    return averaged_gradient


@torch.no_grad()
def gradient_estimate_coordwise(input, label, model, criterion, smoothing):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)

    for name, param in model.named_parameters():
        averaged_gradient[name] = torch.zeros_like(param.data)

    for name, param in model.named_parameters():
        estimated_gradient = torch.zeros_like(param.data)

        for i in range(param.data.numel()):
            param.data.view(-1)[i] += smoothing
            loss_perturbed = criterion(model(input), label)
            param.data.view(-1)[i] -= smoothing

            estimated_gradient.view(-1)[i] = (loss_perturbed - loss_original) / smoothing

        averaged_gradient[name] += estimated_gradient

    return averaged_gradient
