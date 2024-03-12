import torch
from tqdm import tqdm
import torch.distributed as dist

from .zeroth_order import gradient_estimate_randvec, gradient_estimate_coordwise, gradient_estimate_paramwise, learning_rate_estimate_second_order

import numpy as np
import time


def train_dist(train_loader, model, criterion, optimizer, epoch, epoch_pbar=None, verbose=True):
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    for param in model.parameters():
        dist.broadcast(param.data, 0)
    
    model.train()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    if rank != 0: verbose = False

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    for input, label in pbar:
        batch_size = input.size(0)
        batch_worker_from = batch_size * rank // size
        batch_worker_to = batch_size * (rank + 1) // size
        
        input = input[batch_worker_from:batch_worker_to].cuda()
        label = label[batch_worker_from:batch_worker_to].cuda()

        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()

        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)
    
        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        if verbose:
            pbar.set_postfix(train_accuracy=accuracy, train_loss=avg_loss)
        
    accuracy = torch.tensor(num_correct / num_data)
    avg_loss = torch.tensor(sum_loss / num_data)

    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
    accuracy /= dist.get_world_size()

    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss /= dist.get_world_size()

    accuracy = accuracy.item()
    avg_loss = avg_loss.item()

    return accuracy, avg_loss


def train_dist_zo(
    train_loader, model, criterion, optimizer, epoch,
    smoothing=1e-3, query=1, lr_auto=True, lr_max=1e-2, lr_min=1e-5, ge_type='rge',
    config=None, verbose=True
):
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    model.eval()
    
    for param in model.parameters():
        dist.broadcast(param.data, 0)
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    if rank != 0: verbose = False
    
    lr_history = []
    avg_lr = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    
    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        # estimate gradient
        if ge_type == 'rge':
            estimated_gradient = gradient_estimate_randvec(input, label, model, criterion, query=query, smoothing=smoothing)
        elif ge_type == 'cge':
            estimated_gradient = gradient_estimate_coordwise(input, label, model, criterion, smoothing=smoothing)
        elif ge_type == 'paramwise':
            estimated_gradient = gradient_estimate_paramwise(input, label, model, criterion, query=query, smoothing=smoothing)
        
        # all-reduce gradient
        for name, grad in estimated_gradient.items():
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= dist.get_world_size()
            
            
        # estimate learning rate
        if lr_auto:
            lr = learning_rate_estimate_second_order(input, label, model, criterion, estimated_gradient, smoothing=smoothing)
            lr = abs(lr.item()) if lr != 0 else lr_min
            lr = min(max(lr, lr_min), lr_max)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                lr = min(max(lr, lr_min), lr_max)
                param_group['lr'] = lr
                
        lr_history.append(lr)
        avg_lr += lr

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if param.requires_grad:
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
    
    avg_lr /= len(train_loader)
    std_lr = np.std(lr_history)

    if config is not None:
        config['avg_lr'] = avg_lr
        config['std_lr'] = std_lr

    return accuracy, avg_loss