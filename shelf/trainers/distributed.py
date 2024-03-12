import torch
from tqdm import tqdm
import torch.distributed as dist

from .zeroth_order import gradient_estimate_randvec, learning_rate_estimate_second_order, gradient_fo

import numpy as np
import time


def train_dist(train_loader, model, criterion, optimizer, epoch, epoch_pbar=None, verbose=True):
    rank = dist.get_rank()
    size = dist.get_world_size()
    
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


def train_dist_zo_rge_autolr(
        train_loader, model, criterion, optimizer, epoch, 
        smoothing=1e-3, max_lr=1e-2, query=1, verbose=True, confidence=0.1, clip_loss_diff=1e99, one_way=False, config=None
    ):
    model.eval()

    if config is not None:
        start_time = time.time()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0: verbose = False

    num_query_per_process = query // world_size

    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_history = []
    avg_lr = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader

    momentum_buffer = {}
    sync_model = None
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        if sync_model is not None:
            sync_model.wait()

        estimated_gradient = gradient_estimate_randvec(
            input, label, model, criterion, 
            query=num_query_per_process, smoothing=smoothing, one_way=one_way, clip_loss_diff=clip_loss_diff
        )

        reduced_estimated_gradient = {}
        for name, param in model.named_parameters():
            reduced_estimated_gradient[name] = estimated_gradient[name] / dist.get_world_size()

        for name, param in model.named_parameters():
            dist.reduce(reduced_estimated_gradient[name], op=dist.ReduceOp.SUM, dst=0)

        estimated_gradient = reduced_estimated_gradient

        lr = learning_rate_estimate_second_order(input, label, model, criterion, estimated_gradient, smoothing=smoothing)

        lr = abs(lr.item()) * confidence
        lr = min(lr, max_lr)

        optimizer.zero_grad()
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lr_history.append(lr)
        avg_lr += lr

        for name, param in model.named_parameters():
            param.grad = estimated_gradient[name]
        optimizer.step()

        for param in model.parameters():
            sync_model = dist.broadcast(param.data, 0, async_op=True)

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
        config['time_elapsed'] = time.time() - start_time

    return accuracy, avg_loss