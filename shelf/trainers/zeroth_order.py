import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import numpy as np
import math

def train_zo(
    train_loader, model, criterion, optimizer, epoch,
    smoothing=1e-3, query=1, lr_auto=True, lr_max=1e-2, lr_min=1e-5, ge_type='rge',
    config=None, verbose=True
):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
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


@torch.no_grad()
def gradient_estimate_randvec(input, label, model, criterion, query=1, smoothing=1e-3, one_way=False, clip_loss_diff=1e99):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)
    
    state_dict = model.state_dict()

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        averaged_gradient[name] = torch.zeros_like(param.data)

    for q in range(query):
        estimated_gradient = {}


        for name, param in model.named_parameters():
            if not param.requires_grad: continue
        
            estimated_gradient[name] = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if 'weight_orig' in name:
                mask = state_dict[name.replace('_orig', '_mask')]
                estimated_gradient[name] *= mask
            param.data += estimated_gradient[name] * smoothing

        loss_perturbed = criterion(model(input), label)

        for name, param in model.named_parameters():
            if not param.requires_grad: continue
        
            param.data -= estimated_gradient[name] * smoothing

        loss_difference = min(loss_perturbed - loss_original, clip_loss_diff) / smoothing

        if math.isnan(loss_difference.item()):
            print('NaN detected!!', q)
            print(f'loss_diff: {loss_difference}')
            print(f'0, +: {loss_original.item()}, {loss_perturbed.item()}')
            continue

        if one_way and loss_difference > 0:
            query -= 1
            continue

        for param_name in estimated_gradient.keys():
            estimated_gradient[param_name] *= loss_difference

        for param_name in estimated_gradient.keys():
            averaged_gradient[param_name] += estimated_gradient[param_name]

    for param_name in averaged_gradient.keys():
        averaged_gradient[param_name] /= query
    
    return averaged_gradient


@torch.no_grad()
def gradient_estimate_paramwise(input, label, model, criterion, query=1, smoothing=1e-3, one_way=False, clip_loss_diff=1e99):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)
    
    state_dict = model.state_dict()

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        averaged_gradient[name] = torch.zeros_like(param.data)

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        mask = None
        if 'weight_orig' in name:
            mask = state_dict[name.replace('_orig', '_mask')]
        
        for _ in range(query):
            gaussian_noise = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            if mask is not None:
                gaussian_noise *= mask
            
            param.data += gaussian_noise * smoothing

            loss_perturbed = criterion(model(input), label)

            param.data -= gaussian_noise * smoothing

            loss_difference = max(min(loss_perturbed - loss_original, clip_loss_diff), -clip_loss_diff) / smoothing

            gaussian_noise *= loss_difference

            averaged_gradient[name] += gaussian_noise

        averaged_gradient[name] /= query
    
    return averaged_gradient


@torch.no_grad()
def gradient_estimate_coordwise(input, label, model, criterion, smoothing):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)
    
    state_dict = model.state_dict()

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        averaged_gradient[name] = torch.zeros_like(param.data)

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        mask = None
        
        estimated_gradient = torch.zeros_like(param.data)
        if 'weight_orig' in name:
            mask = state_dict[name.replace('_orig', '_mask')]
        
        for i in range(param.data.numel()):
            if mask is not None and mask.view(-1)[i] == 0: continue
            
            param.data.view(-1)[i] += smoothing
            loss_perturbed = criterion(model(input), label)
            param.data.view(-1)[i] -= smoothing

            estimated_gradient.view(-1)[i] = (loss_perturbed - loss_original) / smoothing

        averaged_gradient[name] += estimated_gradient

    return averaged_gradient


def gradient_fo(input, label, model, criterion):
    model.train()
    gradient = {}

    output = model(input)
    loss = criterion(output, label)
    loss.backward()

    for name, param in model.named_parameters():
        gradient[name] = param.grad.clone()
        param.grad.zero_()
    
    return gradient


@torch.no_grad()
def learning_rate_estimate_second_order(input, label, model, criterion, estimated_gradient, smoothing=1e-3):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    smoothing /= num_params
    
    # measure original loss
    loss_original = criterion(model(input), label)

    # perturb in positive direction
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        param.data += estimated_gradient[name] * smoothing
    loss_perturbed_pos = criterion(model(input), label)

    # perturb in negative direction
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        param.data -= 2 * estimated_gradient[name] * smoothing
    loss_perturbed_neg = criterion(model(input), label)

    # explosion check
    if math.isnan(loss_perturbed_pos.item()) or math.isnan(loss_perturbed_neg.item()) or math.isnan(loss_original.item()):
        print('Explosion detected!!')
        print(f'0, +, -: {loss_original.item()}, {loss_perturbed_pos.item()}, {loss_perturbed_neg.item()}')

        return 0

    # restore the original parameters
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        param.data += estimated_gradient[name] * smoothing
    
    # estimate Jz
    Jz = (loss_perturbed_pos - loss_original) / smoothing
    
    # estimate zHz
    zHz = (loss_perturbed_pos + loss_perturbed_neg - 2 * loss_original) / (smoothing ** 2)
    zHz = max(zHz, 0)

    # estimate learning rate
    lr = Jz / (zHz + 1e-4)

    return lr