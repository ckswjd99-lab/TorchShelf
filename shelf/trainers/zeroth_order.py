import torch
from tqdm import tqdm
import numpy as np
import math


def train_fo_autolr(train_loader, model, criterion, optimizer, epoch, smoothing=1e-3, max_lr=1e-2, verbose=True, confidence=0.1, config=None):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_history = []
    avg_lr = 0
    estim_precision = 0
    estim_cos_sim = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        estimated_gradient = gradient_fo(input, label, model, criterion)
        lr = learning_rate_estimate_second_order(input, label, model, criterion, estimated_gradient, smoothing=smoothing)

        lr = abs(lr.item()) * confidence
        lr = min(lr, max_lr)

        optimizer.zero_grad()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for name, param in model.named_parameters():
            param.grad = estimated_gradient[name]
        optimizer.step()

        lr_history.append(lr)
        avg_lr += lr

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
    estim_precision /= len(train_loader)
    estim_cos_sim /= len(train_loader)

    if config is not None:
        config['avg_lr'] = avg_lr
        config['std_lr'] = std_lr
        config['estim_precision'] = estim_precision
        config['estim_cos_sim'] = estim_cos_sim

    return accuracy, avg_loss


def train_zo_rge(train_loader, model, criterion, optimizer, epoch, smoothing=1e-3, query=1, verbose=True, one_way=False):
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

        estimated_gradient = gradient_estimate_randvec(input, label, model, criterion, query=query, smoothing=smoothing, one_way=one_way)

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


def train_zo_rge_autolr(
        train_loader, model, criterion, optimizer, epoch, 
        smoothing=1e-3, max_lr=1e-2, query=1, verbose=True, confidence=0.1, clip_loss_diff=1e99, one_way=False, config=None
    ):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_history = []
    avg_lr = 0
    # estim_precision = 0
    # estim_cos_sim = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        estimated_gradient = gradient_estimate_randvec(input, label, model, criterion, query=query, smoothing=smoothing, one_way=one_way, clip_loss_diff=clip_loss_diff)
        # if config is not None:
        #     real_gradient = gradient_fo(input, label, model, criterion)
            
        #     real_gradient_vector = torch.cat([param.view(-1) for param in real_gradient.values()])
        #     estimated_gradient_vector = torch.cat([param.view(-1) for param in estimated_gradient.values()])
        #     norm_estimated = torch.norm(estimated_gradient_vector)

        #     estim_in_real = torch.dot(real_gradient_vector, estimated_gradient_vector) / norm_estimated
        #     estim_precision += norm_estimated / estim_in_real

        #     cos_sim = torch.dot(real_gradient_vector, estimated_gradient_vector) / (torch.norm(real_gradient_vector) * norm_estimated)
        #     estim_cos_sim += cos_sim

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
    # estim_precision /= len(train_loader)
    # estim_cos_sim /= len(train_loader)

    if config is not None:
        config['avg_lr'] = avg_lr
        config['std_lr'] = std_lr
        # config['estim_precision'] = estim_precision
        # config['estim_cos_sim'] = estim_cos_sim

    return accuracy, avg_loss


def train_zo_rge_ha(
        train_loader, model, criterion, optimizer, epoch, 
        smoothing=1e-3, max_lr=1e-2, query=1, verbose=True, confidence=0.1, clip_loss_diff=1e99, one_way=False, config=None
    ):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_history = []
    avg_lr = 0
    estim_precision = 0
    estim_cos_sim = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        estimated_gradient = gradient_estimate_randvec_ha(
            input, label, model, criterion, 
            query=query, smoothing=smoothing, one_way=one_way, clip_loss_diff=clip_loss_diff, max_lr=max_lr
        )
        lr = confidence

        optimizer.zero_grad()
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lr_history.append(lr)
        avg_lr += lr

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

    avg_lr /= len(train_loader)
    std_lr = np.std(lr_history)
    estim_precision /= len(train_loader)
    estim_cos_sim /= len(train_loader)

    if config is not None:
        config['avg_lr'] = avg_lr
        config['std_lr'] = std_lr
        config['estim_precision'] = estim_precision
        config['estim_cos_sim'] = estim_cos_sim

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
def gradient_estimate_randvec(input, label, model, criterion, query=1, smoothing=1e-3, one_way=False, clip_loss_diff=1e99):
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

        loss_difference = min(loss_perturbed - loss_original, clip_loss_diff) / smoothing

        if math.isnan(loss_difference.item()):
            print('NaN detected!!')
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
def gradient_estimate_randvec_ha(input, label, model, criterion, max_lr=1e-2, query=1, smoothing=1e-3, one_way=False, clip_loss_diff=1e99):
    averaged_gradient = {}
    loss_original = criterion(model(input), label)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        averaged_gradient[name] = torch.zeros_like(param.data)

    for _ in range(query):
        estimated_gradient = {}

        for name, param in model.named_parameters():
            estimated_gradient[name] = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) / num_params
            param.data += estimated_gradient[name] * smoothing

        loss_perturbed_pos = criterion(model(input), label)

        for name, param in model.named_parameters():
            param.data -= 2 * estimated_gradient[name] * smoothing

        loss_perturbed_neg = criterion(model(input), label)

        for name, param in model.named_parameters():
            param.data += estimated_gradient[name] * smoothing

        Jz = (loss_perturbed_pos - loss_original) / smoothing
        zHz = (loss_perturbed_pos + loss_perturbed_neg - 2 * loss_original) / (smoothing ** 2)
        zHz = max(zHz, 0)

        step_size = Jz / (zHz + 1e-4)
        step_size = min(step_size, max_lr)
        step_size = max(step_size, -max_lr)

        for name, param in model.named_parameters():
            averaged_gradient[name] += step_size * estimated_gradient[name]

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
        param.data += estimated_gradient[name] * smoothing
    loss_perturbed_pos = criterion(model(input), label)

    # perturb in negative direction
    for name, param in model.named_parameters():
        param.data -= 2 * estimated_gradient[name] * smoothing
    loss_perturbed_neg = criterion(model(input), label)

    # explosion check
    if math.isnan(loss_perturbed_pos.item()) or math.isnan(loss_perturbed_neg.item()) or math.isnan(loss_original.item()):
        print('Explosion detected!!')
        print(f'0, +, -: {loss_original.item()}, {loss_perturbed_pos.item()}, {loss_perturbed_neg.item()}')

        return 0

    # restore the original parameters
    for name, param in model.named_parameters():
        param.data += estimated_gradient[name] * smoothing
    
    # estimate Jz
    Jz = (loss_perturbed_pos - loss_original) / smoothing
    
    # estimate zHz
    zHz = (loss_perturbed_pos + loss_perturbed_neg - 2 * loss_original) / (smoothing ** 2)
    zHz = max(zHz, 0)

    # estimate learning rate
    lr = Jz / (zHz + 1e-4)

    return lr