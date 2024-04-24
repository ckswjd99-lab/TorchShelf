import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import numpy as np
import math

from functools import partial
import torch.func as fc

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
            if '_orig' in name and name.replace('_orig', '_mask') in state_dict:
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
    
    num_pg = len(list(model.parameters()))

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        mask = None
        if '_orig' in name and name.replace('_orig', '_mask') in state_dict:
            mask = state_dict[name.replace('_orig', '_mask')]
        
        for _ in range(int(query/num_pg)):
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
        if '_orig' in name and name.replace('_orig', '_mask') in state_dict:
            mask = state_dict[name.replace('_orig', '_mask')]
        
        for i in range(param.data.numel()):
            if mask is not None and mask.view(-1)[i] == 0: continue
            
            param.data.view(-1)[i] += smoothing
            loss_perturbed = criterion(model(input), label)
            param.data.view(-1)[i] -= smoothing

            estimated_gradient.view(-1)[i] = (loss_perturbed - loss_original) / smoothing

        averaged_gradient[name] += estimated_gradient

    return averaged_gradient


@torch.no_grad()
def gradient_fwd(input, label, model, criterion_functional, query=1, type='rge', **kwargs):
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()

    estimated_grads = {name: torch.zeros_like(p) for name, p in zip(names, params)}

    if type == 'rge':
        for q in range(query):
            perturb_noise = tuple(torch.randn_like(p) for p in params)

            f = partial(criterion_functional, model=model, names=names, buffers=named_buffers, x=input, t=label)
            loss, jvp = fc.jvp(f, (tuple(params),), (perturb_noise,))

            for name, p in zip(names, perturb_noise):
                estimated_grads[name] += jvp * p

        for name in names:
            estimated_grads[name] /= query
    elif type == 'pge':
        backup_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)

        for q in range(query):
            for p in params:
                p.requires_grad_(True)
                perturb_noise = tuple(torch.randn_like(p) if p.requires_grad else torch.zeros_like(p) for p in params)

                f = partial(criterion_functional, model=model, names=names, buffers=named_buffers, x=input, t=label)
                loss, jvp = fc.jvp(f, (tuple(params),), (perturb_noise,))

                for name, p in zip(names, perturb_noise):
                    estimated_grads[name] += jvp * p
                p.requires_grad_(False)

        for p, requires_grad in zip(params, backup_requires_grad):
            p.requires_grad_(requires_grad)

        for name in names:
            estimated_grads[name] /= query
    elif type == 'gge':
        group_dict = kwargs['group_dict']
        group_sizes = kwargs['group_sizes']
        num_groups = kwargs['num_groups']
        scaled_grads = kwargs['scaled_grads']

        for name, p in zip(names, params):
            scaled_grads[name] = torch.zeros_like(p)

        for q in range(query):
            perturb_noise_total = tuple(torch.randn_like(p) for p in params)
            for group_idx in range(num_groups):
                group_masks = tuple(param_group == group_idx for param_group in group_dict.values())

                group_size = group_sizes[group_idx]
                if group_size == 1:
                    perturb_noise = tuple(mask.float() for p, mask in zip(params, group_masks))
                    norm_noise = 1
                else:
                    perturb_noise = tuple(pnoise * mask for pnoise, mask in zip(perturb_noise_total, group_masks))
                    norm_noise = sum(p.norm() ** 2 for p in perturb_noise)
                
                if norm_noise == 0:
                    norm_noise = 1

                f = partial(criterion_functional, model=model, names=names, buffers=named_buffers, x=input, t=label)
                loss, jvp = fc.jvp(f, (tuple(params),), (perturb_noise,))

                for name, p, in zip(names, perturb_noise):
                    estimated_grads[name] += jvp * p
                    scaled_grads[name] += jvp * p / norm_noise

        for name in names:
            estimated_grads[name] /= query
            scaled_grads[name] /= query

    else:
        raise NotImplementedError

    return estimated_grads


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
def learning_rate_estimate_second_order(input, label, model, criterion, estimated_gradient, smoothing=1e-3, scale_by_num_params=True):
    if scale_by_num_params:
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
    
    if zHz < 0:
        if loss_perturbed_pos < loss_perturbed_neg:
            lr = torch.tensor(smoothing)
        else:
            lr = torch.tensor(-smoothing)
    else:
        if loss_perturbed_pos > loss_original and loss_perturbed_neg > loss_original:
            lr = torch.tensor(0)
        else:
            lr = Jz / (zHz + 1e-4)
    
    return lr

def group_by_gradient_exp(estimated_gradient, num_groups, descending=True, level_noise=0):
    def calc_r_by_gnum(N, d):
        equation = np.poly1d([1] + [0 for _ in range(N-1)] + [-d, d-1], False)
        roots = np.roots(equation)
        roots = roots[np.isreal(roots)]
        r = np.real(np.max(roots))

        if r <= 1:
            raise ValueError("r must be greater than 1")

        return r

    all_gradients = torch.cat([grad.flatten() for grad in estimated_gradient.values()]).abs()
    all_gradients = torch.sort(all_gradients + torch.normal(0, level_noise, size=all_gradients.size(), device=all_gradients.device), descending=True).values
    num_params = all_gradients.size(0)

    # Calculate r
    r = calc_r_by_gnum(num_groups, num_params)

    # Find milestones
    milestones = []
    group_sizes = []
    group_size = 1
    group_start_idx = 0
    for group_idx in range(num_groups):
        milestones.append(all_gradients[group_start_idx])
        group_start_idx += math.floor(group_size)
        group_sizes.append(math.floor(group_size))
        group_size *= r
    # WARNING: it sometimes makes empty group. should be fixed
    
    milestones[-1] = all_gradients[-1]

    # Group the parameters
    group_dict = {}
    for name, grad in estimated_gradient.items():
        group_dict[name] = torch.zeros_like(grad)
        for i, milestone in enumerate(milestones[::-1]):
            group_idx = num_groups - i - 1
            if descending:
                group_dict[name][grad.abs() >= milestone] = group_idx
            else:
                group_dict[name][grad.abs() <= milestone] = group_idx

    return group_dict, group_sizes