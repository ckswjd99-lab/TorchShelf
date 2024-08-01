import torch
import torch.func as fc
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.func import jvp, grad

from tqdm import tqdm

def functional_xent(params, buffers, names, model, x, t):
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return F.cross_entropy(y, t)

@torch.no_grad()
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

@torch.no_grad()
def vthvp(f, primals, tangents):
    def jvp_first(_primals):
        return jvp(f, _primals, tangents)[1]
    
    return jvp(jvp_first, (primals,), (tangents,))[1]

@torch.no_grad()
def get_grasp_score(input, label, model, criterion=functional_xent):
    names = list(dict(model.named_parameters()).keys())
    params = list(model.parameters())
    buffers = dict(model.named_buffers())

    get_loss_with_params = partial(criterion, buffers=buffers, names=names, model=model, x=input, t=label)

    tangent = grad(get_loss_with_params)(params)

    hvp_value = hvp(get_loss_with_params, (params,), (tangent,))

    grasp_score = [-p * h for p, h in zip(params, hvp_value)]

    return grasp_score

@torch.no_grad()
def get_hvp_score(input, label, model, criterion=functional_xent):
    names = list(dict(model.named_parameters()).keys())
    params = list(model.parameters())
    buffers = dict(model.named_buffers())

    get_loss_with_params = partial(criterion, buffers=buffers, names=names, model=model, x=input, t=label)

    tangent = grad(get_loss_with_params)(params)

    hvp_value = hvp(get_loss_with_params, (params,), (tangent,))

    score = [hg for hg in hvp_value]

    return score

@torch.no_grad()
def get_grasp_score_dict(dataloader, model, criterion=functional_xent):
    names = list(dict(model.named_parameters()).keys())
    params = list(model.parameters())
    buffers = dict(model.named_buffers())

    grasp_score_dict = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}

    pbar = tqdm(dataloader, leave=False, desc="Computing GraSP Score")
    num_steps = 0
    for input, label in pbar:
        input, label = input.to('cuda'), label.to('cuda')
        
        grasp_score = get_grasp_score(input, label, model, criterion=criterion)
        for pname, score in zip(names, grasp_score):
            grasp_score_dict[pname] += score

        num_steps += 1

    for score in grasp_score_dict:
        grasp_score_dict[score] /= num_steps

    return grasp_score_dict

@torch.no_grad()
def get_abs_gradient_score(input, label, model, criterion=functional_xent):
    names = list(dict(model.named_parameters()).keys())
    params = list(model.parameters())
    buffers = dict(model.named_buffers())

    get_loss_with_params = partial(criterion, buffers=buffers, names=names, model=model, x=input, t=label)

    gradient = grad(get_loss_with_params)(params)

    score = [torch.abs(grad) for grad in gradient]

    return score

@torch.no_grad()
def get_fwd_grasp_score(input, label, model, tangent, criterion=functional_xent):
    names = list(model.state_dict().keys())
    params = list(model.parameters())
    buffers = {}

    get_loss_with_params = partial(criterion, buffers=buffers, names=names, model=model, x=input, t=label)

    def get_fwd_gradient(params):
        jvp_value = jvp(get_loss_with_params, (params,), (tangent,))[1]
        return [jvp_value * t for t in tangent]
    
    fwd_Hg = jvp(get_fwd_gradient, (params,), (tangent,))[1]

    grasp_score = [-p * h for p, h in zip(params, fwd_Hg)]

    return grasp_score

@torch.no_grad()
def get_zo_grasp_score(data_loader, model, smoothing=1e-3, device='cuda', criterion=nn.CrossEntropyLoss(), query=1):
    model.eval()

    grasp_score = {pname: torch.zeros_like(param) for pname, param in model.named_parameters()}

    pbar = tqdm(data_loader, leave=False, desc="Computing ZO-GraSP Score", total=query)
    num_steps = 0
    for input, label in pbar:
        input, label = input.to(device), label.to(device)

        num_steps += 1
        
        pnoise = {pname: torch.randn_like(param) for pname, param in model.named_parameters()}

        loss_orig = criterion(model(input), label)

        for pname, param in model.named_parameters():
            param.data += smoothing * pnoise[pname]
        
        loss_pos = criterion(model(input), label)

        for pname, param in model.named_parameters():
            param.data -= 2 * smoothing * pnoise[pname]

        loss_neg = criterion(model(input), label)

        for pname, param in model.named_parameters():
            param.data += smoothing * pnoise[pname]

        hg_vec = {
            pname: -param * pnoise[pname] * (loss_pos + loss_neg - 2 * loss_orig) / (smoothing ** 2)
            for pname, param in model.named_parameters()
        }

        for pname, noise in pnoise.items():
            grasp_score[pname] += hg_vec[pname]

        if num_steps >= query:
            break

    if num_steps < query:
        print(f"Warning: The number of steps is less than the query. The number of steps is {num_steps} and the query is {query}.")
        
    for score in grasp_score:
        grasp_score[score] /= num_steps
    
    return grasp_score