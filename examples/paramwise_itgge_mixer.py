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
from shelf.models.transformer import VisionTransformer
from shelf.models.mlp_mixer import MLPMixer
from shelf.dataloaders.cifar import get_CIFAR10_dataset
from shelf.trainers.classic import train, validate
from shelf.trainers.zeroth_order import gradient_fwd, group_by_gradient_exp
from shelf.pruners import global_unstructured_L1, undo_pruning, get_layer_sparsity


## HYPERPARAMS ##

BATCH_SIZE = 128
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

model = MLPMixer(
    in_channels=3,
    img_size=32, 
    patch_size=4, 
    hidden_size=128, 
    hidden_s=512, 
    hidden_c=64, 
    num_layers=8, 
    num_classes=10, 
    drop_p=0.,
    off_act=False,
    is_cls_token=True
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

print(model)
print(f"Model has {num_params} parameters")


## OTHERS ##
criterion = nn.CrossEntropyLoss()

def functional_xent(
    params,
    buffers,
    names,
    model,
    x,
    t,
):
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return F.cross_entropy(y, t)

def get_real_gradient(input, label, model, criterion):
    output = model(input)
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()

    gradient_dict = {}
    for name, param in model.named_parameters():
        gradient_dict[name] = param.grad.clone()

    model.zero_grad()

    return gradient_dict


calc_r_by_gnum_cache = {}

def calc_r_by_gnum(N, d):
    if N not in calc_r_by_gnum_cache:
        calc_r_by_gnum_cache[N] = {}

    if d in calc_r_by_gnum_cache[N]:
        return calc_r_by_gnum_cache[N][d]

    equation = np.poly1d([1] + [0 for _ in range(N-1)] + [-d, d-1], False)
    roots = np.roots(equation)
    roots = roots[np.isreal(roots)]
    r = np.real(np.max(roots))

    calc_r_by_gnum_cache[N][d] = r

    if r <= 1:
        raise ValueError("r must be greater than 1")

    return r

@torch.no_grad()
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

# @torch.no_grad()
@torch.enable_grad()
def get_grasp_score(input, label, model):
    names = list(model.state_dict().keys())
    params = list(model.parameters())
    buffers = {}

    get_loss_with_params = partial(functional_xent, buffers=buffers, names=names, model=model, x=input, t=label)

    tangent = grad(get_loss_with_params)(params)

    hvp_value = hvp(get_loss_with_params, (params,), (tangent,))

    grasp_score = [-p * h for p, h in zip(params, hvp_value)]

    return grasp_score

@torch.no_grad()
def get_fwd_grasp_score(input, label, model, tangent):
    names = list(model.state_dict().keys())
    params = list(model.parameters())
    buffers = {}

    get_loss_with_params = partial(functional_xent, buffers=buffers, names=names, model=model, x=input, t=label)

    def get_fwd_gradient(params):
        jvp_value = jvp(get_loss_with_params, (params,), (tangent,))[1]
        return [jvp_value * t for t in tangent]
    
    fwd_Hg = jvp(get_fwd_gradient, (params,), (tangent,))[1]

    grasp_score = [-p * h for p, h in zip(params, fwd_Hg)]

    return grasp_score

@torch.no_grad()
def gradient_estimate_pwitgge(input, label, model, criterion, logbase, smoothing=1e-3, real_gradient=None, init_momentum=None, layerwise_pruning_ratio=None, config=None):

    param_names = [ name for name, _ in model.named_parameters() ]
    param_list = [ p for p in model.parameters() ]

    # grasp_score = get_grasp_score(input, label, model)
    # tangent = list(torch.rand_like(p) for p in model.parameters())
    # grasp_score = get_grasp_score(input, label, model)

    # grasp_score_dict = { name: score for name, score in zip(param_names, grasp_score) }

    total_query = 0
    # original_loss = criterion(model(input), label)

    estimated_gradient = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    for pname, param in zip(param_names, param_list):
        param_size = param.numel()
        if layerwise_pruning_ratio is not None:
            if 'weight' in pname:
                num_groups = (1 - layerwise_pruning_ratio[pname.split('.weight')[0]]) * param_size
            else:
                num_groups = param_size * 0.01
        else:
            num_groups = param_size * 0.01
        num_groups = int(np.ceil(num_groups))
        num_groups = max(num_groups, 1)
        
        # num_iter =  max(num_groups // 4, 1)
        num_iter = 1

        r = calc_r_by_gnum(num_groups, param_size)

        total_query += num_groups * num_iter

        if init_momentum is not None:
            iter_estimate = init_momentum[pname]
        else:
            iter_estimate = torch.randn_like(param) * smoothing
        # iter_estimate = torch.exp(-grasp_score_dict[pname])
        
        iter_grouping = torch.zeros_like(param)

        real_pgrad_flat = real_gradient[pname].view(-1)

        for iter_idx in range(num_iter):
            # sorted estimated gradient
            iter_estimate_abs = (iter_estimate.view(-1)).abs()
            iter_estimate_sorted = torch.sort(iter_estimate_abs, descending=True).values

            # group using estimated gradient
            milestones = []
            group_sizes = []
            group_size = 1
            group_start_idx = 0
            for group_idx in range(num_groups):
                milestones.append(iter_estimate_sorted[group_start_idx])
                now_group_size = math.floor(group_size)
                group_start_idx += now_group_size
                group_sizes.append(now_group_size)
                group_size *= r
            
            milestones[-1] = iter_estimate_sorted[-1]

            # group idx buffer
            for i, milestone in enumerate(milestones[::-1]):
                group_idx = num_groups - i - 1
                iter_grouping[iter_estimate >= milestone] = group_idx
            
            # estimate and update
            iter_estimate = torch.zeros_like(param)
            perturbing_noise = torch.randn_like(param) * smoothing

            gpnoises = [ perturbing_noise * (iter_grouping == group_idx).float() for group_idx in range(num_groups) ]
            gpnoises = [ gpnoise.view(-1) for gpnoise in gpnoises ]
            gpnoises_mat = torch.stack(gpnoises, dim=0)
            jvp_values = gpnoises_mat @ real_pgrad_flat
            iter_estimate = jvp_values @ gpnoises_mat
            iter_estimate = iter_estimate.view(param.size())

            # estimated_gradient[pname] = iter_estimate

            # for group_idx in range(num_groups):
                # param.data += mask * perturbing_noise
                
                # perturbed_loss_pos = criterion(model(input), label)

                # param.data -= 2 * mask * perturbing_noise

                # perturbed_loss_neg = criterion(model(input), label)

                # param.data += mask * perturbing_noise

                # iter_estimate += (perturbed_loss_pos - perturbed_loss_neg) / (2 * smoothing) * mask * perturbing_noise
            
            estimated_gradient[pname] = estimated_gradient[pname] * (1 - 1/num_iter) + iter_estimate * (1/num_iter)
            
        # print(f"Gradient for {param_size:6d} with {num_groups:5d} groups X {num_iter} iters: Cossim {F.cosine_similarity(real_gradient[pname].view(-1), estimated_gradient[pname].view(-1), dim=0)}")
    if config is not None:
        config['cossim'] = cossim_dict(real_gradient, estimated_gradient)
        config['total_query'] = total_query
    
    # print(f"Total query is {total_query}")
    # print(f"Total cosine similarity is {cossim_dict(real_gradient, estimated_gradient)}")
    # print("")
    
    return estimated_gradient

def cossim_dict(a, b):
    a_flat = torch.cat([ p.view(-1) for p in a.values() ])
    b_flat = torch.cat([ p.view(-1) for p in b.values() ])

    return F.cosine_similarity(a_flat, b_flat, dim=0)


# GET PRUNING RATIO

def get_layerwise_pruning_ratio(model, dataloader, prune_ratio):
    param_names = [ name for name, _ in model.named_parameters() ]
    x, t = next(iter(dataloader))
    x, t = x.to(DEVICE), t.to(DEVICE)
    
    # TRIALS = 2
    # fwd_grasp_scores = []

    # for _ in range(TRIALS):

    #     tangent = list(torch.rand_like(p) for p in model.parameters())
    #     fwd_grasp_score = get_fwd_grasp_score(x, t, model, tangent)
    #     fwd_grasp_scores.append(fwd_grasp_score)

    # fwd_grasp_avg = list(torch.zeros_like(p) for p in model.parameters())
    # for fwd_grasp_score in fwd_grasp_scores:
    #     for idx, score in enumerate(fwd_grasp_score):
    #         fwd_grasp_avg[idx] += score / TRIALS


    grasp_score_init = get_grasp_score(x, t, model)
    grasp_score_init_dict = { name: score for name, score in zip(param_names, grasp_score_init) }

    # prune model with grasp score
    global_unstructured_L1(model, grasp_score_init_dict, prune_ratio)

    # get layerwise sparsity
    layer_sparsity = get_layer_sparsity(model)

    # undo pruning
    undo_pruning(model)

    return layer_sparsity


NUM_EPOCH = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01 ** (1/NUM_EPOCH))

estimated_gradient = None

print(optimizer.state_dict())

for epoch in range(NUM_EPOCH):

    start_time = time.time()

    layerwise_pruning_ratio = get_layerwise_pruning_ratio(model, train_loader, 0.99)

    train_loss_sum = 0
    train_acc_sum = 0
    learning_rate = optimizer.param_groups[0]['lr']

    cossim_sum = 0

    train_loss_avg = 0
    train_acc_avg = 0
    cossim_avg = 0

    pbar = tqdm(train_loader)
    pbar_idx = 0
    for input, label in pbar:
        input, label = input.to(DEVICE), label.to(DEVICE)

        config = {}

        real_gradient = get_real_gradient(input, label, model, criterion)
        estimated_gradient = gradient_estimate_pwitgge(input, label, model, criterion, 2, real_gradient=real_gradient, init_momentum=None, config=config, layerwise_pruning_ratio=layerwise_pruning_ratio)

        # apply estimated gradient

        for pname, param in model.named_parameters():
            param.grad = estimated_gradient[pname]
        
        optimizer.step()
        optimizer.zero_grad()

        loss = criterion(model(input), label)

        num_correct = (model(input).argmax(1) == label).sum().item()
        accuracy = num_correct / input.size(0)

        train_loss_sum += loss.item()
        # train_loss_avg = train_loss_sum / (pbar_idx + 1)
        if train_loss_avg == 0:
            train_loss_avg = loss.item()
        train_loss_avg = train_loss_avg * 0.99 + loss.item() * 0.01

        train_acc_sum += accuracy
        if train_acc_avg == 0:
            train_acc_avg = accuracy
        # train_acc_avg = train_acc_sum / (pbar_idx + 1)
        train_acc_avg = train_acc_avg * 0.99 + accuracy * 0.01

        cossim_sum += config['cossim']
        if cossim_avg == 0:
            cossim_avg = config['cossim']
        # cossim_avg = cossim_sum / (pbar_idx + 1)
        cossim_avg = cossim_avg * 0.99 + config['cossim'] * 0.01
        
        pbar.set_description(f"Loss: {train_loss_avg:.4f}, Acc: {train_acc_avg*100:2.2f}, Cossim: {cossim_avg:.4f}, Query: {config['total_query']:6d}")

        pbar_idx += 1

    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    scheduler.step()

    end_time = time.time()
    print(f"Epoch {epoch:4d}/{NUM_EPOCH:4d} : LR {learning_rate:.4e} | CosSim: {cossim_avg:1.4f}, Training Loss: {train_loss_avg:.4f}, Training Acc: {train_acc_avg*100:2.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:2.2f} | Time: {end_time - start_time:.2f}s")

    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"./saves/paramwise_itgge_resnet/epoch_{epoch:04d}.pt")

