import torch
from tqdm import tqdm
import torch.distributed as dist


def train_dist(train_loader, model, criterion, optimizer, epoch, epoch_pbar=None, verbose=True):
    rank = dist.get_rank()
    
    model.train()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    if rank != 0: verbose = False

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()

        optimizer.step()

        for param in model.parameters():
            dist.broadcast(param.data, 0)

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