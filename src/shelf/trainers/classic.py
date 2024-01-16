import torch
from tqdm import tqdm


def adjust_learning_rate(optimizer, lr_init, epoch, epoch_freq=30, decay_rate=0.5, minimum_lr=1e-5):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = max(lr_init * (decay_rate ** (epoch // epoch_freq)), minimum_lr)
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lr

def train(train_loader, model, criterion, optimizer, epoch, epoch_pbar=None, verbose=True):
    model.train()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

def validate(val_loader, model, criterion, epoch, verbose=True):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False) if verbose else val_loader
        for input, label in pbar:
            input = input.cuda()
            label = label.cuda()

            output = model(input)
            loss = criterion(output, label)

            _, predicted = torch.max(output.data, 1)
            num_data += label.size(0)
            num_correct += (predicted == label).sum().item()
            sum_loss += loss.item() * label.size(0)
    
            accuracy = num_correct / num_data
            avg_loss = sum_loss / num_data

            if verbose:
                pbar.set_postfix(val_accuracy=accuracy, val_loss=avg_loss)

    accuracy = num_correct / num_data
    avg_loss = sum_loss / num_data

    return accuracy, avg_loss