import torch
import torch.nn as nn
import math


def mutate_linear_kaiming(module_linear, in_feature, out_feature):
    if isinstance(module_linear, nn.Linear) is False:
        raise TypeError('The module must be a linear layer.')

    new_linear = nn.Linear(in_feature, out_feature)

    nn.init.kaiming_uniform_(new_linear.weight.data, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_linear.weight.data)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(new_linear.bias.data, -bound, bound)

    new_linear.weight.data[:module_linear.weight.shape[0], :module_linear.weight.shape[1]] = module_linear.weight.data
    new_linear.bias.data[:module_linear.bias.shape[0]] = module_linear.bias.data

    new_linear.weight.data = new_linear.weight.data.to(module_linear.weight.data.device)
    new_linear.bias.data = new_linear.bias.data.to(module_linear.bias.data.device)

    return new_linear.to(module_linear.weight.data.device)

def mutate_conv2d_kaiming(module_conv2d, in_channel, out_channel):
    if isinstance(module_conv2d, nn.Conv2d) is False:
        raise TypeError('The module must be a 2D convolutional layer.')

    new_conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=module_conv2d.kernel_size, stride=module_conv2d.stride, padding=module_conv2d.padding, dilation=module_conv2d.dilation, groups=module_conv2d.groups, bias=module_conv2d.bias is not None)

    nn.init.kaiming_uniform_(new_conv2d.weight.data, a=math.sqrt(5))
    if module_conv2d.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_conv2d.weight.data)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(new_conv2d.bias.data, -bound, bound)

    new_conv2d.weight.data[:module_conv2d.weight.shape[0], :module_conv2d.weight.shape[1], :, :] = module_conv2d.weight.data.to(new_conv2d.weight.data.device)
    if module_conv2d.bias is not None:
        new_conv2d.bias.data[:module_conv2d.bias.shape[0]] = module_conv2d.bias.data.to(new_conv2d.bias.data.device)

    new_conv2d.weight.data = new_conv2d.weight.data.to(module_conv2d.weight.data.device)
    if module_conv2d.bias is not None:
        new_conv2d.bias.data = new_conv2d.bias.data.to(module_conv2d.bias.data.device)

    return new_conv2d.to(module_conv2d.weight.data.device)


def mutate_batchnorm2d_identity(module_bn2d, num_channel):
    if isinstance(module_bn2d, nn.BatchNorm2d) is False:
        raise TypeError('The module must be a batch normalization layer.')

    new_bn2d = nn.BatchNorm2d(num_channel, eps=module_bn2d.eps, momentum=module_bn2d.momentum, affine=module_bn2d.affine, track_running_stats=module_bn2d.track_running_stats)

    new_bn2d.weight.data = torch.ones_like(new_bn2d.weight.data)
    new_bn2d.bias.data = torch.zeros_like(new_bn2d.bias.data)

    new_bn2d.weight.data[:module_bn2d.weight.shape[0]] = module_bn2d.weight.data.to(new_bn2d.weight.data.device)
    new_bn2d.bias.data[:module_bn2d.bias.shape[0]] = module_bn2d.bias.data.to(new_bn2d.bias.data.device)

    new_bn2d.weight.data = new_bn2d.weight.data.to(module_bn2d.weight.data.device)
    new_bn2d.bias.data = new_bn2d.bias.data.to(module_bn2d.bias.data.device)

    return new_bn2d.to(module_bn2d.weight.data.device)

