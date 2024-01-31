import torch
import torch.nn as nn
import torch.nn.functional as F

from .mutator import mutate_linear_kaiming, mutate_conv2d_kaiming, mutate_batchnorm2d_identity

class MutableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MutableBasicBlock, self).__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    def grow_input_tobe(self, new_in_planes):
        self.in_planes = new_in_planes

        self.conv1 = mutate_conv2d_kaiming(self.conv1, self.in_planes, self.planes)
        
    
    def grow_output_tobe(self, new_planes):
        self.planes = new_planes

        self.conv1 = mutate_conv2d_kaiming(self.conv1, self.in_planes, self.planes)
        self.conv2 = mutate_conv2d_kaiming(self.conv2, self.planes, self.planes)
        self.bn1 = mutate_batchnorm2d_identity(self.bn1, self.planes)
        self.bn2 = mutate_batchnorm2d_identity(self.bn2, self.planes)
        if len(self.shortcut) > 0:
            self.shortcut[0] = mutate_conv2d_kaiming(self.shortcut[0], self.in_planes, self.expansion * self.planes)
            self.shortcut[1] = mutate_batchnorm2d_identity(self.shortcut[1], self.expansion * self.planes)
    
    def grow_tobe(self, new_in_planes, new_planes):
        self.grow_input_tobe(new_in_planes)
        self.grow_output_tobe(new_planes)

        


class MutableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(MutableBottleneck, self).__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    def grow_output(self, n):
        self.conv1 = mutate_conv2d_kaiming(self.conv1, self.in_planes, self.planes + n)
        self.conv2 = mutate_conv2d_kaiming(self.conv2, self.planes, self.planes + n)
        self.conv3 = mutate_conv2d_kaiming(self.conv3, self.planes, self.planes + n)
        self.bn1 = mutate_batchnorm2d_identity(self.bn1, self.planes + n)
        self.bn2 = mutate_batchnorm2d_identity(self.bn2, self.planes + n)
        self.bn3 = mutate_batchnorm2d_identity(self.bn3, self.planes + n)
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut[0] = mutate_conv2d_kaiming(self.shortcut[0], self.in_planes, self.planes + n)
            self.shortcut[1] = mutate_batchnorm2d_identity(self.shortcut[1], self.planes + n)

        self.planes += n


class MutableResNet18(nn.Module):
    def __init__(self, input_size, num_output, input_channel=3, shrink_ratio=1.0):
        super(MutableResNet18, self).__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.input_channel = input_channel
        self.num_output = num_output
        self.shrink_ratio = shrink_ratio

        # input: Tensor[batch_size, input_channel, input_size[0], input_size[1]]
        self.conv1 = nn.Conv2d(self.input_channel, int(64 * shrink_ratio), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * shrink_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            MutableBasicBlock(int(64 * shrink_ratio), int(64 * shrink_ratio)),
            MutableBasicBlock(int(64 * shrink_ratio), int(64 * shrink_ratio))
        )
        self.layer2 = nn.Sequential(
            MutableBasicBlock(int(64 * shrink_ratio), int(128 * shrink_ratio), stride=2),
            MutableBasicBlock(int(128 * shrink_ratio), int(128 * shrink_ratio))
        )
        self.layer3 = nn.Sequential(
            MutableBasicBlock(int(128 * shrink_ratio), int(256 * shrink_ratio), stride=2),
            MutableBasicBlock(int(256 * shrink_ratio), int(256 * shrink_ratio))
        )
        self.layer4 = nn.Sequential(
            MutableBasicBlock(int(256 * shrink_ratio), int(512 * shrink_ratio), stride=2),
            MutableBasicBlock(int(512 * shrink_ratio), int(512 * shrink_ratio))
        )

        # input: Tensor[batch_size, 512, input_size[0] // 32, input_size[1] // 32]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * shrink_ratio) * self.input_size[0] // 32 * self.input_size[1] // 32, self.num_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        hidden = self.maxpool(x)

        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)

        hidden = self.avgpool(hidden)
        hidden = torch.flatten(hidden, 1)
        x = self.fc(hidden)

        return x
    
    def grow_tobe(self, shrink_ratio):
        if shrink_ratio < self.shrink_ratio:
            raise ValueError('The shrink ratio must be greater than the current shrink ratio.')
        
        self.shrink_ratio = shrink_ratio

        self.conv1 = mutate_conv2d_kaiming(self.conv1, self.input_channel, int(64 * shrink_ratio))
        self.bn1 = mutate_batchnorm2d_identity(self.bn1, int(64 * shrink_ratio))

        self.layer1[0].grow_tobe(int(64 * shrink_ratio), int(64 * shrink_ratio))
        self.layer1[1].grow_tobe(int(64 * shrink_ratio), int(64 * shrink_ratio))

        self.layer2[0].grow_tobe(int(64 * shrink_ratio), int(128 * shrink_ratio))
        self.layer2[1].grow_tobe(int(128 * shrink_ratio), int(128 * shrink_ratio))

        self.layer3[0].grow_tobe(int(128 * shrink_ratio), int(256 * shrink_ratio))
        self.layer3[1].grow_tobe(int(256 * shrink_ratio), int(256 * shrink_ratio))

        self.layer4[0].grow_tobe(int(256 * shrink_ratio), int(512 * shrink_ratio))
        self.layer4[1].grow_tobe(int(512 * shrink_ratio), int(512 * shrink_ratio))

        self.fc = mutate_linear_kaiming(
            self.fc, 
            int(512 * self.shrink_ratio) * self.input_size[0] // 32 * self.input_size[1] // 32, 
            self.num_output
        )


