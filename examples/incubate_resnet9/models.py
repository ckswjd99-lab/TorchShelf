import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out


class ResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super(ResNet9, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        out = self.block1(x).detach()
        out = self.block2(out).detach()
        out = self.block3(out).detach()
        out = self.block4(out).detach()
        out = self.block5(out).detach()

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # 이 위에 groups=in_channels 가 차이!
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock_Meta(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock_Meta, self).__init__()
        self.conv_res1 = DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out
    
class ResNet9_Meta(nn.Module):
    def __init__(self):
        super(ResNet9_Meta, self).__init__()

        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = ResidualBlock_Meta(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            # DepthwiseSeparableConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=256, momentum=0.9),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5 = nn.Sequential(
            ResidualBlock_Meta(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out


class ResNet9_Incubating(nn.Module):
    def __init__(self, device='cuda'):
        super(ResNet9_Incubating, self).__init__()

        self.block1_candi = [
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=64, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=128, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device),
            nn.Sequential(
                DepthwiseSeparableConv(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=128, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device)
        ]

        self.block2_candi = [
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ).to(device),
            ResidualBlock_Meta(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ).to(device)
        ]

        self.block3_candi = [
                nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device), 
            nn.Sequential(
                DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device)
        ]

        self.block4_candi = [
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device), 
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device)
        ]

        self.block5_candi = [
            nn.Sequential(
                ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ).to(device), 
            nn.Sequential(
                ResidualBlock_Meta(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device)
        ]

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

        # initially meta model
        self.block1 = self.block1_candi[1]
        self.block2 = self.block2_candi[1]
        self.block3 = self.block3_candi[1]
        self.block4 = self.block4_candi[1]
        self.block5 = self.block5_candi[1]

    def set_mode(self, mode):
        if mode == 'meta':
            self.block1 = self.block1_candi[1]
            self.block2 = self.block2_candi[1]
            self.block3 = self.block3_candi[1]
            self.block4 = self.block4_candi[1]
            self.block5 = self.block5_candi[1]
        elif mode == 'block1':
            self.block1 = self.block1_candi[0]
            self.block2 = self.block2_candi[1]
            self.block3 = self.block3_candi[1]
            self.block4 = self.block4_candi[1]
            self.block5 = self.block5_candi[1]
        elif mode == 'block2':
            self.block1 = self.block1_candi[1]
            self.block2 = self.block2_candi[0]
            self.block3 = self.block3_candi[1]
            self.block4 = self.block4_candi[1]
            self.block5 = self.block5_candi[1]
        elif mode == 'block3':
            self.block1 = self.block1_candi[1]
            self.block2 = self.block2_candi[1]
            self.block3 = self.block3_candi[0]
            self.block4 = self.block4_candi[1]
            self.block5 = self.block5_candi[1]
        elif mode == 'block4':
            self.block1 = self.block1_candi[1]
            self.block2 = self.block2_candi[1]
            self.block3 = self.block3_candi[1]
            self.block4 = self.block4_candi[0]
            self.block5 = self.block5_candi[1]
        elif mode == 'block5':
            self.block1 = self.block1_candi[1]
            self.block2 = self.block2_candi[1]
            self.block3 = self.block3_candi[1]
            self.block4 = self.block4_candi[1]
            self.block5 = self.block5_candi[0]
        elif mode == 'full_inflate':
            self.block1 = self.block1_candi[0]
            self.block2 = self.block2_candi[0]
            self.block3 = self.block3_candi[0]
            self.block4 = self.block4_candi[0]
            self.block5 = self.block5_candi[0]

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out