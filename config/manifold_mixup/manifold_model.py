import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
           "PreActResNet18", "PreActResNet34", "PreActResNet50", "PreActResNet101", "PreActResNet152"]
use_cuda = torch.cuda.is_available()


# ====================================================================================== #
# Model helper
# ====================================================================================== #
def to_one_hot(inp, num_classes):
    y_onehot = torch.zeros(inp.size(0), num_classes)
    y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
    y_onehot.requires_grad = False
    return y_onehot.cuda() if use_cuda else y_onehot


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    return out, target_reweighted, target_shuffled_onehot


# ====================================================================================== #
# ResNet for manifold mixup
# ====================================================================================== #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, mixup_hidden, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.mixup_hidden = mixup_hidden
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lam=None, target=None, target_reweighted=None):

        if self.mixup_hidden:
            layer_mix = np.random.randint(0, 3)
        else:
            layer_mix = 0

        out = x
        
        if lam is not None:
            if target_reweighted is None: 
                target_reweighted = to_one_hot(target, self.num_classes)
            else:
                assert target is None

            if layer_mix == 0:
                out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = F.relu(self.bn1(out))

        out = self.layer1(out)

        if lam is not None and self.mixup_hidden and layer_mix == 1:
            out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer2(out)

        if lam is not None and self.mixup_hidden and layer_mix == 2:
            out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if lam is None:
            return out
        else:
            return out, target_reweighted, target_shuffled_onehot


def ResNet18(mixup_hidden, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], mixup_hidden, num_classes)


def ResNet34(mixup_hidden, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], mixup_hidden, num_classes)


def ResNet50(mixup_hidden, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], mixup_hidden, num_classes)


def ResNet101(mixup_hidden, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], mixup_hidden,num_classes)


def ResNet152(mixup_hidden, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], mixup_hidden, num_classes)


# ====================================================================================== #
# PreActResNet for manifold mixup
# ====================================================================================== #
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, mixup_hidden, initial_channels, num_classes):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels

        self.mixup_hidden = mixup_hidden
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, lam=None, target=None, target_reweighted=None, layer_mix='rand'):

        if layer_mix == 'rand':
            if self.mixup_hidden:
                layer_mix = np.random.randint(0, 2)
            else:
                layer_mix = 0

        out = x

        if lam is not None:
            if target_reweighted is None:
                target_reweighted = to_one_hot(target, self.num_classes)
            else:
                assert target is None

            if layer_mix == 0:
                out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = self.layer1(out)

        if lam is not None and layer_mix == 1:
            out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer2(out)

        if lam is not None and layer_mix == 2:
            out, target_reweighted, target_shuffled_onehot = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if lam is None:
            return out
        else:
            return out, target_reweighted, target_shuffled_onehot


def PreActResNet18(mixup_hidden, initial_channels, num_classes):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], mixup_hidden, initial_channels, num_classes)


def PreActResNet34(mixup_hidden, initial_channels, num_classes):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], mixup_hidden, initial_channels,num_classes)


def PreActResNet50(mixup_hidden, initial_channels, num_classes):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], mixup_hidden, initial_channels, num_classes)


def PreActResNet101(mixup_hidden, initial_channels, num_classes):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], mixup_hidden, initial_channels, num_classes)


def PreActResNet152(mixup_hidden, initial_channels, num_classes):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], mixup_hidden, initial_channels, num_classes)
