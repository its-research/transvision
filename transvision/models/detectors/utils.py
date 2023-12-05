import torch
from torch import nn as nn
from torch.nn import functional as F


class ReduceInfTC(nn.Module):

    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel // 2, channel // 4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel // 4, channel // 8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel // 8, channel // 64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel // 64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(channel // 64, channel // 8, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(channel // 8, channel // 4, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(
            channel // 4,
            channel // 2,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=1,
        )
        self.bn2_3 = nn.BatchNorm2d(channel // 2, track_running_stats=True)

    def forward(self, x):
        # outputsize = x.shape
        # out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))

        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))

        # x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1


class PixelWeightedFusion(nn.Module):

    def __init__(self, channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1


def QuantFunc(input, b_n=4):
    alpha = torch.abs(input).max()
    s_alpha = alpha / (2**(b_n - 1) - 1)

    input = input.clamp(min=-alpha, max=alpha)
    input = torch.round(input / s_alpha)
    input = input * s_alpha

    return input


def AttentionMask(image_1, image_2, img_shape=(576, 576), mask_shape=(36, 36), threshold=0.0):
    mask = torch.zeros((image_1.shape[0], mask_shape[0], mask_shape[1])).cuda(image_1.device)

    feat_diff = torch.sum(torch.abs(image_1 - image_2), dim=1)
    stride = int(img_shape[0] / mask_shape[0])
    for bs in range(image_1.shape[0]):
        for kk in range(mask_shape[0]):
            for ll in range(mask_shape[1]):
                patch = feat_diff[bs, kk * stride:(kk + 1) * stride, ll * stride:(ll + 1) * stride]
                if patch.sum() > threshold:
                    mask[bs, kk, ll] = 1
    # sparse_ratio = mask.sum() / mask.numel()
    # print("Sparse Ratio: ", sparse_ratio)

    return mask
