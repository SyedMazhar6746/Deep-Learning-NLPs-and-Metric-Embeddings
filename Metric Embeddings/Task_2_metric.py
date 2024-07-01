#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class _BNReluConv(nn.Sequential): # container for stacking layers in a sequence
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True): # num_maps_in is the number of input channels, num_maps_out is the number of output channels
        super(_BNReluConv, self).__init__()
        # self.append(nn.GroupNorm(1, num_maps_in))  # Using GroupNorm with a group size of 4
        # self.append(nn.ReLU())
        # self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))

        self.add_module('norm', nn.GroupNorm(1, num_maps_in))  # nn.GroupNorm(4, num_maps_in) # Using GroupNorm with a group size of 1
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32): # input_channels is 1 for MNIST
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.conv3 = _BNReluConv(emb_size, emb_size)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def get_features(self, img): # img size torch.Size([64, 1, 28, 28])
        # pdb.set_trace()
        x = self.conv1(img)                     # torch.Size([64, 32, 26, 26])
        x = self.max_pool(x)                    # torch.Size([64, 32, 12, 12])
        # x = F.max_pool2d(x, 3, stride=2)  

        x = self.conv2(x)                       # torch.Size([64, 32, 10, 10])
        x = self.max_pool(x)                    # torch.Size([64, 32, 4, 4])
        # x = F.max_pool2d(x, 3, stride=2)
        x = self.conv3(x)                       # torch.Size([64, 32, 2, 2])
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.avgpool(x)                   # torch.Size([64, 32, 1, 1])
        x = x.view(x.size(0), -1)  # Flatten the spatial dimensions # torch.Size([64, 32])
        return x

    def loss(self, anchor, positive, negative): # margin is a hyoerparameter
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        # Triplet Margin Loss
        margin = 1.0
        distance_positive = F.pairwise_distance(a_x, p_x)
        distance_negative = F.pairwise_distance(a_x, n_x)
        loss = F.relu(distance_positive - distance_negative + margin)
        return loss.mean()
