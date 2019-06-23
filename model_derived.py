import math

import torch.nn as nn
import torch.nn.functional as F

from operations import *


class Block(nn.Module):

    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](in_ch, block_ch, stride, affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch, 1, affine=True, track_running_stats=True))
        self.stack_layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x


class Conv1_1_Block(nn.Module):

    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=block_ch, 
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv1_1(x)


class Network(nn.Module):

    def __init__(self, net_config, dataset, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        aux_config=[True/False, ch, block_idx, aux_weight]
        """
        super(Network, self).__init__()
        self.config = config
        self.net_config = self.parse_net_config(net_config)

        self._C_input = self.net_config[0][0][0]

        self._dataset = dataset
        self._num_classes = 10 if self._dataset=='cifar10' else 1000

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self._C_input, kernel_size=3, 
                    stride=1 if self._dataset=='cifar10' else 2,
                    padding=1, bias=False),
            nn.BatchNorm2d(self._C_input),
            nn.ReLU6(inplace=True)
        )

        self.blocks = nn.ModuleList()
        for cfg in self.net_config:
            self.blocks.append(Block(cfg[0][0], cfg[0][1], 
                                cfg[1], cfg[2], cfg[-1]))

        self.conv1_1_block = Conv1_1_Block(self.net_config[-1][0][1], self.config.optim.last_dim)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.config.optim.last_dim, self._num_classes)
        self.set_bn_param(self.config.optim.bn_momentum, self.config.optim.bn_eps)


    def forward(self,x):
        block_data = self.input_block(x)

        for i, block in enumerate(self.blocks):
            block_data = block(block_data)

        block_data = self.conv1_1_block(block_data)

        out = self.global_pooling(block_data)
        logits = self.classifier(out.view(out.size(0),-1))

        return logits


    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return


    def parse_net_config(self, net_config):
        str_configs = net_config.split('|')
        return [eval(str_config) for str_config in str_configs]
