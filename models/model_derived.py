import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import OPS
from tools.utils import parse_net_config


class Block(nn.Module):

    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](in_ch, block_ch, stride, 
                                        affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch, 1, 
                                        affine=True, track_running_stats=True))
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


class MBV2_Net(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(MBV2_Net, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs = self.net_config[0][0][0]
        self._num_classes = 1000

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_chs, kernel_size=3, 
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_chs),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            if config[1] == 'conv1_1':
                continue
            self.blocks.append(Block(config[0][0], config[0][1], 
                            config[1], config[2], config[-1]))

        if self.net_config[-1][1] == 'conv1_1':
            block_last_dim = self.net_config[-1][0][0]
            last_dim = self.net_config[-1][0][1]
        else:
            block_last_dim = self.net_config[-1][0][1]
        self.conv1_1_block = Conv1_1_Block(block_last_dim, last_dim)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_dim, self._num_classes)

        self.init_model()
        self.set_bn_param(0.1, 0.001)


    def forward(self,x):
        block_data = self.input_block(x)
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)
        block_data = self.conv1_1_block(block_data)

        out = self.global_pooling(block_data)
        logits = self.classifier(out.view(out.size(0),-1))

        return logits

    def init_model(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return


class RES_Net(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(RES_Net, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs = self.net_config[0][0][0]
        self._num_classes = 1000

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_chs, kernel_size=3, 
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_chs),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            self.blocks.append(Block(config[0][0], config[0][1], 
                            config[1], config[2], config[-1]))

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.net_config[-1][1] == 'bottle_neck':
            last_dim = self.net_config[-1][0][-1] * 4
        else:
            last_dim = self.net_config[-1][0][1]
        self.classifier = nn.Linear(last_dim, self._num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine==True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):        
        block_data = self.input_block(x)
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)

        out = self.global_pooling(block_data)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits