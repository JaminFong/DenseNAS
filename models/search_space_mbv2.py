import torch.nn as nn

from .operations import OPS
from .search_space_base import Conv1_1_Block, Block
from .search_space_base import Network as BaseSearchSpace

class Network(BaseSearchSpace):
    def __init__(self, init_ch, dataset, config):
        super(Network, self).__init__(init_ch, dataset, config)

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self._C_input, kernel_size=3, 
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._C_input, affine=False, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )

        self.head_block = OPS['mbconv_k3_t1'](self._C_input, self._head_dim, 1, affine=False, track_running_stats=True)

        self.blocks = nn.ModuleList()
        
        for i in range(self.num_blocks):
            input_config = self.input_configs[i]
            self.blocks.append(Block(
                input_config['in_chs'],
                input_config['ch'],
                input_config['strides'],
                input_config['num_stack_layers'],
                self.config
            ))

        self.conv1_1_block = Conv1_1_Block(self.input_configs[-1]['in_chs'], 
                                            self.config.optim.last_dim)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.config.optim.last_dim, self._num_classes)

        self.init_model(model_init=config.optim.init_mode)
        self.set_bn_param(self.config.optim.bn_momentum, self.config.optim.bn_eps)
