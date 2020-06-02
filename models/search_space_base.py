import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.multadds_count import comp_multadds_fw
from tools.utils import latency_measure_fw
from . import operations
from .operations import OPS


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, primitives):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C_in, C_out, stride, affine=False, track_running_stats=True)
            self._ops.append(op)


class HeadLayer(nn.Module):
    def __init__(self, in_chs, ch, strides, config):
        super(HeadLayer, self).__init__()
        self.head_branches = nn.ModuleList()
        for in_ch, stride in zip(in_chs, strides):
            self.head_branches.append(
                        MixedOp(in_ch, ch, stride, 
                                config.search_params.PRIMITIVES_head)
            )


class StackLayers(nn.Module):
    def __init__(self, ch, num_block_layers, config, primitives):
        super(StackLayers, self).__init__()

        if num_block_layers != 0:
            self.stack_layers = nn.ModuleList()
            for i in range(num_block_layers):
                self.stack_layers.append(MixedOp(ch, ch, 1, primitives))
        else:
            self.stack_layers = None


class Block(nn.Module):
    def __init__(self, in_chs, block_ch, strides, num_block_layers, config):
        super(Block, self).__init__()
        assert len(in_chs) == len(strides)
        self.head_layer = HeadLayer(in_chs, block_ch, strides, config)
        self.stack_layers = StackLayers(block_ch, num_block_layers, config, config.search_params.PRIMITIVES_stack)


class Conv1_1_Branch(nn.Module):
    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Branch, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=block_ch, 
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_ch, affine=False, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv1_1(x)


class Conv1_1_Block(nn.Module):
    def __init__(self, in_chs, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1_branches = nn.ModuleList()
        for in_ch in in_chs:
            self.conv1_1_branches.append(Conv1_1_Branch(in_ch, block_ch))
    
    def forward(self, inputs, betas, block_sub_obj):
        branch_weights = F.softmax(torch.stack(betas), dim=-1)
        return sum(branch_weight * branch(input_data) for input_data, branch, branch_weight in zip(
                    inputs, self.conv1_1_branches, branch_weights)), \
                [block_sub_obj, 0]


class Network(nn.Module):
    def __init__(self, init_ch, dataset, config):
        super(Network, self).__init__()
        self.config = config
        self._C_input = init_ch
        self._head_dim = self.config.optim.head_dim
        self._dataset = dataset
        # use 100-class sub dataset for search
        self._num_classes = 100 

        self.initialize()


    def initialize(self):
        self._init_block_config()
        self._create_output_list()
        self._create_input_list()
        self._init_betas()
        self._init_alphas()
        self._init_sample_branch()


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
                if m.affine==True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine==True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    
    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return


    def _init_betas(self):
        r"""
        beta weights for the output ch choices in the head layer of the block
        """
        self.beta_weights = nn.ParameterList()
        for block in self.output_configs:
            num_betas = len(block['out_chs'])
            self.beta_weights.append(
                nn.Parameter(1e-3 * torch.randn(num_betas))
            )


    def _init_alphas(self):
        r"""
        alpha weights for the op type in the block
        """
        self.alpha_head_weights = nn.ParameterList()
        self.alpha_stack_weights = nn.ParameterList()

        for block in self.input_configs[:-1]:
            num_head_alpha = len(block['in_block_idx'])
            self.alpha_head_weights.append(nn.Parameter(
                1e-3*torch.randn(num_head_alpha, len(self.config.search_params.PRIMITIVES_head)))
            )

            num_layers = block['num_stack_layers']
            self.alpha_stack_weights.append(nn.Parameter(
                1e-3*torch.randn(num_layers, len(self.config.search_params.PRIMITIVES_stack)))
                )

    @property
    def arch_parameters(self):
        arch_params = nn.ParameterList()
        arch_params.extend(self.beta_weights)
        arch_params.extend(self.alpha_head_weights)
        arch_params.extend(self.alpha_stack_weights)
        return arch_params
    
    @property
    def arch_beta_params(self):
        return self.beta_weights
    
    @property
    def arch_alpha_params(self):
        alpha_params = nn.ParameterList()
        alpha_params.extend(self.alpha_head_weights)
        alpha_params.extend(self.alpha_stack_weights)
        return alpha_params


    def display_arch_params(self, display=True):
        branch_weights = []
        head_op_weights = []
        stack_op_weights = []
        for betas in self.beta_weights:
            branch_weights.append(F.softmax(betas, dim=-1))
        for head_alpha in self.alpha_head_weights:
            head_op_weights.append(F.softmax(head_alpha, dim=-1))
        for stack_alpha in self.alpha_stack_weights:
            stack_op_weights.append(F.softmax(stack_alpha, dim=-1))

        if display:
            logging.info('branch_weights \n' + '\n'.join(map(str, branch_weights)))
            if len(self.config.search_params.PRIMITIVES_head) > 1:
                logging.info('head_op_weights \n' + '\n'.join(map(str, head_op_weights)))
            logging.info('stack_op_weights \n' + '\n'.join(map(str, stack_op_weights)))

        return [x.tolist() for x in branch_weights], \
                [x.tolist() for x in head_op_weights], \
                [x.tolist() for x in stack_op_weights]        


    def _init_sample_branch(self):
        _, _ = self.sample_branch('head', 1, training=False)
        _, _ = self.sample_branch('stack', 1, training=False)


    def sample_branch(self, params_type, sample_num, training=True, search_stage=1, if_sort=True):
        r"""
        the sampling computing is based on torch
        input: params_type
        output: sampled params
        """

        def sample(param, weight, sample_num, sample_policy='prob', if_sort=True):
            if sample_num >= weight.shape[-1]:
                sample_policy = 'all'
            assert param.shape == weight.shape
            assert sample_policy in ['prob', 'uniform', 'all']
            if param.shape[0] == 0:
                return [], []
            if sample_policy == 'prob':
                sampled_index = torch.multinomial(weight, num_samples=sample_num, replacement=False)
            elif sample_policy == 'uniform':
                weight = torch.ones_like(weight)
                sampled_index = torch.multinomial(weight, num_samples=sample_num, replacement=False)
            else:
                sampled_index = torch.arange(start=0, end=weight.shape[-1], step=1, device=weight.device
                                            ).repeat(param.shape[0], 1)
            if if_sort:
                sampled_index, _ = torch.sort(sampled_index, descending=False)
            sampled_param_old = torch.gather(param, dim=-1, index=sampled_index)
            return sampled_param_old, sampled_index

        if params_type=='head':
            params = self.alpha_head_weights
        elif params_type=='stack':
            params = self.alpha_stack_weights
        else:
            raise TypeError

        weights = []
        sampled_params_old = []
        sampled_indices = []
        if training:
            sample_policy = self.config.search_params.sample_policy if search_stage==1 else 'uniform'
        else:
            sample_policy = 'all'

        for param in params:
            weights.append(F.softmax(param, dim=-1))
        
        for param, weight in zip(params, weights): #list dim
            sampled_param_old, sampled_index = sample(
                                                    param, weight, sample_num, sample_policy, if_sort)
            sampled_params_old.append(sampled_param_old)
            sampled_indices.append(sampled_index)
        
        if params_type=='head':
            self.alpha_head_index = sampled_indices
        elif params_type=='stack':
            self.alpha_stack_index = sampled_indices
        return sampled_params_old, sampled_indices


    def _init_block_config(self):
        self.block_chs = self.config.search_params.net_scale.chs
        self.block_fm_sizes = self.config.search_params.net_scale.fm_sizes
        self.num_blocks = len(self.block_chs) - 1  # not include the head and tail
        self.num_block_layers = self.config.search_params.net_scale.num_layers
        if hasattr(self.config.search_params.net_scale, 'stage'):
            self.block_stage = self.config.search_params.net_scale.stage

        self.block_chs.append(self.config.optim.last_dim)
        self.block_fm_sizes.append(self.block_fm_sizes[-1])
        self.num_block_layers.append(0)
    

    def _create_output_list(self):
        r"""
        Generate the output config of each block, which contains: 
        'ch': the channel number of the block 
        'out_chs': the possible output channel numbers 
        'strides': the corresponding stride
        """

        self.output_configs = []
        for i in range(len(self.block_chs)-1):
            if hasattr(self, 'block_stage'):
                stage = self.block_stage[i]
            output_config = {'ch': self.block_chs[i], 
                            'fm_size': self.block_fm_sizes[i], 
                            'out_chs': [],
                            'out_fms': [],
                            'strides': [],
                            'out_id': [],
                            'num_stack_layers': self.num_block_layers[i]}
            for j in range(self.config.search_params.adjoin_connect_nums[stage]):
                out_index = i + j + 1
                if out_index >= len(self.block_chs):
                    break
                if hasattr(self, 'block_stage'):
                    block_stage = getattr(self, 'block_stage')
                    if block_stage[out_index]-block_stage[i] > 1:
                        break
                fm_size_ratio = self.block_fm_sizes[i] / self.block_fm_sizes[out_index]
                if fm_size_ratio == 2:
                    output_config['strides'].append(2)
                elif fm_size_ratio == 1:
                    output_config['strides'] .append(1)
                else:
                    break  # only connet to the block whose fm size expansion ratio is 1 or 2
                output_config['out_chs'].append(self.block_chs[out_index])
                output_config['out_fms'].append(self.block_fm_sizes[out_index])
                output_config['out_id'].append(out_index)
            
            self.output_configs.append(output_config)

        logging.info('Network output configs: \n' + '\n'.join(map(str, self.output_configs)))
    

    def _create_input_list(self):
        r"""
        Generate the input config of each block for constructing the whole network.
        Each config dict contains:
        'ch': the channel number of the block
        'in_chs': all the possible input channel numbers
        'strides': the corresponding stride
        'in_block_idx': the index of the input block 
        'beta_idx': the corresponding beta weight index.
        """

        self.input_configs = []
        for i in range(1, len(self.block_chs)):
            input_config = {'ch': self.block_chs[i], 
                            'fm_size': self.block_fm_sizes[i], 
                            'in_chs': [],
                            'in_fms': [],
                            'strides': [],
                            'in_block_idx': [],
                            'beta_idx': [],
                            'num_stack_layers': self.num_block_layers[i]}
            for j in range(i):
                in_index = i - j - 1
                if in_index < 0:
                    break
                output_config = self.output_configs[in_index]
                if i in output_config['out_id']:
                    beta_idx = output_config['out_id'].index(i)
                    input_config['in_block_idx'].append(in_index)
                    input_config['in_chs'].append(output_config['ch'])
                    input_config['in_fms'].append(output_config['fm_size'])
                    input_config['beta_idx'].append(beta_idx)
                    input_config['strides'].append(output_config['strides'][beta_idx])
                else:
                    continue

            self.input_configs.append(input_config)

        logging.info('Network input configs: \n' + '\n'.join(map(str, self.input_configs)))


    def get_cost_list(self, data_shape, cost_type='flops', 
                            use_gpu=True, meas_times=1000):
        cost_list = []
        block_datas = []
        total_cost = 0
        if cost_type == 'flops':
            cost_func = lambda module, data: comp_multadds_fw(
                                        module, data, use_gpu)
        elif cost_type == 'latency':
            cost_func = lambda module, data: latency_measure_fw(
                                        module, data, meas_times)
        else:
            raise NotImplementedError

        if len(data_shape) == 3:
            input_data = torch.randn((1,) + tuple(data_shape))
        else:
            input_data = torch.randn(tuple(data_shape))
        if use_gpu:
            input_data = input_data.cuda()

        cost, block_data = cost_func(self.input_block, input_data)
        cost_list.append(cost)
        block_datas.append(block_data)
        total_cost += cost
        if hasattr(self, 'head_block'):
            cost, block_data = cost_func(self.head_block, block_data)
            cost_list[0] += cost
            block_datas[0] = block_data

        block_flops = []
        for block_id, block in enumerate(self.blocks):
            input_config = self.input_configs[block_id]
            inputs = [block_datas[i] for i in input_config['in_block_idx']]

            head_branch_flops = []
            for branch_id, head_branch in enumerate(block.head_layer.head_branches):
                op_flops = []
                for op in head_branch._ops:
                    cost, block_data = cost_func(op, inputs[branch_id])
                    op_flops.append(cost)
                    total_cost += cost

                head_branch_flops.append(op_flops)
            
            stack_layer_flops = []
            if block.stack_layers.stack_layers is not None:
                for stack_layer in block.stack_layers.stack_layers:
                    op_flops = []
                    for op in stack_layer._ops:
                        cost, block_data = cost_func(op, block_data)
                        if isinstance(op, operations.Skip) and \
                                self.config.optim.sub_obj.skip_reg:
                            # skip_reg is used for regularization as the cost of skip is too small
                            cost = op_flops[0] / 10.
                        op_flops.append(cost)
                        total_cost += cost
                    stack_layer_flops.append(op_flops)
            block_flops.append([head_branch_flops, stack_layer_flops])
            block_datas.append(block_data)
            
        cost_list.append(block_flops)
        
        conv1_1_flops = []
        input_config = self.input_configs[-1]
        inputs = [block_datas[i] for i in input_config['in_block_idx']]
        for branch_id, branch in enumerate(self.conv1_1_block.conv1_1_branches):
            cost, block_data = cost_func(branch, inputs[branch_id])
            conv1_1_flops.append(cost)
            total_cost += cost
        block_datas.append(block_data)

        cost_list.append(conv1_1_flops)
        out = block_datas[-1]
        out = self.global_pooling(out)

        cost, out = cost_func(self.classifier, out.view(out.size(0), -1))
        cost_list.append(cost)
        total_cost += cost

        return cost_list, total_cost
