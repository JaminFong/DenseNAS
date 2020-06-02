import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOp(nn.Module):
    def __init__(self, dropped_mixed_ops, softmax_temp=1.):
        super(MixedOp, self).__init__()
        self.softmax_temp = softmax_temp
        self._ops = nn.ModuleList()
        for op in dropped_mixed_ops:
            self._ops.append(op)

    def forward(self, x, alphas, branch_indices, mixed_sub_obj):
        op_weights = torch.stack([alphas[branch_index] for branch_index in branch_indices])
        op_weights = F.softmax(op_weights / self.softmax_temp, dim=-1)
        return sum(op_weight * op(x) for op_weight, op in zip(op_weights, self._ops)), \
                sum(op_weight * mixed_sub_obj[branch_index] for op_weight, branch_index in zip(
                    op_weights, branch_indices))


class HeadLayer(nn.Module):
    def __init__(self, dropped_mixed_ops, softmax_temp=1.):
        super(HeadLayer, self).__init__()
        self.head_branches = nn.ModuleList()
        for mixed_ops in dropped_mixed_ops:
            self.head_branches.append(MixedOp(mixed_ops, softmax_temp))

    def forward(self, inputs, betas, alphas, head_index, head_sub_obj):
        head_data = []
        count_sub_obj = []
        for input_data, head_branch, alpha, head_idx, branch_sub_obj in zip(
                        inputs, self.head_branches, alphas, head_index, head_sub_obj):
            data, sub_obj = head_branch(input_data, alpha, head_idx, branch_sub_obj) 
            head_data.append(data)
            count_sub_obj.append(sub_obj)

        return sum(branch_weight * data for branch_weight, data in zip(betas, head_data)), \
                count_sub_obj


class StackLayers(nn.Module):
    def __init__(self, num_block_layers, dropped_mixed_ops, softmax_temp=1.):
        super(StackLayers, self).__init__()

        if num_block_layers != 0:
            self.stack_layers = nn.ModuleList()
            for i in range(num_block_layers):
                self.stack_layers.append(MixedOp(dropped_mixed_ops[i], softmax_temp))
        else:
            self.stack_layers = None

    def forward(self, x, alphas, stack_index, stack_sub_obj):
        
        if self.stack_layers is not None:
            count_sub_obj = 0
            for stack_layer, alpha, stack_idx, layer_sub_obj in zip(self.stack_layers, alphas, stack_index, stack_sub_obj):
                x, sub_obj = stack_layer(x, alpha, stack_idx, layer_sub_obj)
                count_sub_obj += sub_obj
            return x, count_sub_obj

        else:
            return x, 0


class Block(nn.Module):
    def __init__(self, num_block_layers, dropped_mixed_ops, softmax_temp=1.):
        super(Block, self).__init__()
        self.head_layer = HeadLayer(dropped_mixed_ops[0], softmax_temp)
        self.stack_layers = StackLayers(num_block_layers, dropped_mixed_ops[1], softmax_temp)
    
    def forward(self, inputs, betas, head_alphas, stack_alphas, head_index, stack_index, block_sub_obj):
        x, head_sub_obj = self.head_layer(inputs, betas, head_alphas, head_index, block_sub_obj[0])
        x, stack_sub_obj = self.stack_layers(x, stack_alphas, stack_index, block_sub_obj[1])

        return x, [head_sub_obj, stack_sub_obj]


class Dropped_Network(nn.Module):
    def __init__(self, super_model, alpha_head_index=None, alpha_stack_index=None, softmax_temp=1.):
        super(Dropped_Network, self).__init__()

        self.softmax_temp = softmax_temp
        # static modules loading
        self.input_block = super_model.module.input_block
        if hasattr(super_model.module, 'head_block'):
            self.head_block = super_model.module.head_block
        self.conv1_1_block = super_model.module.conv1_1_block
        self.global_pooling = super_model.module.global_pooling
        self.classifier = super_model.module.classifier

        # architecture parameters loading
        self.alpha_head_weights = super_model.module.alpha_head_weights
        self.alpha_stack_weights = super_model.module.alpha_stack_weights
        self.beta_weights = super_model.module.beta_weights
        self.alpha_head_index = alpha_head_index if alpha_head_index is not None else \
                                                super_model.module.alpha_head_index
        self.alpha_stack_index = alpha_stack_index if alpha_stack_index is not None else \
                                                super_model.module.alpha_stack_index

        # config loading
        self.config = super_model.module.config
        self.input_configs = super_model.module.input_configs
        self.output_configs = super_model.module.output_configs
        self.sub_obj_list = super_model.module.sub_obj_list

        # dynamic blocks loading
        self.blocks = nn.ModuleList()

        for i, block in enumerate(super_model.module.blocks):
            input_config = self.input_configs[i]

            dropped_mixed_ops = []
            # for the head layers
            head_mixed_ops = []
            for j, head_index in enumerate(self.alpha_head_index[i]):
                head_mixed_ops.append([block.head_layer.head_branches[j]._ops[k] for k in head_index])
            dropped_mixed_ops.append(head_mixed_ops)

            stack_mixed_ops = []
            for j, stack_index in enumerate(self.alpha_stack_index[i]):
                stack_mixed_ops.append([block.stack_layers.stack_layers[j]._ops[k] for k in stack_index])
            dropped_mixed_ops.append(stack_mixed_ops)

            self.blocks.append(Block(
                input_config['num_stack_layers'],
                dropped_mixed_ops
            ))

    def forward(self, x):
        '''
        To approximate the the total sub_obj(latency/flops), we firstly create the obj list for blocks
        as follows:
            [[[head_flops_1, head_flops_2, ...], stack_flops], ...]
        Then we compute the whole obj approximation from the end to the beginning. For block b, 
            flops'_b = sum(beta_{bi} * (head_flops_{bi} + stack_flops_{i}) for i in out_idx[b])
        The total flops equals flops'_0
        '''
        sub_obj_list = []
        block_datas = []
        branch_weights = []
        for betas in self.beta_weights:
            branch_weights.append(F.softmax(betas / self.softmax_temp, dim=-1))

        block_data = self.input_block(x)
        if hasattr(self, 'head_block'):
            block_data = self.head_block(block_data)

        block_datas.append(block_data)
        sub_obj_list.append([[],torch.tensor(self.sub_obj_list[0]).cuda()])

        for i in range(len(self.blocks)+1):
            config = self.input_configs[i]
            inputs = [block_datas[i] for i in config['in_block_idx']]
            betas = [branch_weights[block_id][beta_id] 
                        for block_id, beta_id in zip(config['in_block_idx'], config['beta_idx'])]

            if i == len(self.blocks):
                block_data, block_sub_obj = self.conv1_1_block(inputs, betas, self.sub_obj_list[2])

            else:
                block_data, block_sub_obj = self.blocks[i](inputs,
                                                        betas, 
                                                        self.alpha_head_weights[i], 
                                                        self.alpha_stack_weights[i],
                                                        self.alpha_head_index[i],
                                                        self.alpha_stack_index[i],
                                                        self.sub_obj_list[1][i])
            block_datas.append(block_data)
            sub_obj_list.append(block_sub_obj)

        out = self.global_pooling(block_datas[-1])
        logits = self.classifier(out.view(out.size(0),-1))

        # chained cost estimation
        for i, out_config in enumerate(self.output_configs[::-1]):
            block_id = len(self.output_configs)-i-1
            sum_obj = []
            for j, out_id in enumerate(out_config['out_id']):
                head_id = self.input_configs[out_id-1]['in_block_idx'].index(block_id)
                head_obj = sub_obj_list[out_id][0][head_id]
                stack_obj = sub_obj_list[out_id][1]
                sub_obj_j = branch_weights[block_id][j] * (head_obj + stack_obj)
                sum_obj.append(sub_obj_j)
            sub_obj_list[-i-2][1] += sum(sum_obj)

        net_sub_obj = torch.tensor(self.sub_obj_list[-1]).cuda() + sub_obj_list[0][1]
        return logits, net_sub_obj.expand(1)
    
    @property
    def arch_parameters(self):
        arch_params = nn.ParameterList()
        arch_params.extend(self.beta_weights)
        arch_params.extend(self.alpha_head_weights)
        arch_params.extend(self.alpha_stack_weights)
        return arch_params

    @property
    def arch_alpha_params(self):
        alpha_params = nn.ParameterList()
        alpha_params.extend(self.alpha_head_weights)
        alpha_params.extend(self.alpha_stack_weights)
        return alpha_params
