import torch
import torch.nn as nn

from models.dropped_model import Dropped_Network


class Optimizer(object):

    def __init__(self, model, criterion, config):
        self.config = config
        self.weight_sample_num = self.config.search_params.weight_sample_num
        self.criterion = criterion
        self.Dropped_Network = lambda model: Dropped_Network(
                        model, softmax_temp=config.search_params.softmax_temp)

        arch_params_id = list(map(id, model.module.arch_parameters))
        weight_params = filter(lambda p: id(p) not in arch_params_id, model.parameters())

        self.weight_optimizer = torch.optim.SGD(
                                weight_params,
                                config.optim.weight.init_lr,
                                momentum=config.optim.weight.momentum,
                                weight_decay=config.optim.weight.weight_decay)

        self.arch_optimizer = torch.optim.Adam(
                            [{'params': model.module.arch_alpha_params, 'lr': config.optim.arch.alpha_lr},
                                {'params': model.module.arch_beta_params, 'lr': config.optim.arch.beta_lr}],
                            betas=(0.5, 0.999),
                            weight_decay=config.optim.arch.weight_decay)


    def arch_step(self, input_valid, target_valid, model, search_stage):
        head_sampled_w_old, alpha_head_index = \
            model.module.sample_branch('head', 2, search_stage= search_stage)
        stack_sampled_w_old, alpha_stack_index = \
            model.module.sample_branch('stack', 2, search_stage= search_stage)
        self.arch_optimizer.zero_grad()

        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_valid)
        if self.config.optim.if_sub_obj:
            loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
            sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
            loss += loss_sub_obj * sub_loss_factor
        loss.backward()
        self.arch_optimizer.step()

        self.rescale_arch_params(head_sampled_w_old,
                                stack_sampled_w_old,
                                alpha_head_index,
                                alpha_stack_index,
                                model)
        return logits.detach(), loss.item(), sub_obj.item()
    

    def weight_step(self, *args, **kwargs):
        return self.weight_step_(*args, **kwargs)


    def weight_step_(self, input_train, target_train, model, search_stage):
        _, _ = model.module.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.module.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        self.weight_optimizer.zero_grad()
        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        logits, sub_obj = dropped_model(input_train)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_train)
        loss.backward()
        self.weight_optimizer.step()

        return logits.detach(), loss.item(), sub_obj.item()


    def valid_step(self, input_valid, target_valid, model):
        _, _ = model.module.sample_branch('head', 1, training=False)
        _, _ = model.module.sample_branch('stack', 1, training=False)

        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_valid)

        return logits, loss.item(), sub_obj.item()


    def rescale_arch_params(self, alpha_head_weights_drop, 
                            alpha_stack_weights_drop,
                            alpha_head_index,
                            alpha_stack_index,
                            model):

        def comp_rescale_value(old_weights, new_weights, index):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
            new_exp_sum = new_drop_arch_params.exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum)
            rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value)
            return rescale_value, rescale_mat
        
        def rescale_params(old_weights, new_weights, indices):
            for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
                for j, (old_weights_branch, indices_branch) in enumerate(
                                                    zip(old_weights_block, indices_block)):
                    rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
                                                                new_weights[i][j],
                                                                indices_branch)
                    new_weights[i][j].data.add_(rescale_mat)

        # rescale the arch params for head layers
        rescale_params(alpha_head_weights_drop, model.module.alpha_head_weights, alpha_head_index)
        # rescale the arch params for stack layers
        rescale_params(alpha_stack_weights_drop, model.module.alpha_stack_weights, alpha_stack_index)


    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'Arch':
            state_list = [True, False] # [arch, weight]
        elif stage == 'Weights':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.weight_optimizer.param_groups, state_list[1])
