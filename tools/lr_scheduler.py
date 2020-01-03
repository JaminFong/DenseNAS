import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
import math

class CosineRestartAnnealingLR(object):
# decay as step
# T_max refers to the max update step

    def __init__(self, optimizer, T_max, lr_period, lr_step, eta_min=0, last_step=-1,
                use_warmup=False, warmup_mode='linear', warmup_steps=0, warmup_startlr=0, 
                warmup_targetlr=0, use_restart=False):

        self.use_warmup = use_warmup
        self.warmup_mode = warmup_mode
        self.warmup_steps = warmup_steps
        self.warmup_startlr = warmup_startlr
        self.warmup_targetlr = warmup_targetlr
        self.use_restart = use_restart
        self.T_max = T_max
        self.eta_min = eta_min

        if self.use_restart == False:
            self.lr_period = [self.T_max - self.warmup_steps]
            self.lr_step = [self.warmup_steps]
        else:
            self.lr_period = lr_period
            self.lr_step = lr_step

        self.last_step = last_step
        self.cycle_length = self.lr_period[0]
        self.cur = 0

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_step == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))


    def step(self, step=None):

        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1
            
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


    def get_lr(self):

        lrs = []
        for base_lr in self.base_lrs:
            if self.use_warmup and self.last_step < self.warmup_steps:
                if self.warmup_mode == 'constant':
                    lrs.append(self.warmup_startlr)
                elif self.warmup_mode =='linear':
                    cur_lr = self.warmup_startlr + \
                        float(self.warmup_targetlr-self.warmup_startlr)/self.warmup_steps*self.last_step
                    lrs.append(cur_lr)
                else:
                    raise NotImplementedError

            else:
                if (self.last_step) in self.lr_step:
                    self.cycle_length = self.lr_period[self.lr_step.index(self.last_step)]
                    self.cur = self.last_step

                peri_iter = self.last_step-self.cur

                if peri_iter <= self.cycle_length:
                    unit_cycle = (1 + math.cos(peri_iter * math.pi / self.cycle_length)) / 2
                    adjusted_cycle = unit_cycle * (base_lr - self.eta_min) + self.eta_min
                    lrs.append(adjusted_cycle)
                else:
                    lrs.append(self.eta_min)

        return lrs


    def display_lr_curve(self, total_steps):
        lrs = []
        for _ in range(total_steps):
            self.step()
            lrs.append(self.get_lr()[0])
        import matplotlib.pyplot as plt
        plt.plot(lrs)
        plt.show()
        

def get_lr_scheduler(config, optimizer, num_examples=None):

    if num_examples is None:
        num_examples = config.data.num_examples
    epoch_steps = num_examples // config.data.batch_size + 1

    if config.optim.use_multi_stage:
        max_steps = epoch_steps * config.optim.multi_stage.stage_epochs
    else:
        max_steps = epoch_steps * config.train_params.epochs
    
    period_steps = [epoch_steps * x for x in config.optim.cosine.restart.lr_period]
    step_steps = [epoch_steps * x for x in config.optim.cosine.restart.lr_step]

    init_lr = config.optim.init_lr

    use_warmup = config.optim.use_warm_up
    if use_warmup:
        warmup_steps = config.optim.warm_up.epoch * epoch_steps
        warmup_startlr = config.optim.warm_up.init_lr
        warmup_targetlr = config.optim.warm_up.target_lr
    else:
        warmup_steps = 0
        warmup_startlr = init_lr
        warmup_targetlr = init_lr

    if config.optim.lr_schedule == 'cosine':
        scheduler = CosineRestartAnnealingLR(optimizer,
                                        float(max_steps),
                                        period_steps,
                                        step_steps,
                                        eta_min=config.optim.min_lr,
                                        use_warmup=use_warmup,
                                        warmup_steps=warmup_steps,
                                        warmup_startlr=warmup_startlr,
                                        warmup_targetlr=warmup_targetlr,
                                        use_restart=config.optim.cosine.use_restart)
        # scheduler = CosineAnnealingLR(optimizer, config.train_params.epochs, config.optim.min_lr)
    elif config.optim.lr_schedule == 'poly':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return scheduler

