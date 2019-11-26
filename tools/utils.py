import logging
import os
import shutil

import numpy as np
import torch
import torch.nn as nn


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_net_config(path):
    with open(path, 'r') as f:
        net_config = ''
        while True:
            line = f.readline().strip()
            if 'net_type' in line:
                net_type = line.split(': ')[-1]
                break
            else:
                net_config += line
    return net_config, net_type


def load_model(model, model_path):
    logging.info('Start loading the model from ' + model_path)
    model.load_state_dict(torch.load(model_path))
    logging.info('Loading the model finished!')


def create_exp_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.):
    """
    Label smoothing implementation.
    This function is taken from https://github.com/MIT-HAN-LAB/ProxylessNAS/blob/master/proxyless_nas/utils.py
    """

    logsoftmax = nn.LogSoftmax().cuda()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def parse_net_config(net_config):
    str_configs = net_config.split('|')
    return [eval(str_config) for str_config in str_configs]
