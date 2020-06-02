import logging
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
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
    if 'http' in model_path:
        model_addr = model_path
        model_path = model_path.split('/')[-1]
        if os.path.isfile(model_path):
            os.system('rm ' + model_path)
        os.system('wget -q ' + model_addr)
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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_logging(save_path, log_name='log.txt'):
    log_format = '%(asctime)s %(message)s'
    date_format = '%m/%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt=date_format)
    fh = logging.FileHandler(os.path.join(save_path, log_name))
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logging.getLogger().addHandler(fh)


def create_save_dir(save_path, job_name):
    if job_name != '':
        job_name = time.strftime("%Y%m%d-%H%M%S-") + job_name
        save_path = os.path.join(save_path, job_name)
        create_exp_dir(save_path)
        os.system('cp -r ./* '+save_path)
        save_path = os.path.join(save_path, 'output')
        create_exp_dir(save_path)
    else:
        save_path = os.path.join(save_path, 'output')
        create_exp_dir(save_path)
    return save_path, job_name


def latency_measure(module, input_size, batch_size, meas_times, mode='gpu'):
    assert mode in ['gpu', 'cpu']
    
    latency = []
    module.eval()
    input_size = (batch_size,) + tuple(input_size)
    input_data = torch.randn(input_size)
    if mode=='gpu':
        input_data = input_data.cuda()
        module.cuda()

    for i in range(meas_times):
        with torch.no_grad():
            start = time.time()
            _ = module(input_data)
            torch.cuda.synchronize()
            if i >= 100:
                latency.append(time.time() - start)
    print(np.mean(latency) * 1e3, 'ms')
    return np.mean(latency) * 1e3


def latency_measure_fw(module, input_data, meas_times):
    latency = []
    module.eval()
    
    for i in range(meas_times):
        with torch.no_grad():
            start = time.time()
            output_data = module(input_data)
            torch.cuda.synchronize()
            if i >= 100:
                latency.append(time.time() - start)
    print(np.mean(latency) * 1e3, 'ms')
    return np.mean(latency) * 1e3, output_data


def record_topk(k, rec_list, data, comp_attr, check_attr):
    def get_insert_idx(orig_list, data, comp_attr):
        start = 0
        end = len(orig_list)
        while start < end:
            mid = (start + end) // 2
            if data[comp_attr] < orig_list[mid][comp_attr]:
                start = mid + 1
            else:
                end = mid
        return start
    
    if_insert = False
    insert_idx = get_insert_idx(rec_list, data, comp_attr)
    if insert_idx < k:
        rec_list.insert(insert_idx, data)
        if_insert = True
    while len(rec_list) > k:
        rec_list.pop()
    return if_insert
