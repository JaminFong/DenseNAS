import argparse
import importlib
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from configs.imagenet_train_cfg import cfg
from configs.search_config import search_cfg
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--save', type=str, default='./', help='experiment name')
    parser.add_argument('--input_size', type=str, default='[32, 3, 224, 224]', help='data input size')
    parser.add_argument('--meas_times', type=int, default=5000, help='measure times')
    parser.add_argument('--list_name', type=str, default='', help='output list name')
    parser.add_argument('--device', choices=['gpu', 'cpu'])
    parser.add_argument('-c', '--config', metavar='C', default=None, help='The Configuration file')

    args = parser.parse_args()

    update_cfg_from_cfg(search_cfg, cfg)
    if args.config is not None:
        merge_cfg_from_file(args.config, cfg)
    config = cfg

    args.save = os.path.join(args.save, 'output')
    utils.create_exp_dir(args.save)

    args.input_size = eval(args.input_size)
    if len(args.input_size) != 4:
        raise ValueError('The batch size should be specified.')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True

    SearchSpace = importlib.import_module('models.search_space_'+config.net_type).Network
    super_model = SearchSpace(config.optim.init_dim, config.data.dataset, config)

    super_model.eval()
    logging.info("Params = %fMB" % utils.count_parameters_in_MB(super_model))

    if args.device == 'gpu':
        super_model = super_model.cuda()

    latency_list, total_latency = super_model.get_cost_list(
                        args.input_size, cost_type='latency',
                        use_gpu = (args.device == 'gpu'),
                        meas_times = args.meas_times)

    logging.info('latency_list:\n' + str(latency_list))
    logging.info('total latency: ' + str(total_latency) + 'ms')

    with open(os.path.join(args.save, args.list_name), 'w') as f:
        f.write(str(latency_list))
