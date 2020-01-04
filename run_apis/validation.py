import argparse
import logging
import os
import pprint
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from configs.imagenet_val_cfg import cfg
from dataset import imagenet_data
from models import model_derived
from tools import utils
from tools.multadds_count import comp_multadds

from .trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--data_path', type=str, default='../data', help='location of the dataset')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='./', help='the path of output')

    args = parser.parse_args()
    config = cfg

    args.save = os.path.join(args.save, 'output')
    utils.create_exp_dir(args.save)

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
    
    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))

    config.net_config, net_type = utils.load_net_config(os.path.join(args.load_path, 'net_config'))

    derivedNetwork = getattr(model_derived, '%s_Net' % net_type.upper())
    model = derivedNetwork(config.net_config, config=config)
    
    logging.info("Network Structure: \n" + '\n'.join(map(str, model.net_config)))
    logging.info("Params = %.2fMB" % utils.count_parameters_in_MB(model))
    logging.info("Mult-Adds = %.2fMB" % comp_multadds(model, input_size=config.data.input_size))

    model = model.cuda()
    model = nn.DataParallel(model)
    utils.load_model(model, os.path.join(args.load_path, 'weights.pt'))

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
                            testFolder=os.path.join(args.data_path, 'val'),
                            num_workers=config.data.num_workers,
                            data_config=config.data)
    valid_queue = imagenet.getTestLoader(config.data.batch_size)
    trainer = Trainer(None, valid_queue, None, None, 
                        None, config, args.report_freq)

    with torch.no_grad():
        val_acc_top1, val_acc_top5, valid_obj, batch_time = trainer.infer(model)
