import argparse
import logging
import os
import pprint
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from configs.config import cfg
from dataset import imagenet_data
from model_derived import Network
from tools import utils
from tools.multadds_count import comp_multadds

parser = argparse.ArgumentParser("Params")
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
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


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    data_time = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.eval()

    start = time.time()
    for step, (input, target) in enumerate(valid_queue):
        data_t = time.time() - start

        input = input.cuda()
        target = target.cuda()
        n = input.size(0)

        logits = model(input)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        batch_t = time.time() - start

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        data_time.update(data_t)
        batch_time.update(batch_t)

        if step!=0 and step % args.report_freq == 0:
            logging.info(
                'Val step %03d | top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                step, top1.avg, top5.avg, batch_time.avg, data_time.avg)

        start = time.time()

    return top1.avg, top5.avg, objs.avg, batch_time.avg


if __name__ == '__main__':

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True
    
    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))

    config.net_config = utils.load_net_config(os.path.join(args.load_path, 'net_config'))

    model = Network(config.net_config, 'ImageNet', config=config)
    
    logging.info("Network Structure: \n" + '\n'.join(map(str, model.net_config)))
    logging.info("Params = %.2fMB" % utils.count_parameters_in_MB(model))
    logging.info("Mult-Adds = %.2fMB" % comp_multadds(model, input_size=config.data.input_size))

    model = model.cuda()
    model = nn.DataParallel(model)

    utils.load_model(model, os.path.join(args.load_path, 'weights.pt'))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
                            testFolder=os.path.join(args.data_path, 'val'),
                            num_workers=config.train_params.num_workers,
                            data_config=config.data)
    
    valid_queue = imagenet.getTestLoader(config.train_params.batch_size)
    
    with torch.no_grad():
        val_acc_top1, val_acc_top5, valid_obj, batch_time = infer(valid_queue, model, criterion)
    logging.info('Valid_acc  top1 %.2f top5 %.2f batch_time %.3f', val_acc_top1, val_acc_top5, batch_time)
