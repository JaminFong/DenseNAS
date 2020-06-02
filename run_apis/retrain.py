import argparse
import ast
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter

from configs.imagenet_train_cfg import cfg as config
from dataset import imagenet_data
from models import model_derived
from tools import utils
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds

from .trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train_Params")
    parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
    parser.add_argument('--data_path', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='../', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--meas_lat', type=ast.literal_eval, default='False', help='whether to measure the latency of the model')
    parser.add_argument('--job_name', type=str, default='', help='job_name')
    args = parser.parse_args()

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        utils.create_exp_dir(args.save)
        os.system('cp -r ./* '+args.save)
    else:
        args.save = os.path.join(args.save, 'output')
        utils.create_exp_dir(args.save)

    if args.tb_path == '':
        args.tb_path = args.save
    writer = SummaryWriter(args.tb_path)

    utils.set_logging(args.save)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True
    
    if config.train_params.use_seed:
        utils.set_seed(config.train_params.seed)

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))

    if os.path.isfile(os.path.join(args.load_path, 'net_config')):
        config.net_config, config.net_type = utils.load_net_config(
                                os.path.join(args.load_path, 'net_config'))
    derivedNetwork = getattr(model_derived, '%s_Net' % config.net_type.upper())
    model = derivedNetwork(config.net_config, config=config)

    model.eval()
    if hasattr(model, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.net_config)))
    if args.meas_lat:
        latency_cpu = utils.latency_measure(model, (3, 224, 224), 1, 2000, mode='cpu')
        logging.info('latency_cpu (batch 1): %.2fms' % latency_cpu)
        latency_gpu = utils.latency_measure(model, (3, 224, 224), 32, 5000, mode='gpu')
        logging.info('latency_gpu (batch 32): %.2fms' % latency_gpu)
    params = utils.count_parameters_in_MB(model)
    logging.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=config.data.input_size)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)
    
    model = nn.DataParallel(model)

    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(model, config.optim.resume.load_path)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0

    model = model.cuda()

    if config.optim.label_smooth:
        criterion = utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.optim.init_lr,
        momentum=config.optim.momentum,
        weight_decay=config.optim.weight_decay
    )

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
                            testFolder=os.path.join(args.data_path, 'val'),
                            num_workers=config.data.num_workers,
                            type_of_data_augmentation=config.data.type_of_data_aug,
                            data_config=config.data)
    
    if config.optim.use_multi_stage:
        (train_queue, week_train_queue), valid_queue = imagenet.getSetTrainTestLoader(config.data.batch_size)
    else:
        train_queue, valid_queue = imagenet.getTrainTestLoader(config.data.batch_size)
    
    scheduler = get_lr_scheduler(config, optimizer, train_queue.dataset.__len__())
    scheduler.last_step = start_epoch * (train_queue.dataset.__len__() // config.data.batch_size + 1)-1

    trainer = Trainer(train_queue, valid_queue, optimizer, criterion, scheduler, config, args.report_freq)

    best_epoch = [0, 0, 0] # [epoch, acc_top1, acc_top5]
    for epoch in range(start_epoch, config.train_params.epochs):
        
        if config.optim.use_multi_stage and epoch>=config.optim.multi_stage.stage_epochs:
            train_data = week_train_queue
        else:
            train_data = train_queue

        train_acc_top1, train_acc_top5, train_obj, batch_time, data_time = trainer.train(model, epoch)

        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, epoch)

        if val_acc_top1 > best_epoch[1]:
            best_epoch = [epoch, val_acc_top1, val_acc_top5]
            utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])

        writer.add_scalar('train_acc_top1', train_acc_top1, epoch)
        writer.add_scalar('train_loss', train_obj, epoch)
        writer.add_scalar('val_acc_top1', val_acc_top1, epoch)

    if hasattr(model.module, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.module.net_config)))
    logging.info("Params = %.2fMB" % params)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)
