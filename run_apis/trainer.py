import logging
import time

import torch.nn as nn

from dataset.prefetch_data import data_prefetcher
from tools import utils


class Trainer(object):
    def __init__(self, train_data, val_data, optimizer=None, criterion=None, 
                    scheduler=None, config=None, report_freq=None):
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.report_freq = report_freq
    
    def train(self, model, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.train()
        start = time.time()

        prefetcher = data_prefetcher(self.train_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:        
            data_t = time.time() - start
            self.scheduler.step()
            n = input.size(0)
            if step==0:
                logging.info('epoch %d lr %e', epoch, self.optimizer.param_groups[0]['lr'])
            self.optimizer.zero_grad()
            
            logits= model(input)
            if self.config.optim.label_smooth:
                loss = self.criterion(logits, target, self.config.optim.smooth_alpha)
            else:
                loss = self.criterion(logits, target)

            loss.backward()
            if self.config.optim.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        
            batch_t = time.time() - start
            start = time.time()

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)
            if step!=0 and step % self.report_freq == 0:
                logging.info(
                    'Train epoch %03d step %03d | loss %.4f  top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, objs.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            input, target = prefetcher.next()
            step += 1
        logging.info('EPOCH%d Train_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)

        return top1.avg, top5.avg, objs.avg, batch_time.avg, data_time.avg


    def infer(self, model, epoch=0):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.eval()

        start = time.time()
        prefetcher = data_prefetcher(self.val_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)

            logits = model(input)
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            batch_t = time.time() - start
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step % self.report_freq == 0:
                logging.info(
                    'Val epoch %03d step %03d | top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            start = time.time()
            input, target = prefetcher.next()

        logging.info('EPOCH%d Valid_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)
        return top1.avg, top5.avg, batch_time.avg, data_time.avg


class SearchTrainer(object):
    def __init__(self, train_data, val_data, search_optim, criterion, scheduler, config, args):
        self.train_data = train_data
        self.val_data = val_data
        self.search_optim = search_optim
        self.criterion = criterion
        self.scheduler = scheduler
        self.sub_obj_type = config.optim.sub_obj.type
        self.args = args


    def train(self, model, epoch, optim_obj='Weights', search_stage=0):
        assert optim_obj in ['Weights', 'Arch']
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        sub_obj_avg = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.train()

        start = time.time()
        if optim_obj == 'Weights':
            prefetcher = data_prefetcher(self.train_data)
        elif optim_obj == 'Arch':
            prefetcher = data_prefetcher(self.val_data)
            
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            input, target = input.cuda(), target.cuda()
            data_t = time.time() - start
            n = input.size(0)
            if optim_obj == 'Weights':
                self.scheduler.step()
                if step==0:
                    logging.info('epoch %d weight_lr %e', epoch, self.search_optim.weight_optimizer.param_groups[0]['lr'])
                logits, loss, sub_obj = self.search_optim.weight_step(input, target, model, search_stage)
            elif optim_obj == 'Arch':
                if step==0:
                    logging.info('epoch %d arch_lr %e', epoch, self.search_optim.arch_optimizer.param_groups[0]['lr'])
                logits, loss, sub_obj = self.search_optim.arch_step(input, target, model, search_stage)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            del logits, input, target

            batch_t = time.time() - start
            objs.update(loss, n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            sub_obj_avg.update(sub_obj)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step!=0 and step % self.args.report_freq == 0:
                logging.info(
                    'Train%s epoch %03d step %03d | loss %.4f %s %.2f top1_acc %.2f top5_acc %.2f | batch_time %.3f data_time %.3f', 
                    optim_obj ,epoch, step, objs.avg, self.sub_obj_type, sub_obj_avg.avg, 
                    top1.avg, top5.avg, batch_time.avg, data_time.avg)
            start = time.time()
            step += 1
            input, target = prefetcher.next()
        return top1.avg, top5.avg, objs.avg, sub_obj_avg.avg, batch_time.avg


    def infer(self, model, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        sub_obj_avg = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        
        model.train() # don't use running_mean and running_var during search
        start = time.time()
        prefetcher = data_prefetcher(self.val_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)

            logits, loss, sub_obj = self.search_optim.valid_step(input, target, model)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            batch_t = time.time() - start
            objs.update(loss, n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            sub_obj_avg.update(sub_obj)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step % self.args.report_freq == 0:
                logging.info(
                    'Val epoch %03d step %03d | loss %.4f %s %.2f top1_acc %.2f top5_acc %.2f | batch_time %.3f data_time %.3f', 
                    epoch, step, objs.avg, self.sub_obj_type, sub_obj_avg.avg, top1.avg, top5.avg,
                    batch_time.avg, data_time.avg)
            start = time.time()
            input, target = prefetcher.next()

        return top1.avg, top5.avg, objs.avg, sub_obj_avg.avg, batch_time.avg
