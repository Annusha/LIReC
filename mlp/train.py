#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2019'

from datetime import datetime
from os.path import join
import torch
import time

from utils.util_functions import Averaging, dir_check
from utils.model_saver import ModelSaver
from utils.arg_pars import opt
from mlp.test import testing
import copy


def training(train_dataset, **kwargs):
    train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('set parameters and model, train start time: %s' % train_start_time)

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   drop_last=False)

    print('epochs: %s', opt.epochs)
    model_saver_val = ModelSaver(path=opt.store_root)
    for epoch in range(opt.epochs):
        model.to(opt.device)
        model.train()
        train_dataset.epoch = epoch

        print('Epoch # %d' % epoch)
        end = time.time()
        counter = 0
        if opt.tr_sum_max:
            if epoch == 20:
                opt.tr_sum_max_flag = True
        for i, input in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            labels = input['labels']
            if len(labels) == 1:
                continue
            output = model(input)
            loss_values = loss(output, input)
            losses.update(loss_values.item(), len(labels))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            counter += len(labels)

            if i % 10 == 0 and i:
                print('Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_dataloader), batch_time=batch_time, data_time=data_time, loss=losses))
        print(counter)
        print('loss: %f' % losses.avg)
        losses.reset()
        if epoch % opt.test_fr == 0:
            testing(train_dataset, model, loss, total_iter=epoch, mode='train',
                    train_start_time=train_start_time)
            if opt.test:
                check_val = testing(kwargs['val_dataset'], model, loss, total_iter=epoch,
                                    train_start_time=train_start_time, mode='val')
                if model_saver_val.check(check_val):
                    save_dict = {'epoch': epoch,
                                 'state_dict': copy.deepcopy(model.state_dict()),
                                 'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                    model_saver_val.update(check_val, save_dict, epoch)


                    testing(kwargs['test_dataset'], model, loss, total_iter=epoch,
                            train_start_time=train_start_time, mode='test')

            print(opt.log_prefix)

        if opt.save_model and opt.save_model_often and epoch % 30 == 0:
            model_saver_val.save()


    check_str = join(opt.store_root)
    opt.resume_str = join(check_str, '%d.pth.tar' % epoch)
    if opt.save_model:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(check_str)
        torch.save(save_dict, opt.resume_str)
    return model