#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = '2020'

# import sys
# sys.path.append('path to root of  the project if smth doesn't work')

import os

from mixed_utils.classification_dataloader import MixedFeaturesDataset
from mixed_utils import update_arg_pars as mixed_arg_update
from utils.util_functions import load_model, load_optimizer
from utils.arg_pars import opt
import mlp.model
import mlp.train
import mlp.test



def catch_inner():
    train_dataset = MixedFeaturesDataset(mode='val')
    train_dataset.cache()
    if opt.test:
        val_dataset = MixedFeaturesDataset(mode='val')
        val_dataset.n_classes = train_dataset.n_classes
        val_dataset.cache()

        test_dataset = MixedFeaturesDataset(mode='test')
        test_dataset.n_classes = train_dataset.n_classes
        test_dataset.cache()
    if opt.rels or opt.rels_multitask:
        train_dataset.init_relships()
        if opt.test:
            val_dataset.init_relships()
            test_dataset.init_relships()

    n_classes = train_dataset.n_classes
    n_rels = len(train_dataset.rels_list) - 1

    model, loss, optimizer = mlp.model.create_model(n_classes, n_rels=n_rels)
    if opt.resume or opt.resume_train:
        model.load_state_dict(load_model(name=opt.model_name))
        if opt.resume_train:
            optimizer.load_state_dict(load_optimizer())

    if not opt.resume or opt.resume_train:
        if opt.test:
            mlp.train.training(train_dataset,
                               model=model,
                               loss=loss,
                               optimizer=optimizer,
                               name=opt.model_name,
                               test_dataset=val_dataset)
        else:
            mlp.train.training(train_dataset,
                               model=model,
                               loss=loss,
                               optimizer=optimizer,
                               name=opt.model_name)

    out = []
    if opt.resume:

        if opt.test:
            print('testing on validation set')
            if opt.rels_multi_clip:
                mlp.test.testing(val_dataset, model=model, loss=loss, total_iter=0, mode='val')
                print('testing on test set')
                mlp.test.testing(test_dataset, model=model, loss=loss, total_iter=0, mode='test')
            else:
                if opt.tr_maximize:
                    mlp.test.testing(val_dataset, model=model, loss=loss, total_iter=0, mode='val')
                else:
                    mlp.test.testing(val_dataset, model=model, loss=loss, total_iter=0, mode='val')
                print('testing on test set')
                mlp.test.testing(test_dataset, model=model, loss=loss, total_iter=0, mode='test')
    return out

def pipeline(name=''):
    mixed_arg_update.update(name)
    out = catch_inner()
    return out

def resume_ints_rels():
    opt.resume = True
    opt.test = True

    opt.tracks = True

    opt.rels_multitask = True
    opt.rels_multi_clip = True
    opt.rels_n_clips = 18


    opt.ints = 1
    opt.gates = 1
    opt.ctx = 1

    opt.lymbda = 1

    resume_str = opt.data_root + '/models_release/int_rel.pth.tar'

    if opt.sanity_check:
        inter_class = 'm'
    else:
        inter_class = 'all'


    opt.inter_class = inter_class
    opt.resume_str = resume_str
    pipeline()


if __name__ == '__main__':
    ##############################
    # TRUE = check on small subset if everything works
    # FALSE = test on the entire dataset
    opt.sanity_check = False

    resume_ints_rels()

