#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import torch
import os.path as ops

from utils.arg_pars import opt
from utils.util_functions import dir_check


def update():
    args_map = {'text_features': '',
                'contextualization': '',
                'lr': 'lr',
                'epochs': 'ep',
                'batch_size': 'bs',
                'model_name': '',
                'log_prefix': '',
                'mlp_dim': 'dim',
                'inter_class': 'ic',
                'feature_type': 'ft_',
                'pool_features': 'pf_'}
    # default='/sequoia/data2/akukleva/moviegraph/features/bert'
    opt_d = vars(opt)
    if torch.cuda.is_available():
        opt_d['device'] = 'cuda'
    else:
        opt_d['device'] = 'cpu'
    if opt.text_features == 'bert_base':
        opt_d['bert_model'] = 'bert-base-uncased'
        if ops.isdir('/sequoia/data2/'):
            opt_d['text_path'] = '/sequoia/data2/akukleva/moviegraph/features/bert/bert_base'
        else:
            opt_d['text_path'] = '/sequoia/data1/akukleva/projects/inter_recog/moviegraphs/bert_base'
        opt_d['text_dim'] = 768
        opt_d['text_layers'] = 12
    if opt.text_features == 'bert_large':
        opt_d['bert_model'] = ''
        opt_d['text_path'] = '/sequoia/data1/akukleva/projects/inter_recog/moviegrophs/bert_large'
        opt_d['text_dim'] = 1024
        opt_d['text_layers'] = 24
    opt_d['mlp_dim'] = opt_d['text_dim']
    opt_d['model_name'] = 'mlp_text'
    dir_check(opt.text_path)


    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        print('%s: %s' % (arg, getattr(opt, arg)))
