#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import sys

from utils.arg_pars import opt
from utils.util_functions import dir_check


def update(model_name):
    opt_d = vars(opt)
    if torch.cuda.is_available():
        opt_d['device'] = 'cuda'
    else:
        opt_d['device'] = 'cpu'

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    # visual features
    root = '/sequoia/data1/akukleva/datasets/moviegraph/'
    opt_d['visual_path'] = opt.data_root + '/features/spat_i3d'
    opt_d['visual_dim'] = 2048
    opt_d['sampling_fr'] = 0.0625

    # text features
    opt_d['bert_model'] = 'bert_base_uncased'
    opt_d['text_path'] = opt.data_root + '/features/bert/bert_base'
    opt_d['text_dim'] = 768
    opt_d['text_layers'] = 12

    if opt.feature_type == 'v': opt_d['text_dim'] = 0
    if opt.feature_type == 't': opt_d['visual_dim'] = 0
    opt_d['mlp_dim'] = opt_d['visual_dim'] + opt_d['text_dim']
    if opt.tracks:
        opt_d['track_dim'] = opt.visual_dim
        opt.mlp_dim = opt.mlp_dim + opt.track_dim * 2

    opt_d['model_name'] = model_name
    dir_check(opt.visual_path)

    opt.dialogs_path = opt.data_root + opt.dialogs_path
    opt.frame2time_path = opt.data_root + opt.frame2time_path
    opt.labeled_interactions = opt.data_root + opt.labeled_interactions
    opt.merged_interactions = opt.data_root + opt.merged_interactions
    opt.annotations = opt.data_root + opt.annotations
    opt.split_path = opt.data_root + opt.split_path
    opt.intersected = opt.data_root + opt.intersected
    opt.relships2_15 = opt.data_root + opt.relships2_15
    opt.relships_opp = opt.data_root + opt.relships_opp
    opt.merged_videos = opt.data_root + opt.merged_videos
    opt.ftack_ids = opt.data_root + opt.ftack_ids
    opt.ftracks = opt.data_root + opt.ftracks
    opt.orig_res = opt.data_root + opt.orig_res

    sys.path.append(opt.project_root + '/moviegraphs/py3loader/')

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        print('%s: %s' % (arg, getattr(opt, arg)))
