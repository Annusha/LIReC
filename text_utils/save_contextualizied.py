#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import shutil
import os
import os.path as ops
import numpy as np
import re

from utils.arg_pars import opt
from text_utils.classification_dataloader import f_dataloader
from utils.util_functions import dir_check

def save_from_data2_to_data1():

    if 'data2' not in opt.text_path:
        raise FileNotFoundError('Check location of the features')

    p_in = '/sequoia/data1/akukleva/projects/inter_recog/moviegraphs/bert_base'

    # f_dataloader(mode='train')
    f_dataloader(mode='val')
    # f_dataloader(mode='test')

    feature_dir_path = ops.join(p_in, opt.contextualization)
    dir_check(feature_dir_path)

    for root, dirs, files in os.walk(ops.join(opt.text_path, opt.contextualization)):
        for dirname in dirs:
            dir_check(ops.join(feature_dir_path, dirname))
        for filename in files:
            if filename.endswith('npy'):
                dirname = ops.join(feature_dir_path, root.split('/')[-1])
                dir_check(dirname)
                shutil.copy(ops.join(root, filename), ops.join(dirname, filename))

'''
p_out = '/sequoia/data2/akukleva/moviegraph/features/bert/bert_base' 
p_in = '/sequoia/data1/akukleva/projects/inter_recog/moviegraphs/bert_base'
import os
import os.path as ops
import shutil
for root, dirs, files in os.walk(p_out): 
    for filename in files: 
        if filename.endswith('token2idx'): 
            shutil.copy(ops.join(root, filename), ops.join(p_in, root.split('/')[-1], filename))
'''

def check_token2idx():
    p_base = '/sequoia/data2/akukleva/moviegraph/features/bert/bert_base'
    p_file = '/sequoia/data1/akukleva/projects/inter_recog/mixed_up_files'
    idx = 0
    for root, dirs, files in os.walk(p_base):
        for filename in files:
            if filename.endswith('token2idx'):
                with open(ops.join(root, filename), 'r') as f:
                    token_numbers = len(f.read().strip().split('\n'))
                filename = re.sub('token2idx', 'npy', filename)
                n_rows = np.load(ops.join(root, filename)).shape[0]
                try:
                    assert token_numbers == n_rows
                except AssertionError:
                    with open(p_file, 'a') as f:
                        f.write('%s\n' % filename)
                    print(filename)
                idx += 1
                if idx % 500 == 0:
                    print(idx)


if __name__ == '__main__':
    check_token2idx()