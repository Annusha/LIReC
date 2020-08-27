#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2019'


import torch
import os
import os.path as ops
from collections import defaultdict

from utils.util_functions import dir_check

class ModelSaver:
    def __init__(self, n=4, path=''):
        '''
        :param n: how many models to store
        '''
        self.n = n
        self.eval = defaultdict(dict)
        self.models = defaultdict(dict)
        self.worst_idx = defaultdict(lambda: -1)
        self.saved = defaultdict(dict)

        self.path = path
        dir_check(path)

    def check(self, val:dict):
        for key in val:
            if len(self.eval[key]) < self.n: return True
            if val[key] > self.eval[key][self.worst_idx[key]]: return True
        return False

    def update(self, val, model, epoch):
        for key in val:
            self.eval[key][epoch] = val[key]
            self.models[key][epoch] = model
            if len(self.eval[key]) > self.n:
                self.eval[key].pop(self.worst_idx[key])
                self.models[key].pop(self.worst_idx[key])
                self.saved[key].pop(self.worst_idx[key],0)
            self.worst_idx[key] = val[key]
            worst = val[key]
            for epoch_worst, val_worst in self.eval[key].items():
                if val_worst <= worst:
                    worst = val_worst
                    self.worst_idx[key] = epoch_worst
            assert len(self.eval[key]) <= self.n

    def save(self):
        for key in self.eval:
            dir_check(ops.join(self.path, key))
            saved = list(self.saved[key].values())
            for filename in os.listdir(ops.join(self.path, key)):
                if ops.join(self.path, key, filename) not in saved:
                    os.remove(ops.join(self.path, key, filename))
            for epoch, val in self.eval[key].items():
                path = ops.join(self.path, key, 'v%.4f_ep%d.pth.tar' % (val, epoch))
                if path not in saved:
                    self.saved[key][epoch] = path
                    torch.save(self.models[key][epoch], path)






