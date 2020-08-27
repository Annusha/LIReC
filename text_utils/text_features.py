#!/usr/bin/env python

"""Functions to load only demanded part of the feature
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import os.path as ops
import re
import numpy as np
from collections import defaultdict

from utils.arg_pars import opt
from utils.util_functions import dir_check
from text_utils.feature_extraction import preprocess_file, preprocess_text


class Time:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def includes(self, start, end):
        if self.start <= start <= self.end:
            return True
        if self.start <= end <= self.end:
            return True
        if start <= self.start and end >= self.end:
            return True
        return False

    def include_point(self, point):
        if self.start <= point <= self.end:
            return True
        return False


class TextFeatures:
    def __init__(self, video_idx, scene_idx, fname, n=4):
        self.video_idx = video_idx
        self.scene_idx = '%03d' % scene_idx
        self.fname = re.search(r'(.*).mp4', fname).group(1)
        self.features = None
        self._n = n
        self.tokens = []
        self.dialogs = []
        self.times = []
        self.time_idx2token_range = []
        self.names = defaultdict(list)

        # init some variables
        self._read_times()
        self._tokens_range()
        # self._read_features()

    def _read_times(self):
        def time_str_to_sec(time_str):
            pattern = r'(\d*):(\d*):(\d*).\d*'
            match = re.match(pattern, time_str)
            return int(match.group(1)) * 3600 + \
                   int(match.group(2)) * 60 + \
                   int(match.group(3))

        with open(ops.join(opt.dialogs_path, self.video_idx, self.fname + '.%s' % opt.ext_dialog), 'rb') as f:
            text = f.read().decode('unicode_escape')
            flag = False
            for subtext in preprocess_file(text):
                self.dialogs.append(preprocess_text(subtext))
            for line in text.split('\n'):
                if line.strip().endswith('...'):
                    flag = True
                elif '-->' in line:
                    line = line.strip().split()
                    start = time_str_to_sec(line[0])
                    end = time_str_to_sec(line[-1])
                    if flag:  # join with previous narration
                        self.times[-1].end = end
                        flag = False
                    else:
                        self.times.append(Time(start, end))
                elif flag and line.strip():
                    flag = False
        try:
            assert len(self.dialogs) in [len(self.times), 1]
        except AssertionError:
            print(self.video_idx, self.scene_idx, self.fname)
            print(len(self.dialogs), len(self.times))
            print(self.dialogs)

    def _tokens_range(self):
        with open(ops.join(opt.text_path, self.video_idx,
                           '%s_%s.token2idx' % (self.video_idx, self.scene_idx)),
                  'r') as f:
            start = 0
            for line_idx, line in enumerate(f):
                self.tokens.append(line.split()[0])
                if '[CLS]' in line:
                    if line_idx:
                        token_range = list(range(start, line_idx))
                        self.time_idx2token_range.append(token_range)
                    start = line_idx
            token_range = list(range(start, line_idx + 1))
            self.time_idx2token_range.append(token_range)

    def _read_features(self):
        try:
            feature_path_context = ops.join(opt.text_path, opt.contextualization,
                                    self.video_idx, '%s_%s.npy' % (self.video_idx,
                                                                   self.scene_idx))
            self.features = np.load(feature_path_context)
        except FileNotFoundError:
            print(feature_path_context)
            feature_path = ops.join(opt.text_path, self.video_idx,
                                    '%s_%s.npy' % (self.video_idx, self.scene_idx))
            try:
                self.features = np.load(feature_path)
            except ValueError as err:
                print('%s\n%s' % (err, feature_path))
                raise ValueError(err)
            self.features = self.features.reshape((-1, opt.text_layers, opt.text_dim))

            if opt.contextualization == 'second-to-last':
                self.second_to_last()
            if opt.contextualization == 'last':
                self.last()
            if opt.contextualization == 'sum-all':
                self.sum_all()
            if opt.contextualization == 'sum-last-4':
                self.sum_last_n()
            if opt.contextualization == 'cat-last-4':
                self.cat_last_n()

            feature_dir_path = ops.join(opt.text_path, opt.contextualization)
            dir_check(feature_dir_path)
            feature_dir_path = ops.join(feature_dir_path, self.video_idx)
            dir_check(feature_dir_path)
            np.save(feature_path_context, self.features)

    def get_features_by_time(self, time_node=None, html=False):
        '''Given time range return corresponding features
        '''
        if self.features is None:
            self._read_features()
        print(self.video_idx, 'video is here already', self.features.shape)
        if time_node is None:
            if html:
                return_dialog = '</br>'.join(list([self.dialogs[idx][0] for idx in range(len(self.times)) if self.dialogs[idx]]))
                return self.features, return_dialog
            else:
                return self.features
        tokens_range = []
        dialog_idxs = []
        for time_idx, time in enumerate(self.times):
            try:
                if time.includes(time_node['start'], time_node['end']):
                    tokens_range += self.time_idx2token_range[time_idx]
                    dialog_idxs.append(time_idx)
            except KeyError:
                if html:
                    return self.features, '</br>'.join(self.dialogs)
                return self.features
            except IndexError:
                raise IndexError
        if tokens_range:
            try:
                return self.features[tokens_range]
            except IndexError:
                print('IndexError %s' % str(tokens_range))
                raise IndexError
        else:
            if opt.contextualization.endswith('4'):
                features = np.zeros((1, opt.text_dim * 4))
            else:
                features = np.zeros((1, opt.text_dim))
            if html:
                return features, ''
            return features

    def second_to_last(self):
        self.features = self.features[:, -2, :]

    def last(self):
        self.features = self.features[:, -1, :]

    def sum_all(self):
        self.features = np.sum(self.features, axis=1)

    def sum_last_n(self):
        self.features = np.sum(self.features[:, -self._n:, :], axis=1)

    def cat_last_n(self):
        self.features = self.features[:, -self._n:, :].reshape(-1, opt.text_dim * self._n)

    def get_features_by_track(self, track):
        tokens_range = []
        for elem in track:
            for time_idx, time in enumerate(self.times):
                if time.include_point(elem['timestamp']):
                    tokens_range += self.time_idx2token_range[time_idx]
            if tokens_range:
                return self.features[tokens_range]
            else:
                if opt.contextualization.endswith('4'):
                    features = np.zeros((1, opt.text_dim * 4))
                else:
                    features = np.zeros((1, opt.text_dim))
                return features
