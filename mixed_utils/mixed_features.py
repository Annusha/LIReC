#!/usr/bin/env python

"""Functions to load only demanded part of the feature
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import os.path as ops
import re
import numpy as np

from utils.arg_pars import opt
from text_utils.text_features import TextFeatures
from visual_utils.visual_features import VisualFeatures
from utils.util_functions import dir_check, join_data


class MixedFeatures:
    def __init__(self, video_idx, scene_idx, fname):
        self.video_idx = video_idx
        self.scene_idx = scene_idx
        self.fname = fname
        if opt.feature_type in ['m', 'v']:
            self.visual = VisualFeatures(video_idx, scene_idx, fname)
        else: self.visual = None
        if opt.feature_type in ['m', 't']:
            self.textual = TextFeatures(video_idx, scene_idx, fname)
        else: self.textual = None

        self.f_text = np.max
        self.f_visual = np.max

        self.cached = {}
        self.cached_tracks = {}

    def get_features_by_time(self, time_node=None, idx=None):
        '''Given time range return corresponding features
        '''
        if idx in self.cached:
            return self.cached[idx]
        path = ops.join(opt.visual_path, 'cached', 'time', '%s' % opt.feature_type,  self.video_idx)
        str_time_node = '_'.join(str(time_node).split())
        fname = '%s_time_%s.npy' % (self.fname, str_time_node)

        try: features = np.load(ops.join(path, fname))
        except FileNotFoundError: pass
        else:
            if idx is not None: self.cached[idx] = features
            return features

        features = None
        if opt.feature_type in ['m', 'v']:
            features = self.f_visual(self.visual.get_features_by_time(time_node), axis=0, keepdims=True)
            if not opt.spat_pool:
                shape = features.shape
                assert shape[0] == 1
                features = np.mean(features.reshape((shape[0], shape[1], -1)), axis=2)

        if opt.feature_type in ['m', 't']:
            textual = self.f_text(self.textual.get_features_by_time(time_node), axis=0).reshape(1, -1)
            features = join_data(textual, features, np.hstack)
        dir_check(path)
        np.save(ops.join(path, fname), features)

        if idx is not None: self.cached[idx] = features
        return features

    def get_features_by_time_two_stream(self, time_node=None, idx=None):
        if idx in self.cached:
            return self.cached[idx]
        textual = self.f_text(self.textual.get_features_by_time(time_node), axis=0, keepdims=True)
        visual = self.f_visual(self.visual.get_features_by_time(time_node), axis=0, keepdims=True)

        features = {'text': textual,
                    'vid': visual}
        if idx is not None: self.cached[idx] = features
        return features

    def get_duration(self):
        n_frames = len(self.visual.frame2time) - 1
        return self.visual.frame2time[n_frames]

    def get_features_by_track(self, track=None, idx=None, name=''):
        if idx in self.cached_tracks:
            return self.cached_tracks[idx]
        if name:
            path = ops.join(opt.visual_path, 'cached', 'tracks', self.video_idx)
            try:
                first, last, len_tr = track[0]['frame'], track[-1]['frame'], len(track)
            except IndexError:
                track_vis = np.zeros((1, opt.visual_dim))
                if idx is not None: self.cached_tracks[idx] = track_vis
                return track_vis
            name = '_'.join(name.split('/'))
            name = '_'.join(name.split())
            fname = '%s_track.%s.%d-%d(%d).npy' % (self.fname, name, first, last, len_tr)
            try: track_vis = np.load(ops.join(path, fname))
            except FileNotFoundError: pass
            else:
                if idx is not None: self.cached_tracks[idx] = track_vis
                return track_vis

        track_vis = self.visual.get_features_by_track(track)
        track_vis = self.f_visual(track_vis, axis=0, keepdims=True)

        if name:
            dir_check(path)
            np.save(ops.join(path, fname), track_vis)

        if idx is not None: self.cached_tracks[idx] = track_vis
        return track_vis


    def create_ch1_ch2_rel_mat(self, clips):
        if len(clips) == 0: return None
        mat = np.zeros((len(clips), opt.visual_dim + opt.text_dim + opt.track_dim * 2))
        for idx in range(len(clips)):
            time_idx, track1idx, track2idx = clips[idx]
            clipf = self.get_features_by_time(idx=time_idx)
            track1 = self.get_features_by_track(idx=track1idx)
            track2 = self.get_features_by_track(idx=track2idx)
            mat[idx] = np.hstack((clipf, track1, track2))

        return mat

    def free(self):
        if self.visual is not None:
            self.visual.features = None
        if self.textual is not None:
            self.textual.features = None





