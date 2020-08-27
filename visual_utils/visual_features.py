#!/usr/bin/env python

"""Functions to load only demanded part of the feature
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'

import re
import numpy as np
import os.path as ops
from collections import defaultdict

from utils.arg_pars import opt
from utils.util_functions import load_orig_resol


class VisualFeatures:
    def __init__(self, video_idx, scene_idx, fname):
        self.video_idx = video_idx
        self.scene_idx = '%03d' % scene_idx
        self.fname = re.search(r'(.*).mp4', fname).group(1)
        self.features = None
        self.frame2time = {}
        self.time2frame = defaultdict(list)
        self.dims = None

        # init some variables
        self._read_frame2time()
        if opt.tf_crop or not opt.spat_pool:
            self._get_resolution()

    def _read_features(self):
        feature_path_context = ops.join(opt.visual_path, self.video_idx,
                                        '%s.npy' % self.fname)
        print('read feaures', self.video_idx)
        self.features = np.load(feature_path_context)

        # if opt.i3d == 'spat' and not opt.tf_crop and opt.spat_pool:
        #     shape = self.features.shape
        #     self.features = self.features.reshape((shape[0], shape[1], -1))
        #     self.features = np.mean(self.features, axis=2)


    def _read_frame2time(self):
        path = ops.join(opt.frame2time_path, self.video_idx, '%s.%s' % (self.fname,
                                                                  opt.ext_frame2time))
        with open(path, 'r') as f:
            for line in f:
                frame, time = line.strip().split()
                frame = int(frame)
                time = int(time.split('.')[0])
                self.frame2time[frame] = time
                self.time2frame[time].append(frame)

    def _get_resolution(self):
        # height, width
        self.dims = load_orig_resol()[self.video_idx]

    def get_features_by_time(self, time_node=None):
        '''Given time range return corresponding features
        '''
        if self.features is None:
            self._read_features()
        # if opt.i3d == 'spat' and opt.tf_crop and opt.spat_pool:
            # apply spatial average pooling
        shape = self.features.shape
        features = self.features.reshape((shape[0], shape[1], -1))
        features = np.mean(features, axis=2)
        # else:
        #     # features're already pooled
        #     features = self.features

        if time_node is None:
            features = features
            return features
        try:
            start = self.time2frame[int(time_node['start'])][0]
            end_time = int(time_node['end'])
            # due to problems with time rounding
            end_time = end_time if end_time in self.time2frame else end_time-1
            end = self.time2frame[end_time][-1]
            if opt.sampling_fr < 1:
                start = int(start * opt.sampling_fr)
                end = int(end * opt.sampling_fr)
            if end >= features.shape[0]:
                features_range = range(start, features.shape[0],
                                       1 if opt.sampling_fr < 1 else opt.sampling_fr)
                features = features[features_range]
                return features
            features_range = range(start, end + 1,
                                   1 if opt.sampling_fr < 1 else opt.sampling_fr)
            features = features[features_range]
            return features

        except KeyError:
            return self.features
        except IndexError:
            import pdb; pdb.set_trace()
            print(str(self.time2frame))
            print(str(end_time))
            print(str(time_node))
            raise IndexError

    def get_features_by_track(self, track):
        if self.features is None:
            self._read_features()
        if opt.tf_crop:
            # reserve memory for pooled spatialy features
            features = np.zeros((len(track), opt.visual_dim))
            # grid resolution
            hgrid, wgrid= self.features.shape[2], self.features.shape[3]
            sh, sw = hgrid / self.dims[0], wgrid / self.dims[1]

            FH0, FH1 = 0.10, 0.25  # height ratio
            # FW0, FW1 = 0.30, 0.70 # face has this width ratio in person bbox (tight crop, good for clothing)
            FW0, FW1 = 0.35, 0.65  # face has this width ratio in person bbox (loose crop, good for wavy hands)
            for t_elem_idx, t_elem in enumerate(track):
                # blow up face bbox to correspond to person bbox
                fx, fy, fw, fh = t_elem['x'] / 2., t_elem['y'] / 2., t_elem['w'] / 2., t_elem['h'] / 2.
                pw, ph = fw / (FW1 - FW0), fh / (FH1 - FH0)
                px, py = fx - FW0 * pw, fy - FH0 * ph
                # scale to fit the grid
                spx, spw = px * sw, pw * sw
                spy, sph = py * sh, ph * sh
                # get ranges (left including, right excluding)
                rx = [max(0, int(np.floor(spx))), min(int(wgrid), int(np.ceil(spx + spw)))]
                ry = [max(0, int(np.floor(spy))), min(int(hgrid), int(np.ceil(spy + sph)))]
                # pool features in that grid and append
                frame_idx = int(t_elem['frame'] * opt.sampling_fr)
                if frame_idx == self.features.shape[0]: continue
                this_feat = self.features[frame_idx]
                this_feat = this_feat[:, ry[0]: ry[1], rx[0]: rx[1]].reshape(1, opt.visual_dim, -1)
                features[t_elem_idx] = np.mean(this_feat, axis=2)
            return features
        else:
            frame_range = []
            for t_elem in track:
                frame_idx = int(t_elem['frame'] * opt.sampling_fr)
                if frame_idx == self.features.shape[0]: continue
                frame_range.append(frame_idx)
            frame_range = list(np.unique(frame_range))
            return self.features[frame_range]

    def get_frame(self, idx):
        idx = int(idx * opt.sampling_fr)
        return self.features[idx]
