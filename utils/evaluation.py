#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'


import numpy as np
import torch
from collections import defaultdict
from scipy.special import expit

from utils.arg_pars import opt
from utils.util_functions import load_interaction_names, load_iou2_any

class Precision(object):
    def __init__(self, soft=False, inter2mgd=None, n_rels=0):
        self._top1 = 0
        self._trks_top1 = 0
        self._cls_top1 = 0
        self._rels_top1 = 0
        self._top3 = 0
        self._top5 = 0
        self._top10 = 0
        self.total = 0
        self.total_trcl = 0
        self.total_cl = 0
        self.total_rels = 0

        self._pr_probs_rels = {}
        self._pr_probs_weak_rels = {}

        self._gt_rels = {}
        self._tp = 0
        self._fp_tp = 0
        self._fn_tp = 0
        self._times_tp = []
        self._times_f = []
        self._soft = soft
        self.idx2set = {}
        self.inter2mgd = inter2mgd
        if opt.soft_gt: self.inter2set()
        self._top1_sf, self._top5_sf = 0, 0
        self._top1mgd = 0
        self.movie_paar = defaultdict(lambda: np.zeros(n_rels))
        self._rels_top1 = 0
        self._rels_global = 0

        self.preds = []

    def inter2set(self):
        interactions, inter2idx = load_interaction_names()
        map_inter_class2idx = {'t': 0, 'v': 1, 'm': 2}
        intersects = load_iou2_any()
        for inter, inter_list in intersects.items():
            if inter not in interactions[opt.inter_class]: continue
            inter_idxs = []
            for i in inter_list:
                if i not in interactions[opt.inter_class]: continue
                idx = inter2idx[i][0 if opt.inter_class == 'all' else 2]
                inter_idxs.append(idx)
            idx = inter2idx[inter][0 if opt.inter_class == 'all' else 2]
            inter_idxs.append(idx)
            self.idx2set[idx] = inter_idxs

    def update_probs(self, pr_probs=None, gt=None, top_n_labels=0, pr_classes=None,
                     **kwargs):
        self.total += len(gt)
        if pr_classes is None:
            assert len(pr_probs) == len(gt)
            pr_probs = self._to_numpy(pr_probs)
            gt = self._to_numpy(gt)
            pr_classes = np.argsort(-pr_probs, axis=1)
        else:
            assert len(pr_classes) == len(gt)

        # top 1 precision
        self._top1 += np.sum((pr_classes[:, 0] == gt))
        # top 3 precision
        self._top3 += np.sum([1 for i, j in zip(pr_classes[:, :3], gt) if j in i])
        # top 5 precision
        self._top5 += np.sum([1 for i, j in zip(pr_classes[:, :5], gt) if j in i])

        # if prediction is one of the labels with which it can intersect
        if opt.soft_gt:
            soft_labels = kwargs['soft_labels']
            for idx, pr_5 in enumerate(pr_classes[:, :5]):
                for pr_idx, pr_i in enumerate(pr_5):
                    if pr_i in soft_labels[idx]:
                        if pr_idx == 0: self._top1_sf += 1
                        self._top5_sf += 1
                        break
        try:
            m = kwargs['conf_mat']
            for gt_label, pr_label in zip(gt, pr_classes[:, 0]):
                m[gt_label, pr_label] += 1
            return m
        except KeyError:pass

        if top_n_labels:
            pr_classes = pr_classes[:, :top_n_labels]
            probs = []
            for idx, pr_top_n in enumerate(pr_classes):
                probs.append(pr_probs[idx, list(pr_top_n)])
            return pr_classes[:, :top_n_labels], np.array(probs)

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        return tensor

    def update_probs_max_tracks(self, pr_probs, gt_tracks, gt_classes, print_batch=False, n_names=None,
                                mask=None, just_zeros=None, html=False):

        pr_probs_all = self._to_numpy(pr_probs)
        gt_tracks_all = self._to_numpy(gt_tracks)
        gt_classes_all = self._to_numpy(gt_classes)
        just_zeros = self._to_numpy(just_zeros)
        not_zeros = np.where(~np.array(just_zeros, dtype=bool))
        mask = self._to_numpy(mask)

        pr_probs_all[np.logical_not(mask)] = float('-inf')
        pr_probs = pr_probs_all[not_zeros]
        gt_classes = gt_classes_all[not_zeros]
        gt_tracks = gt_tracks_all[not_zeros]

        self.total += pr_probs.shape[0]
        self.total_cl += pr_probs_all.shape[0]

        pr_probs = expit(pr_probs)
        # pr_probs = pr_probs * mask[..., np.newaxis]
        batch_idxs = np.arange(pr_probs.shape[0])
        batch_idxs_all = np.arange(pr_probs_all.shape[0])
        # if correct label is given, maximize wrt tracks
        pr_tracks = np.argmax(pr_probs[batch_idxs, :, gt_classes], axis=1)
        # if print_batch:
        #     print(list(zip(gt_tracks, pr_tracks, n_names)))
        # nothing is given
        batch = pr_probs.shape[0]
        n_tracks = pr_probs.shape[1]
        n_classes = pr_probs.shape[2]
        pr_probs_flat = pr_probs.reshape(batch, -1)
        argmax_idx_flat = np.argmax(pr_probs_flat, axis=1)
        prpr_tracks = argmax_idx_flat // n_classes
        prpr_labels = argmax_idx_flat % n_classes

        # in classification dataloader max number of possible correct tracks are 2
        for i in range(2):
            # import pdb; pdb.set_trace()
            pr_labels_all = np.argmax(pr_probs_all[batch_idxs_all, gt_tracks_all[:, i], :], axis=1)
            if i == 0:
                zero_idxs = np.ones(pr_probs_all.shape[0], dtype=bool)
                nothing_mask = np.ones(pr_probs_all.shape[0], dtype=bool)
                # if correct tracks are given, maximize wrt labels
                fs_label_mask = pr_labels_all != gt_classes_all
                self._cls_top1 += np.sum(pr_labels_all == gt_classes_all)
            else:
                zero_idxs = (gt_tracks_all[:, 1] != 0) * zero_idxs
                nothing_mask = zero_idxs * (~nothing_mask)
                self._cls_top1 += np.sum(pr_labels_all[fs_label_mask] == gt_classes_all[fs_label_mask])
            self._trks_top1 += np.sum(pr_tracks[zero_idxs[not_zeros]] == gt_tracks[zero_idxs[not_zeros], i])
            if True in zero_idxs[not_zeros]:
                zero_idxs[not_zeros] = pr_tracks != gt_tracks[:, i]
            if html and i == 0:
                predicted_cr_tracks = pr_tracks == gt_tracks[:, i]


            # prpr_tracks = np.argmax(pr_probs[batch_idxs, :, pr_labels], axis=1)
            nothing_mask_not_zero = nothing_mask[not_zeros]
            nothing_mask_not_zero[nothing_mask_not_zero] = prpr_labels[nothing_mask_not_zero] == gt_classes[nothing_mask_not_zero]
            nothing_mask_not_zero[nothing_mask_not_zero] = prpr_tracks[nothing_mask_not_zero] == gt_tracks[:, i][nothing_mask_not_zero]
            nothing_mask[not_zeros] = nothing_mask_not_zero
            self._top1 += np.sum(nothing_mask_not_zero)



    def update_probs_max_tracks_rels(self, pr_probs_cl, pr_probs_rels, gt_classes, gt_rels, gt_tracks, just_zeros=None,
                                     mask=None, rels_mask=None, html=False):

        pr_probs_all_cl = self._to_numpy(pr_probs_cl)
        pr_probs_all_rels = self._to_numpy(pr_probs_rels)
        gt_rels_all = self._to_numpy(gt_rels)
        gt_tracks_all = self._to_numpy(gt_tracks)
        gt_classes_all = self._to_numpy(gt_classes)
        just_zeros = self._to_numpy(just_zeros)
        not_zeros = np.where(~np.array(just_zeros, dtype=bool))
        # where tracks mask
        mask = self._to_numpy(mask)

        # mask lines without tracks at all
        pr_probs_all_cl[np.logical_not(mask)] = float('-inf')
        pr_probs_all_rels[np.logical_not(mask)] = float('-inf')

        # take just lines where tracks are not zeros
        pr_probs_cl = pr_probs_all_cl[not_zeros]
        pr_probs_rels = pr_probs_all_rels[not_zeros]
        gt_classes = gt_classes_all[not_zeros]
        gt_rels = gt_rels_all[not_zeros][:,0]
        gt_tracks = gt_tracks_all[not_zeros]

        # take only tracks which have relationships presented
        if rels_mask.shape[0]:
            pr_probs_all_rels = pr_probs_all_rels[rels_mask].squeeze()

        self.total += pr_probs_cl.shape[0]
        self.total_cl += pr_probs_all_cl.shape[0]

        batch_idxs = np.arange(pr_probs_cl.shape[0])
        batch_idxs_all = np.arange(pr_probs_all_cl.shape[0])
        if rels_mask.shape[0]:
            batch_idxs_all_rels = np.arange(rels_mask.shape[0])
            self.total_rels += rels_mask.shape[0]

        # sigmoid
        pr_probs_cl = expit(pr_probs_cl)
        pr_probs_rels = expit(pr_probs_rels)
        sh_pr = pr_probs_rels.shape
        pr_probs_rels = np.concatenate((pr_probs_rels, np.zeros((sh_pr[0], sh_pr[1], 1))), axis=2)
        pr_tracks = np.argmax(pr_probs_cl[batch_idxs, :, gt_classes] +
                              pr_probs_rels[batch_idxs, :, gt_rels], axis=1)

        batch = pr_probs_cl.shape[0]
        n_tracks = pr_probs_cl.shape[1]
        n_classes = pr_probs_cl.shape[2]
        n_rels =  pr_probs_rels.shape[2]

        pr_probs_cl_tiled = np.tile(pr_probs_cl.reshape(batch * n_tracks, n_classes, 1), (1, 1, n_rels))
        pr_probs_rels_tiled = np.tile(pr_probs_rels.reshape(batch * n_tracks, 1, n_rels), (1, n_classes, 1))
        pr_probs_flat = (pr_probs_cl_tiled + pr_probs_rels_tiled).reshape(batch, -1)
        argmax_idx_flat = np.argmax(pr_probs_flat, axis=1)
        prpr_tracks = argmax_idx_flat // (n_classes * n_rels)
        prpr_labels = (argmax_idx_flat % (n_classes * n_rels)) // n_rels
        prpr_rels = (argmax_idx_flat % (n_classes * n_rels)) % n_rels


        # in classification dataloader max number of possible correct tracks are 2
        for i in range(2):
            # import pdb; pdb.set_trace()
            pr_labels_all = np.argmax(pr_probs_all_cl[batch_idxs_all, gt_tracks_all[:, i], :], axis=1)
            if rels_mask.shape[0]:
                pr_rels_all = np.argmax(pr_probs_all_rels[batch_idxs_all_rels, gt_tracks_all[rels_mask, i].reshape(-1), :], axis=1)
            if i == 0:
                zero_idxs = np.ones(pr_probs_all_cl.shape[0], dtype=bool)
                nothing_mask = np.ones(pr_probs_all_cl.shape[0], dtype=bool)
                # if correct tracks are given, maximize wrt labels
                fs_label_mask = pr_labels_all != gt_classes_all
                self._cls_top1 += np.sum(pr_labels_all == gt_classes_all)
                if rels_mask.shape[0]:
                    gt_rels_now = gt_rels_all[rels_mask.reshape(-1), gt_tracks_all[rels_mask, i].reshape(-1)]
                    fs_label_mask_rels = pr_rels_all != gt_rels_now
                    self._rels_top1 += np.sum(pr_rels_all == gt_rels_now)
            else:
                zero_idxs = (gt_tracks_all[:, 1] != 0) * zero_idxs
                nothing_mask = zero_idxs * (~nothing_mask)
                self._cls_top1 += np.sum(pr_labels_all[fs_label_mask] == gt_classes_all[fs_label_mask])
                if rels_mask.shape[0]:
                    gt_rels_now = gt_rels_all[rels_mask.reshape(-1), gt_tracks_all[rels_mask, i].reshape(-1)]
                    self._rels_top1 += np.sum(pr_rels_all[fs_label_mask_rels] == gt_rels_now[fs_label_mask_rels])
            self._trks_top1 += np.sum(pr_tracks[zero_idxs[not_zeros]] == gt_tracks[zero_idxs[not_zeros], i])
            if True in zero_idxs[not_zeros]:
                zero_idxs[not_zeros] = pr_tracks != gt_tracks[:, i]

            # prpr_tracks = np.argmax(pr_probs[batch_idxs, :, pr_labels], axis=1)
            nothing_mask_not_zero = nothing_mask[not_zeros]
            nothing_mask_not_zero[nothing_mask_not_zero] = prpr_labels[nothing_mask_not_zero] == gt_classes[nothing_mask_not_zero]
            nothing_mask_not_zero[nothing_mask_not_zero] = prpr_rels[nothing_mask_not_zero] == gt_rels[nothing_mask_not_zero]
            nothing_mask_not_zero[nothing_mask_not_zero] = prpr_tracks[nothing_mask_not_zero] == gt_tracks[:, i][nothing_mask_not_zero]
            nothing_mask[not_zeros] = nothing_mask_not_zero
            self._top1 += np.sum(nothing_mask_not_zero)

    def movie_rels(self):
        for id_rel, rels_scores in self.movie_paar.items():
            pr_rel = np.argmax(rels_scores)
            if pr_rel == 0:
                self._rels_global += 1

    def multiclasses_update(self, pr, gt, thr=0.3):
        if isinstance(pr, torch.Tensor):
            pr = pr.numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()
        pr = 1 / (1 + np.exp(-pr))
        pr[pr < thr] = 0
        pr[pr >= thr] = 1
        self._tp += np.sum(pr[gt == 1] == gt[gt == 1])
        self._fp_tp += np.sum(pr)
        self._fn_tp += np.sum(gt)

    def precision(self):
        return self._tp / self._fp_tp
    def recall(self):
        return self._tp / self._fn_tp

    def  times_update(self, pr_time, gt_positions):
        self._times_tp += list(pr_time[gt_positions == 1].flatten())
        self._times_f += list(pr_time[gt_positions != 1].flatten())

    def time_mean(self):
        return np.mean(self._times_tp), np.mean(self._times_f)
    def time_var(self):
        return np.var(self._times_tp), np.var(self._times_f)


    def multiclass_max_update(self, pr, gt):
        if isinstance(pr, torch.Tensor):
            pr = pr.numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()
        for i in range(pr.shape[0]):
            n_labels = int(np.sum(gt[i]))
            pr_labels = np.argsort(pr[i])[-n_labels:]
            gt_labels = np.where(gt[i])[0]
            pr_labels.sort()
            gt_labels.sort()
            self._top1 += np.sum(pr_labels == gt_labels)
            self.total += len(gt_labels)


    def closest_label(self, output, embedded_labels, ret_dist=False):
        t2v = output[:, 0, :].numpy()
        embedded_labels = embedded_labels.numpy()
        dists = -2 * np.dot(t2v, embedded_labels.T) + np.sum(embedded_labels ** 2, axis=1) + np.sum(t2v ** 2, axis=1)[:, np.newaxis]
        closest = np.argsort(dists, axis=1)
        if ret_dist: return closest, dists
        return closest

    def top1(self):
        return self._top1 / self.total

    def top3(self):
        return self._top3 / self.total

    def top5(self):
        return self._top5 / self.total

    def top10(self):
        return self._top10 / self.total

    def top1_sf(self):
        return self._top1_sf / self.total

    def top5_sf(self):
        return self._top5_sf / self.total

    def trks_top1(self):
        return self._trks_top1 / self.total

    def cls_top1(self):
        return self._cls_top1 / self.total_cl

    def rels_top1(self):
        return self._rels_top1 / self.total_rels

    def top1_mgd(self):
        return self._top1mgd / self.total

    def rels_top1(self):
        return self._rels_top1 / self.total

    def rels_global(self):
        return self._rels_global / len(self.movie_paar)



class RelationshipsAcc:
    def __init__(self, n_rels):
        self.total = 0
        self._pr_probs = {}
        self._gt = {}
        self._top1 = 0
        self._top3 = 0
        self.preds = []
        self.conf_mat = np.zeros((n_rels, n_rels))


    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        return tensor

    def update(self, pr_probs, gt, hash):
        assert len(pr_probs) == len(gt)
        assert -1 not in hash
        pr_probs = self._to_numpy(pr_probs)
        pr_probs = expit(pr_probs)
        gt = self._to_numpy(gt)

        for i, h in enumerate(hash):
            h = int(h)
            if h in self._gt:
                # print('more than one rels predicted')
                self._pr_probs[h] += pr_probs[i]
            else:
                self._gt[h] = gt[i]
                self._pr_probs[h] = pr_probs[i]

    def _compute(self):
        self.total = len(self._gt)
        for h in self._gt:
            pr_classes = np.argsort(-self._pr_probs[h], axis=0)
            self.preds.append((self._gt[h],  pr_classes[0]))
            self.conf_mat[self._gt[h], pr_classes[0]] += 1
            if self._gt[h] == pr_classes[0]:
                self._top1 += 1
                self._top3 += 1
            elif self._gt[h] in pr_classes[:3]:
                self._top3 += 1

    def top1(self):
        if self.total == 0:
            self._compute()
        return self._top1 / self.total

    def top3(self):
        return self._top3 / self.total

        # top 1 precision
        self._top1 += np.sum((pr_classes[:, 0] == gt))
        try:
            m = kwargs['conf_mat']
            m[gt, pr_classes[:, 0]] += 1
        except KeyError:pass
        self.preds.append(pr_classes[:, 0])
        # top 3 precision
        self._top3 += np.sum([1 for i, j in zip(pr_classes[:, :3], gt) if j in i])
        # top 5 precision
        self._top5 += np.sum([1 for i, j in zip(pr_classes[:, :5], gt) if j in i])



class TracksSearch:
    def __init__(self):
        self._total = 0
        self._top1 = 0
        self._random = 0
        self._tp, self._tn = 0,  0
        self._fp, self._fn = 0, 0

        self._tr_with_gtinter = 0
        self._tr_top1 = 0


    def track_pair(self, score_mat, gt_inter_idx, gt_track_idxs):
        '''
        given idx for the interaction define which pair of tracks gives
        the highest score
        :param score_mat: n_pairs x n_interactions
        :param gt_inter_idx: ground truth interaction idx
        :param gt_track_idx: ground truth track idx
        '''
        if isinstance(score_mat, torch.Tensor):
            score_mat = score_mat.numpy()
        if len(score_mat.shape) == 1:
            score_mat = score_mat.reshape(1, -1)
        max_score = -np.inf
        pr_inter_idx, pr_track_idx = -1, -1
        for inter_idx in range(score_mat.shape[1]):
            max_track = np.argmax(score_mat[:, inter_idx])
            if inter_idx == gt_inter_idx:
                if max_track in gt_track_idxs:
                    # pretend that we are given ground truth interaction
                    self._tr_with_gtinter += 1
            if score_mat[max_track, inter_idx] > max_score:
                max_score = score_mat[max_track, inter_idx]
                pr_inter_idx, pr_track_idx = inter_idx, max_track

        if pr_track_idx in gt_track_idxs and pr_inter_idx == gt_inter_idx:
            # if maximized group correspond ground truth
            self._tr_top1 += 1
        self._random += 1 / score_mat.shape[0]
        self._total += 1


    def top1(self):
        return self._tr_top1 / self._total

    def top1_gt_inter(self):
        return self._tr_with_gtinter / self._total

    def random(self):
        return self._random / self._total








