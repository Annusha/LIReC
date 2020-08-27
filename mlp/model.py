#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2019'

import torch
import pprint
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.arg_pars import opt



class Modalities(nn.Module):

    def __init__(self, n_classes, n_rels=0):
        super(Modalities, self).__init__()
        self.n_classes = n_classes
        self.n_rels = n_rels

        out_dim_ints = 0
        if opt.modality in ['m', 't']:
            ## TEXT ##
            self.txt_ints = nn.Linear(opt.text_dim, opt.joint_dim)
            self.txt2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            out_dim_ints += opt.joint_dim

        if opt.modality in ['m', 'v']:
            ## VIS ##
            self.vis_ints = nn.Linear(opt.visual_dim, opt.joint_dim)
            self.vis2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            out_dim_ints += opt.joint_dim

        if opt.tracks:
            ## TRACKS ##
            self.tracks1_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks2_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks12_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)
            self.tracks22_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)

            out_dim_ints += opt.joint_dim


        ## LAST ##
        self.out_ints = nn.Linear(out_dim_ints, n_classes)

        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        # text features

        output_ints = None
        if opt.modality in ['m', 't']:
            txt_ints = x['features'][:, 0, :opt.text_dim].float().unsqueeze(1)
            if opt.device == 'cuda': txt_ints = txt_ints.cuda(non_blocking=True)
            txt_ints = self.txt_ints(txt_ints)
            txt_ints = self.txt2_ints(torch.relu(self.dropout(txt_ints)))

        if opt.modality in ['m', 'v']:
            vis_ints = x['features'][:, 0, opt.text_dim:opt.text_dim + opt.visual_dim].float().unsqueeze(1)
            if opt.device == 'cuda': vis_ints = vis_ints.cuda(non_blocking=True)
            vis_ints = self.vis_ints(vis_ints)
            vis_ints = self.vis2_ints(torch.relu(self.dropout(vis_ints)))

        if opt.tracks:
            tracks_ints = x['features'][:, 0, opt.text_dim + opt.visual_dim:].float().unsqueeze(1)
            if opt.device == 'cuda': tracks_ints = tracks_ints.cuda(non_blocking=True)
            tracks1_ints = self.tracks1_ints(tracks_ints[:, :, :opt.track_dim])
            tracks2_ints = self.tracks2_ints(tracks_ints[:, :, opt.track_dim:])
            tracks1_ints = self.tracks12_ints(torch.relu(self.dropout(tracks1_ints)))
            tracks2_ints = self.tracks22_ints(torch.relu(self.dropout(tracks2_ints)))

        if opt.modality == 'm':
            if opt.tracks:
                output_ints = torch.cat((txt_ints, vis_ints, tracks1_ints, tracks2_ints), dim=2).squeeze(1)
            else:
                output_ints = torch.cat((txt_ints, vis_ints), dim=2).squeeze(1)
        if opt.modality == 't':
            output_ints = txt_ints.squeeze(1)
        if opt.modality == 'v':
            output_ints = vis_ints.squeeze(1)

        output_ints = self.dropout(torch.tanh(output_ints))

        output_ints = self.out_ints(output_ints)

        return {'inters': output_ints}


class MidFusionMultiClip(nn.Module):
    def __init__(self, n_classes, n_rels=0):
        super(MidFusionMultiClip, self).__init__()
        self.n_classes = n_classes
        self.n_rels = n_rels


        if opt.ints == 1:
            ## TEXT ##
            self.txt_ints = nn.Linear(opt.text_dim, opt.joint_dim)
            self.txt2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## VIS ##
            self.vis_ints = nn.Linear(opt.visual_dim, opt.joint_dim)
            self.vis2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## TRACKS ##
            self.tracks1_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks2_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks12_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)
            self.tracks22_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)

            out_dim_ints = opt.joint_dim * 3


        if opt.ctx == 1:
            ## TEXT ##
            self.txt_ctx = nn.Linear(opt.text_dim, opt.joint_dim)
            self.txt2_ctx = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## VIS ##
            self.vis_ctx = nn.Linear(opt.visual_dim, opt.joint_dim)
            self.vis2_ctx = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## TRACKS ##
            self.tracks1_ctx = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks2_ctx = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks12_ctx = nn.Linear(opt.joint_dim, opt.joint_dim // 2)
            self.tracks22_ctx = nn.Linear(opt.joint_dim, opt.joint_dim // 2)

            out_dim_ctx = opt.joint_dim * 3


        ## GATING ##
        if opt.gates == 1:
            out_dim_ints = opt.joint_dim * opt.mid_m_ints
            self.gates_ints = GatingUnit(in_dim1=opt.joint_dim * 3, in_dim2=opt.joint_dim * 3, out_dim=out_dim_ints)

        ## LAST ##
        if opt.ints == 1:
            self.out_ints = nn.Linear(out_dim_ints, n_classes)
        if opt.ctx == 1:
            self.out_ctx = nn.Linear(out_dim_ctx, n_rels)

        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        # text features
        batch = x['features'].shape[0]
        output_ints, output_ctx = None, None
        if opt.ints == 1:
            txt_ints = x['features'][:, 0, :opt.text_dim].float().unsqueeze(1)
            if opt.device == 'cuda': txt_ints = txt_ints.cuda(non_blocking=True)
            txt_ints = self.txt_ints(txt_ints)
            txt_ints = self.txt2_ints(torch.relu(self.dropout(txt_ints)))

            vis_ints = x['features'][:, 0, opt.text_dim:opt.text_dim + opt.visual_dim].float().unsqueeze(1)
            if opt.device == 'cuda': vis_ints = vis_ints.cuda(non_blocking=True)
            vis_ints = self.vis_ints(vis_ints)
            vis_ints = self.vis2_ints(torch.relu(self.dropout(vis_ints)))

            tracks_ints = x['features'][:, 0, opt.text_dim + opt.visual_dim:].float().unsqueeze(1)
            if opt.device == 'cuda': tracks_ints = tracks_ints.cuda(non_blocking=True)
            tracks1_ints = self.tracks1_ints(tracks_ints[:, :, :opt.track_dim])
            tracks2_ints = self.tracks2_ints(tracks_ints[:, :, opt.track_dim:])
            tracks1_ints = self.tracks12_ints(torch.relu(self.dropout(tracks1_ints)))
            tracks2_ints = self.tracks22_ints(torch.relu(self.dropout(tracks2_ints)))

            output_ints = torch.cat((txt_ints, vis_ints, tracks1_ints, tracks2_ints), dim=2).squeeze(1)
            output_ints = self.dropout(torch.tanh(output_ints))


        if opt.ctx == 1:
            mod_mask = x['rels_mask'].float().to(opt.device)
            divider = mod_mask.sum(1)
            mod_mask = mod_mask.repeat(1,1,opt.joint_dim)
            txt_ctx = x['features'][:, 1:, :opt.text_dim].float().view(-1, 1, opt.text_dim)
            if opt.device == 'cuda': txt_ctx = txt_ctx.cuda(non_blocking=True)
            txt_ctx = self.txt_ctx(txt_ctx)
            txt_ctx = self.txt2_ctx(torch.relu(self.dropout(txt_ctx)))
            txt_ctx = (txt_ctx.view(batch, -1, opt.joint_dim) * mod_mask).sum(1) / divider

            vis_ctx = x['features'][:, 1:, opt.text_dim:opt.text_dim + opt.visual_dim].float().view(-1, 1, opt.visual_dim)
            if opt.device == 'cuda': vis_ctx = vis_ctx.cuda(non_blocking=True)
            vis_ctx = self.vis_ctx(vis_ctx)
            vis_ctx = self.vis2_ctx(torch.relu(self.dropout(vis_ctx)))
            vis_ctx = (vis_ctx.view(batch, -1, opt.joint_dim) * mod_mask).sum(1) / divider

            tracks_ctx = x['features'][:, 1:, opt.text_dim + opt.visual_dim:].float().view(-1, 1, opt.track_dim * 2)
            if opt.device == 'cuda': tracks_ctx = tracks_ctx.cuda(non_blocking=True)
            tracks1_ctx = self.tracks1_ctx(tracks_ctx[:, :, :opt.track_dim])
            tracks2_ctx = self.tracks2_ctx(tracks_ctx[:, :, opt.track_dim:])
            tracks1_ctx = self.tracks12_ctx(torch.relu(self.dropout(tracks1_ctx)))
            tracks2_ctx = self.tracks22_ctx(torch.relu(self.dropout(tracks2_ctx)))
            tracks1_ctx = (tracks1_ctx.view(batch, -1, opt.joint_dim // 2) * mod_mask[:, :, :opt.joint_dim // 2]).sum(1) / divider
            tracks2_ctx = (tracks2_ctx.view(batch, -1, opt.joint_dim // 2) * mod_mask[:, :, :opt.joint_dim // 2]).sum(1) / divider

            output_ctx = torch.cat((txt_ctx, vis_ctx, tracks1_ctx, tracks2_ctx), dim=1)
            output_ctx = self.dropout(torch.tanh(output_ctx))

        if opt.gates == 1:
            output_ints_gate = torch.tensor(output_ints, requires_grad=True)
            output_ints = self.gates_ints({'inters': output_ints, 'rels': output_ctx})

        if opt.ctx == 1:
            output_ctx = self.out_ctx(output_ctx)

        if opt.ints == 1:
            output_ints = self.out_ints(output_ints)

        return {'inters': output_ints, 'rels': output_ctx}


class MidFusionMultiClipMaxTracks(nn.Module):
    def __init__(self, n_classes, n_rels=0):
        super(MidFusionMultiClipMaxTracks, self).__init__()
        self.n_classes = n_classes
        self.n_rels = n_rels

        if opt.ints == 1:
            ## TEXT ##
            self.txt_ints = nn.Linear(opt.text_dim, opt.joint_dim)
            self.txt2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## VIS ##
            self.vis_ints = nn.Linear(opt.visual_dim, opt.joint_dim)
            self.vis2_ints = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## TRACKS ##
            self.tracks1_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks2_ints = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks12_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)
            self.tracks22_ints = nn.Linear(opt.joint_dim, opt.joint_dim // 2)

            out_dim_ints = opt.joint_dim * 3

        if opt.ctx == 1:
            ## TEXT ##
            self.txt_ctx = nn.Linear(opt.text_dim, opt.joint_dim)
            self.txt2_ctx = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## VIS ##
            self.vis_ctx = nn.Linear(opt.visual_dim, opt.joint_dim)
            self.vis2_ctx = nn.Linear(opt.joint_dim, opt.joint_dim)
            ## TRACKS ##
            self.tracks1_ctx = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks2_ctx = nn.Linear(opt.track_dim, opt.joint_dim)
            self.tracks12_ctx = nn.Linear(opt.joint_dim, opt.joint_dim // 2)
            self.tracks22_ctx = nn.Linear(opt.joint_dim, opt.joint_dim // 2)

            out_dim_ctx = opt.joint_dim * 3

        ## GATING ##
        if opt.gates == 1:
            out_dim_ints = opt.joint_dim * opt.mid_m_ints
            self.gates_ints = GatingUnit(in_dim1=opt.joint_dim * 3, in_dim2=opt.joint_dim * 3, out_dim=out_dim_ints)

        ## LAST ##
        if opt.ints == 1:
            self.out_ints = nn.Linear(out_dim_ints, n_classes)
        if opt.ctx == 1:
            self.out_ctx = nn.Linear(out_dim_ctx, n_rels)

        self.dropout = nn.Dropout(p=opt.dropout)


    # MidFusionMultiClipMaxTracks
    def forward(self, x):
        # text features
        assert opt.tr_maximize
        batch = x['features'].shape[0]
        n_tracks = x['features'].shape[1]
        if opt.ctx == 1:
            n_rels = x['features'].shape[2]-1
            x['features'] = x['features'].view(-1, n_rels+1, opt.mlp_dim)
        else:
            x['features'] = x['features'].view(-1, 1, opt.mlp_dim)
        output_ints, output_ctx = None, None

        # MidFusionMultiClipMaxTracks
        if opt.ints == 1:
            txt_ints = x['features'][:, 0, :opt.text_dim].float().unsqueeze(1)
            if opt.device == 'cuda': txt_ints = txt_ints.cuda(non_blocking=True)
            txt_ints = self.txt_ints(txt_ints)
            txt_ints = self.txt2_ints(torch.relu(self.dropout(txt_ints)))

            vis_ints = x['features'][:, 0, opt.text_dim:opt.text_dim + opt.visual_dim].float().unsqueeze(1)
            if opt.device == 'cuda': vis_ints = vis_ints.cuda(non_blocking=True)
            vis_ints = self.vis_ints(vis_ints)
            vis_ints = self.vis2_ints(torch.relu(self.dropout(vis_ints)))

            tracks_ints = x['features'][:, 0, opt.text_dim + opt.visual_dim:].float().unsqueeze(1)
            if opt.device == 'cuda': tracks_ints = tracks_ints.cuda(non_blocking=True)
            tracks1_ints = self.tracks1_ints(tracks_ints[:, :, :opt.track_dim])
            tracks2_ints = self.tracks2_ints(tracks_ints[:, :, opt.track_dim:])
            tracks1_ints = self.tracks12_ints(torch.relu(self.dropout(tracks1_ints)))
            tracks2_ints = self.tracks22_ints(torch.relu(self.dropout(tracks2_ints)))

            output_ints = torch.cat((txt_ints, vis_ints, tracks1_ints, tracks2_ints), dim=2).squeeze(1)
            output_ints = self.dropout(torch.tanh(output_ints))

        # MidFusionMultiClipMaxTracks
        if opt.ctx == 1:
            mod_mask = x['rels_mask'].float().to(opt.device).view(-1, n_rels, 1)
            divider = mod_mask.sum(1)
            divider[divider == 0] = 1
            mod_mask = mod_mask.repeat(1,1,opt.joint_dim)
            txt_ctx = x['features'][:, 1:, :opt.text_dim].float()
            if opt.device == 'cuda': txt_ctx = txt_ctx.cuda(non_blocking=True)
            txt_ctx = self.txt_ctx(txt_ctx)
            txt_ctx = self.txt2_ctx(torch.relu(self.dropout(txt_ctx)))
            txt_ctx = (txt_ctx.view(batch * n_tracks, -1, opt.joint_dim) * mod_mask).sum(1) / divider

            vis_ctx = x['features'][:, 1:, opt.text_dim:opt.text_dim + opt.visual_dim].float()
            if opt.device == 'cuda': vis_ctx = vis_ctx.cuda(non_blocking=True)
            vis_ctx = self.vis_ctx(vis_ctx)
            vis_ctx = self.vis2_ctx(torch.relu(self.dropout(vis_ctx)))
            vis_ctx = (vis_ctx.view(batch * n_tracks, -1, opt.joint_dim) * mod_mask).sum(1) / divider

            tracks_ctx = x['features'][:, 1:, opt.text_dim + opt.visual_dim:].float()
            if opt.device == 'cuda': tracks_ctx = tracks_ctx.cuda(non_blocking=True)
            tracks1_ctx = self.tracks1_ctx(tracks_ctx[:, :, :opt.track_dim])
            tracks2_ctx = self.tracks2_ctx(tracks_ctx[:, :, opt.track_dim:])
            tracks1_ctx = self.tracks12_ctx(torch.relu(self.dropout(tracks1_ctx)))
            tracks2_ctx = self.tracks22_ctx(torch.relu(self.dropout(tracks2_ctx)))
            tracks1_ctx = (tracks1_ctx.view(batch * n_tracks, -1, opt.joint_dim // 2) * mod_mask[:, :, :opt.joint_dim // 2]).sum(1) / divider
            tracks2_ctx = (tracks2_ctx.view(batch * n_tracks, -1, opt.joint_dim // 2) * mod_mask[:, :, :opt.joint_dim // 2]).sum(1) / divider

            output_ctx = torch.cat((txt_ctx, vis_ctx, tracks1_ctx, tracks2_ctx), dim=1)
            output_ctx = self.dropout(torch.tanh(output_ctx))

        if opt.gates == 1:
            output_ints = self.gates_ints({'inters': output_ints, 'rels': output_ctx})

        if opt.ctx == 1:
            output_ctx = self.out_ctx(output_ctx).view(batch, -1, self.n_rels)

        if opt.ints == 1:
            output_ints = self.out_ints(output_ints).view(batch, -1, self.n_classes)

        # MidFusionMultiClipMaxTracks
        return {'inters': output_ints, 'rels': output_ctx}


class GatingUnit(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super(GatingUnit, self).__init__()
        self.in_dim1, self.in_dim2, self.out_dim = in_dim1, in_dim2, out_dim
        self.fc_out = nn.Linear(in_dim1 + in_dim2, out_dim)
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        xi, xr = x['inters'], x['rels']
        batch = xr.shape[0]
        tmp = torch.cat((xr, xi.view(batch, -1, self.in_dim1)[:, 0, :]), dim=-1)
        tmp = self.dropout(torch.relu(self.fc_out(tmp)))
        return tmp


class MultiTaskCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, weights=None, n_rels=0):
        super(MultiTaskCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.n_rels = n_rels
        if weights is not None:
            self.weights = torch.tensor(weights).float().to(opt.device)
        else:
            self.weights = None

    def forward(self, x, args):
        target_rels = args['rels_label'].long().to(opt.device)
        rel_mask = torch.nonzero(target_rels - self.n_rels)
        loss = 0
        target_inters = args['labels'].long().to(opt.device).reshape(-1)
        if self.weights is None:
            loss += F.cross_entropy(x['inters'], target_inters)
        else:
            loss += F.cross_entropy(x['inters'], target_inters, weight=self.weights)
        if rel_mask.shape[0]:
            loss += F.cross_entropy(x['rels'][rel_mask, np.arange(x['rels'].shape[-1])], target_rels[rel_mask].squeeze(1))
        return loss


class MultiTaskMaxMargin(nn.Module):
    def __init__(self, n_rels=0):
        super(MultiTaskMaxMargin, self).__init__()
        self.m = opt.margin
        self.n_rels = n_rels

    def forward(self, x, args):
        loss = torch.Tensor([0]).to(opt.device)
        batch = len(args['rels_label'])
        # interaction part
        if opt.ints == 1:
            inters = x['inters'].view(batch, -1, x['inters'].shape[-1])[:, 0]
            target = args['labels'][:, 0].squeeze()
            batch_idxs = list(range(target.shape[0]))
            neg_mask = torch.ByteTensor(np.ones(inters.shape)).to(opt.device)
            neg_mask[batch_idxs, target] = 0
            multilab_weights = args['multilab_weights'].to(opt.device).byte()
            neg_mask = neg_mask * multilab_weights
            inters = torch.sigmoid(inters)
            pos = inters[batch_idxs, target]
            neg = inters * neg_mask.float()
            loss += opt.lymbda * (((self.m - pos).view(-1, 1) + neg).relu() * neg_mask.float()).sum(1).mean()

        if opt.ctx == 1:
            # relationships part
            target_rels = args['rels_label']
            rel_mask = torch.nonzero(target_rels - self.n_rels)
            if rel_mask.shape[0]:
                target_rels = target_rels[rel_mask].squeeze(1)
                rels = x['rels'][rel_mask].squeeze(1)
                batch_idxs_rels = list(range(target_rels.shape[0]))
                neg_mask_rels = torch.ByteTensor(np.ones(rels.shape)).to(opt.device)

                neg_mask_rels[batch_idxs_rels,  target_rels] = 0
                rels = torch.sigmoid(rels)
                pos_rels = rels[batch_idxs_rels, target_rels]
                neg_rels = rels * neg_mask_rels.float()
                loss += (((self.m - pos_rels).view(-1,1) + neg_rels).relu() * neg_mask_rels.float()).sum(1).mean()
        return loss


class MaxMarginCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaxMarginCrossEntropyLoss, self).__init__()
        self.m = opt.margin

    def forward(self, x, args):
        batch = len(args['labels'])
        inters = x['inters']
        target = args['labels']
        batch_idxs = list(range(target.shape[0]))
        neg_mask = torch.ByteTensor(np.ones(inters.shape)).to(opt.device)
        neg_mask[batch_idxs, target] = 0
        multilab_weights = args['multilab_weights'].to(opt.device).byte()
        neg_mask = neg_mask * multilab_weights
        inters = torch.sigmoid(inters)
        pos = inters[batch_idxs, target]
        neg = inters * neg_mask.float()
        loss = (((self.m - pos).view(-1, 1) + neg).relu() * neg_mask.float()).sum(1).mean()

        return loss


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

        self.m = opt.tr_margin

    def forward(self, input, args):
        # loss = 0
        target = args['labels']
        mem_mask = args['mem_mask'].float()
        mem_mask_device = mem_mask.to(opt.device)
        assert opt.tr_maximize
        # define values for positive examples
        x = input['inters']
        batch_idxs = list(range(x.shape[0]))
        neg_mask = torch.ByteTensor(np.ones(x.shape)).to(opt.device) * mem_mask_device.unsqueeze(2).byte()
        x[~neg_mask] = float('-inf')
        multilab_weights = args['multilab_weights'].to(opt.device).unsqueeze(1).byte()
        neg_mask = neg_mask * multilab_weights
        if opt.tr_correct:
            neg_mask[batch_idxs, args['gt_tracks'][:,0], target] = 0
            neg_mask[batch_idxs, args['gt_tracks'][:,1], target] = 0
        else:
            neg_mask[batch_idxs, :, target] = 0
        if opt.tr_cat_distr:
            assert not opt.tr_correct
            probs = torch.softmax(x[batch_idxs, :, target], dim=1)
            max_idxs = torch.multinomial(probs, 1).view(-1)
            x = torch.sigmoid(x)
        else:
            x = torch.sigmoid(x)
            if opt.tr_correct:
                max_idxs = [0] * x.shape[0]
            else:
                # batch_size
                max_idxs = torch.argmax(x[batch_idxs, :, target] * mem_mask_device, dim=1)
        pos = x[batch_idxs, max_idxs, target]
        # define negatives based on wrong labels
        # for this particular tracks consider wrong labels
        if opt.tr_max_neg and opt.tr_sum_max_flag:
            # (batch_size, n_tracks)
            neg_max = torch.max(x * neg_mask.float(), dim=2)[0].float()
            loss = torch.sum(torch.relu((self.m - pos).view(-1, 1) + neg_max), dim=1)
        else:
            neg_mask = neg_mask.float()
            neg = (x * neg_mask).view(len(batch_idxs), -1)
            neg_mask = neg_mask.view(neg.shape)
            # (self.m - pos) -> (batch_size); view -> (batch_size, 1); + neg -> (batch_size, -1); -> (batch_size)
            loss = torch.sum(torch.relu((self.m - pos).view(-1, 1) + neg) * neg_mask, dim=1)
        loss = loss.mean()
        return loss


class MarginTrackRelsLoss(nn.Module):
    def __init__(self, n_rels=0):
        super(MarginTrackRelsLoss, self).__init__()
        self.m = opt.tr_margin
        self.n_rels = n_rels

    def forward(self, x, args):
        loss = torch.Tensor([0]).to(opt.device)
        batch = len(args['labels'])
        ######################### INTERACTIONS #################################
        target = args['labels']
        mem_mask = args['mem_mask'].float().to(opt.device)
        ints = x['inters']
        neg_mask_ints = torch.ByteTensor(np.ones(ints.shape)).to(opt.device) * mem_mask.unsqueeze(2).byte()
        batch_ints = list(range(ints.shape[0]))
        ints[~neg_mask_ints] = float('-inf')

        ######################### RELATIONSHIPS #################################
        target_rels = args['rels_label']
        rel_mask = ((target_rels - self.n_rels) != 0).to(opt.device).view(batch, -1, 1)

        rels = x['rels']
        batch_idxs_rels = list(range(target_rels.shape[0]))
        neg_mask_rels = torch.ByteTensor(np.ones(rels.shape)).to(opt.device) * mem_mask.unsqueeze(-1).byte() * rel_mask
        neg_mask_rels = torch.cat((neg_mask_rels, torch.zeros(rels.shape[0], rels.shape[1], 1).byte().to(opt.device)), dim=-1)
        rels = torch.cat((rels, torch.zeros(rels.shape[0], rels.shape[1], 1).to(opt.device)), dim=-1)
        # mask None class
        rels[~neg_mask_rels] = float('-inf')

        multilab_weights = args['multilab_weights'].to(opt.device).unsqueeze(1).byte()
        neg_mask_ints = neg_mask_ints * multilab_weights
        if opt.tr_correct:
            neg_mask_ints[batch_ints, args['gt_tracks'][:,0], target] = 0
            neg_mask_ints[batch_ints, args['gt_tracks'][:,1], target] = 0
            neg_mask_rels = neg_mask_rels.view(-1, self.n_rels + 1)
            neg_mask_rels[list(range(neg_mask_rels.shape[0])), target_rels.view(-1)] = 0
            neg_mask_rels = neg_mask_rels.view(batch, -1, self.n_rels + 1)
        else:
            neg_mask_ints[batch_ints, :, target] = 0
            neg_mask_rels[batch_idxs_rels, :, target_rels[batch_idxs_rels, args['gt_tracks'][:,0]]] = 0
            neg_mask_rels[batch_idxs_rels, :, target_rels[batch_idxs_rels, args['gt_tracks'][:,1]]] = 0
        if opt.tr_cat_distr:
            assert not opt.tr_correct
            probs_cl = torch.softmax(ints[batch_ints, :, target], dim=1)
            probs_rels= torch.softmax(rels[batch_idxs_rels, :, target_rels[batch_idxs_rels, args['gt_tracks'][:,0]]], dim=1)
            probs_rels[probs_rels!=probs_rels] = 0
            max_idxs = torch.multinomial((probs_cl + probs_rels)/2, 1).view(-1)
            ints = torch.sigmoid(ints)
            rels = torch.sigmoid(rels)
        else:
            ints = torch.sigmoid(ints)
            rels = torch.sigmoid(rels)
            if opt.tr_correct:
                max_idxs = [0] * ints.shape[0]
            else:
                mat = ints[batch_ints, :, target] + rels[batch_idxs_rels, :, target_rels[batch_idxs_rels, args['gt_tracks'][:,0]]]
                max_idxs = torch.argmax(mat * mem_mask, dim=1)
        pos = ints[batch_ints, max_idxs, target]
        pos_rels = rels[batch_idxs_rels, max_idxs, target_rels[batch_idxs_rels, args['gt_tracks'][:,0]]]

        if opt.tr_max_neg and opt.tr_sum_max_flag:
            neg_max = torch.max(ints * neg_mask_ints.float(), dim=2)[0].float()
            neg_max_rels = torch.max(rels * neg_mask_rels.float(), dim=2)[0].float()

            loss += opt.lymbda * torch.sum(torch.relu((self.m - pos).view(-1, 1) + neg_max), dim=1).mean()
            loss += torch.sum(torch.relu((self.m - pos_rels).view(-1,1) + neg_max_rels), dim=1).mean()
        else:
            neg_mask_ints = neg_mask_ints.float()
            neg = (ints * neg_mask_ints).view(len(batch_ints), -1)
            neg_mask_ints = neg_mask_ints.view(neg.shape)

            neg_mask_rels = neg_mask_rels.float()
            neg_rels = (rels * neg_mask_rels).view(batch, -1)
            neg_mask_rels = neg_mask_rels.view(neg_rels.shape)

            loss += opt.lymbda * torch.sum(torch.relu((self.m - pos).view(-1, 1) + neg) * neg_mask_ints, dim=1).mean()
            loss += torch.sum(torch.relu((self.m - pos_rels).view(-1, 1) + neg_rels) * neg_mask_rels, dim=1).mean()

        return loss


def create_model(n_classes,  n_rels=0):
    if opt.tr_maximize:
        model = MidFusionMultiClipMaxTracks(n_classes=n_classes, n_rels=n_rels).to(opt.device)
    else:
        model = MidFusionMultiClip(n_classes=n_classes, n_rels=n_rels).to(opt.device)
    if opt.mod_check:
        model = Modalities(n_classes=n_classes).to(opt.device)

    # loss
    if opt.tr_maximize:
        if opt.rels_multitask:
            # maximizing over evrything
            loss = MarginTrackRelsLoss(n_rels=n_rels)
        else:
            loss = MarginLoss()
    else:
        if opt.rels_multitask:
            loss = MultiTaskMaxMargin(n_rels=n_rels)
        else:
            loss = MaxMarginCrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    print(str(model))
    for name, param in model.named_parameters():
        print('%s\n%s' % (str(name),  str(param.norm())))
    # pprint({name: param.norm() for name, param in model.named_parameters()})
    print(str(loss))
    print(str(optimizer))
    return model, loss, optimizer

