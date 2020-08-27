#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2019'


import torch
import numpy as np
from utils.arg_pars import opt
from utils.util_functions import Averaging, load_interaction_names
from utils.evaluation import Precision, RelationshipsAcc


def testing(test_dataset, model, loss,total_iter=1, mode='val', train_start_time=''):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              drop_last=False)
    losses = Averaging()
    model.eval()
    prec = Precision(inter2mgd=test_dataset.interidx2mgdidx, n_rels=opt.rels_dim)
    n_rels = test_dataset.n_rels
    if opt.rels_multitask:
        prec_rels = RelationshipsAcc(n_rels=n_rels)
    interactions, inter2idx = load_interaction_names()
    idx2inter = {}
    for inter, inter_idxs in inter2idx.items():
        idx2inter[inter_idxs[0 if opt.inter_class == 'all' else 2]] = inter
    total_tracks = 0
    conf_mat = np.zeros((test_dataset.n_classes, test_dataset.n_classes))
    with torch.no_grad():
        for idx, input in enumerate(test_loader):
            labels = input['labels']
            if len(labels) == 1:
                continue
            output = model(input)
            loss_values = loss(output, input)
            losses.update(loss_values.item(), len(output))
            if opt.soft_gt:
                conf_mat = prec.update_probs(output['inters'].cpu(), input['labels'], soft_labels=input['soft_labels'],
                                              conf_mat=conf_mat)
            elif opt.tr_maximize:
                if opt.ints == 1 and opt.ctx == 0:
                    bs = input['labels'].shape[0]
                    n_classes = output['inters'].shape[-1]
                    inters = output['inters'].cpu().reshape(bs, -1, n_classes)
                    target_inters = input['labels']
                    prec.update_probs_max_tracks(inters.cpu(), gt_tracks=input['gt_tracks'],
                                                 gt_classes=target_inters, print_batch=idx==0,
                                                 n_names=input['n_names'], mask=input['mem_mask'].cpu(), just_zeros=input['just_zeros'])
                if opt.ints == 1 and opt.ctx == 1:
                    bs = input['labels'].shape[0]
                    n_classes = output['inters'].shape[-1]
                    inters = output['inters'].cpu().reshape(bs, -1, n_classes)
                    target_inters = input['labels']
                    rels = output['rels'].cpu()
                    target_rels = input['rels_label']
                    rels_mask = torch.nonzero(input['rels_label'][:, 0] - n_rels + 1)
                    prec.update_probs_max_tracks_rels(inters, rels, target_inters, target_rels,
                                                      gt_tracks=input['gt_tracks'],
                                                      just_zeros=input['just_zeros'],
                                                      mask=input['mem_mask'].cpu(),
                                                      rels_mask=rels_mask)


            else:
                if opt.rels_multitask:
                    if opt.tracks and opt.tr_maximize:
                        total_tracks += np.sum(output['just_zeros'])
                    if opt.ints == 1:
                        bs = input['labels'].shape[0]
                        n_classes = output['inters'].shape[-1]
                        inters = output['inters'].cpu().reshape(bs, -1, n_classes)[:, 0]
                        target_inters = input['labels'][:, 0].reshape(-1)
                        conf_mat = prec.update_probs(inters, target_inters, conf_mat=conf_mat)
                    if opt.ctx == 1:
                        rels_mask = torch.nonzero(input['rels_label'] - n_rels + 1)
                        if rels_mask.shape[0]:
                            rels = output['rels'].cpu()[rels_mask].squeeze(1)
                            target_rels = input['rels_label'][rels_mask].squeeze(1)
                            target_hashes = input['hash_rel'][rels_mask].squeeze(1)

                            prec_rels.update(rels, target_rels, target_hashes)


                else:
                    if opt.tracks:
                        total_tracks += np.sum(input['just_zeros'].cpu().detach().numpy())
                    conf_mat = prec.update_probs(output['inters'].cpu(), input['labels'], conf_mat=conf_mat)
    print(prec.total)
    print('tracks # %d' % total_tracks)

    out_val_ints = 0
    out_val_rels = 0
    out_val_tr = 0
    out_val_joint = 0
    out_val = 0
    if opt.ints == 1:
        print('%s loss: %f' % (mode.upper(), losses.avg))
        print('%s pr@1: %f' % (mode.upper(), prec.top1()))
        if not opt.tr_maximize:
            print('%s pr@5: %f' % (mode.upper(), prec.top5()))
        out_val_ints = prec.top1()
        out_val_joint = prec.top1()
        out_val += out_val_ints

    if opt.soft_gt:
        print('%s pr soft@1 %f' % (mode.upper(), prec.top1_sf()))
        print('%s pr soft@5 %f' % (mode.upper(), prec.top5_sf()))
    if opt.tr_maximize:
        out_val_ints = prec.cls_top1()
        out_val_tr = prec.trks_top1()
        out_val = out_val + out_val_tr + out_val_ints
        print('%s pr@trks: %f' % (mode.upper(), prec.trks_top1()))
        print('%s pr@cls: %f' % (mode.upper(), prec.cls_top1()))
        if opt.ctx == 1:
            out_val_rels = prec.rels_top1()
            print('%s pr@rels: %f' % (mode.upper(), prec.rels_top1()))
            out_val += out_val_rels

    if opt.rels_maximize:
        prec.movie_rels()
        print('%s pr@rels: %f' % (mode.upper(), prec.rels_top1()))
        print('%s pr@cls: %f' % (mode.upper(), prec.cls_top1()))
        print('%s glob@rels: %f' % (mode.upper(), prec.rels_global()))
    if opt.rels_multitask and opt.ctx == 1:
        if not opt.tr_maximize:
            out_val_rels = prec_rels.top1()
            out_val += out_val_rels
            print('%s rels@top1: %f' % (mode.upper(), prec_rels.top1()))
            print('%s rels@top3: %f' % (mode.upper(), prec_rels.top3()))
            print('%s rel+int: %f' % (mode.upper(), out_val))

    out = {'total':out_val, 'ints':out_val_ints}
    if opt.rels_multitask:
        out.update({'rels':out_val_rels})

    if opt.tr_maximize:
        out.update({'tracks':out_val_tr, 'joint': out_val_joint})

    return out
