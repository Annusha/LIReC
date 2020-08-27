#!/usr/bin/env python

""" General parameters.
"""

__all__ = ['opt']
__author__ = 'Anna Kukleva'
__date__ = '2020'


import argparse

parser = argparse.ArgumentParser()

##########################################
# PARAMETERS TO CHANGE
parser.add_argument('--project_root', default='/sequoia/data1/akukleva/projects/cvpr20')
parser.add_argument('--data_root', default='/meleze/data1/akukleva/moviegraph/files_to_release/')
parser.add_argument('--store_root', default='/meleze/data1/akukleva/moviegraph/store')

##########################################
# DO NOT CHANGE THE REST
parser.add_argument('--dialogs_path', default='/dialogs')
parser.add_argument('--frame2time_path', default='/frame2time')

parser.add_argument('--labeled_interactions',
                    default='/others/all_train_set.txt')
parser.add_argument('--merged_interactions',
                    default='/others/merged_interactions.txt')
parser.add_argument('--annotations', default='/others/mg3.pkl')
parser.add_argument('--split_path', default='/others/split.json',
                    help='json file with splits: train | val | test')
parser.add_argument('--intersected', default='/intersections',
                    help='first column - name, others are with which intersects')
parser.add_argument('--relships2_15', default='/others/relships_many2_15.txt')
parser.add_argument('--relships_opp', default='/others/relships_15_opp.txt')
parser.add_argument('--merged_videos',
                    default='/others/use_vid_for_moviegraphs')
parser.add_argument('--inter_class', default='m',
                    help='t - text based interactions'
                         'v - visual based interactions'
                         'm - interaction based on text and visual features'
                         'all - everything together')
parser.add_argument('--feature_type', default='m',
                    help='m - mixed: text + visual'
                         't - text features'
                         'v - visual features')
parser.add_argument('--modality', default='m', help='m | t | v')

parser.add_argument('--soft_gt', default=False, type=bool,
                    help='(de)activate soft testing for interactions')
parser.add_argument('--multilab_weights', default=True)

##########################################
# TEXT

parser.add_argument('--ext_dialog', default='webvtt',
                    help='extension of the files which contain dialogs')
parser.add_argument('--text_features', default='bert_base',
                    help='bert_base ')
parser.add_argument('--contextualization', default='second-to-last',
                    help='second-to-last | last | sum-all | sum-last-4 | cat-last-4')


##########################################
# VISUAL

parser.add_argument('--visual_features', default='i3d')
# fixed (16 fps)
parser.add_argument('--sampling_fr', default=0.0625, type=int)
parser.add_argument('--ext_frame2time', default='matidx')


##########################################
# HYPERPARAMETERS

parser.add_argument('--joint_dim', default=512, type=int,
                    help='dimensionality of the joined embedding for text and video')
parser.add_argument('--pool_features', default='max',
                    help='how to join arbitrary number of features for the interaction'
                         'max | sum | mix | avg')
parser.add_argument('--i3d', default='spat', help='')
parser.add_argument('--spat_pool', default=True, type=bool,
                    help='True: first spatial average and then temporal max'
                         'False: first temporal max and then spatial average')
parser.add_argument('--merged', default=True, type=bool,
                    help='(dis)able merged interaction classes from 324 to 101 classes')


##########################################
# MAX MARGIN LOSS

parser.add_argument('--margin', default=0.101, type=float)

##########################################
# PERSONS TRACKS

parser.add_argument('--ftack_ids', default='/ftrack_ids')
parser.add_argument('--ftracks', default='/ftracks')
parser.add_argument('--tracks', action='store_true',
                    help='(de)activate person tracking')
parser.add_argument('--tf_crop', default=True, type=bool,
                    help='(un)croped spatially persons in spatial dim')
parser.add_argument('--orig_res', default='/others/org_res.txt',
                    help='txt file with original resolition of movies')
parser.add_argument('--tr_maximize', action='store_true',
                    help='apply maximization over possible tracks during training')
parser.add_argument('--tr_cat_distr', action='store_true',
                    help='instead of taking max -> sample from multinomial (categorical) distribution')
parser.add_argument('--tr_max_neg', action='store_true',
                    help='instead of sum for negatives use max to choose right track')
parser.add_argument('--tr_margin', default=0.101, type=float)
parser.add_argument('--tr_sum_max', action='store_true')
parser.add_argument('--tr_sum_max_flag', action='store_false')
parser.add_argument('--tr_correct', action='store_true')

##########################################
# PERSONS RELATIONSHIPS

parser.add_argument('--rels', action='store_true',
                    help='turn on/off relationships between people')
parser.add_argument('--rels_dim', default=0, type=int)
parser.add_argument('--rels_dim_out', default=24, type=int,
                    help='output dim of the embedding for the relationship branch')
parser.add_argument('--rels_maximize', default=False, type=bool,
                    help='apply maximization over possible relationships during training')
parser.add_argument('--rels_multitask', action='store_true',
                    help='wihtout relationships as input but try to predict relationship from clip + tracks')
parser.add_argument('--rels_multi_clip', action='store_true')
parser.add_argument('--rels_n_clips', default=6, type=int)


##########################################
# GATING AND AGGREGATION

parser.add_argument('--lymbda', default=1, type=float)

parser.add_argument('--ints', default=0, type=int)
parser.add_argument('--ctx', default=0, type=int)
parser.add_argument('--gates', default=0, type=int)

parser.add_argument('--mid_m_ints', default=6, type=int)

parser.add_argument('--mod_check', action='store_true')

##########################################
# NETWORK HYPERPARAMETERS

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lr', default=3e-5, type=float)
parser.add_argument('--lr_int', default=5, type=int)
parser.add_argument('--lr_pfx', default=3, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--num_workers', default=4,
                    help='number of threads for dataloading')
parser.add_argument('--device', default='cuda',
                    help='cuda | cpu')


##########################################
# MODELS

parser.add_argument('--save_model', default=True, type=bool,
                    help='save or not the trained model in the end of training')
parser.add_argument('--save_model_often', default=False, type=bool,
                    help='save model every 10 epochs')
parser.add_argument('--test', default=True, type=bool)
parser.add_argument('--test_fr', default=2, type=int)
parser.add_argument('--resume', default=False, type=bool,
                    help='resume without training, just testing')
parser.add_argument('--resume_train', default=False, type=bool)
parser.add_argument('--resume_str', default='',
                    help='name of the model to save or resume')
parser.add_argument('--model_name', default='')

parser.add_argument('--sanity_check', default=False, help='to check performance on the one video')



opt = parser.parse_args()

