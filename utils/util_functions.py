#!/usr/bin/env python

"""Utilities for the project"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import re
import csv
import json
import time
import torch
import pickle
import numpy as np
import os.path as ops
from itertools import combinations
from collections import defaultdict, Counter

from utils.arg_pars import opt


class Averaging(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Precision(object):
    def __init__(self):
        self.true_prediction = 0
        self.total = 0
        self.avg = 0

    def update(self, true_pr, total):
        self.true_prediction += true_pr
        self.total += total
        self.avg = self.true_prediction / self.total


class Relationship:
    def __init__(self, rels_name, scene_idx):
        self.rels_name = rels_name
        self.scenes = set([scene_idx])
        self.rel2scenes = defaultdict(list)
        self._scene2rel = defaultdict(list)
        self.rel2scenes[rels_name].append(scene_idx)
        self._scene2rel[scene_idx].append(rels_name)

    def append_scene(self, rels_name, scene_idx):
        if rels_name not in [self.rels_name, None]:
            self.rels_name = rels_name
        if scene_idx in self.scenes and self.rels_name in self._scene2rel[scene_idx]:
            return
        self.scenes.add(scene_idx)
        self.rel2scenes[self.rels_name].append(scene_idx)
        self._scene2rel[scene_idx].append(self.rels_name)

    def scene2rel(self, scene_idx):
        if scene_idx in self._scene2rel:
            return np.random.choice(self._scene2rel[scene_idx])
        else:
            return 'None'



class AnnotatedInter:
    def __init__(self, clip, node_id):
        self.inter_node = clip.G.node[node_id]  # node which contains an interaction
        self.video_descr = clip.video  # ss, movie, scene (list), fname (endswith mp4), es
        self.time_node = None  # start, end
        self.ftracks = defaultdict(list)
        self.ftracks_names = {}
        self.id2names = {}  # old names
        self.paired_names = {}
        self.name2id = {}  # old names_id
        self.bi = False  # if this interaction is bidirectional
        self.id = None

        self.triplets = {}
        self.tripl_counter = 0
        self.relships = {}

        self._add_time(clip, node_id)
        self._add_names(clip, node_id)

    def _add_time(self, clip, node_id):
        for neighbor in clip.G.neighbors(node_id):
            if clip.G.node[neighbor]['type'] == 'time':
                self.time_node = clip.G.node[neighbor]
                if self.video_descr['movie'] == 'tt0119822' and 'scene-006.ss-0045.es-0048' in self.video_descr['fname'][0]:
                    if clip.G.node[neighbor]['start'] == 9:
                        self.time_node = {'start':8, 'end': 9, 'type': 'time'}
                    # import pdb; pdb.set_trace()
                break


    def _add_names(self, clip, node_id):
        idx = 0
        for node_entity in clip.get_node_ids_of_type('entity'):
            if clip.G.has_edge(node_id, node_entity) or clip.G.has_edge(node_entity, node_id):
                name = clip.G.node[node_entity]['name'].lower()
                self.name2id[name] = node_entity
                self.id2names[node_entity] = name

        # if 'vijay' in self.name2id and 'paulie bleeker' in self.name2id:
        #     print('vijay + pauilie')
        #     print(self.video_descr)
        #     print()
        # if 'juno macguff' in self.name2id and 'mac macguff' in self.name2id:
        #     print('juno + mac')
        #     print(self.video_descr)
        #     print()


    def order_names(self, clip, node_id):
        # if len(self.names) <= 1: raise ValueError('smth wrong with names, should be at least two')
        for name_id1, name_id2 in combinations(self.name2id.values(), 2):
            # if bi
            self.bi = False
            if clip.G.has_edge(name_id1, node_id) and clip.G.has_edge(node_id, name_id2) \
                    and clip.G.has_edge(name_id2, node_id) and clip.G.has_edge(node_id, name_id1):
                self.bi = True

            name1, name2 = self.id2names[name_id1], self.id2names[name_id2]

            if clip.G.has_edge(name_id1, node_id) and clip.G.has_edge(node_id, name_id2):
                self.triplets[self.tripl_counter] = {0: name1, 1: name2}
                self.tripl_counter += 1
            if clip.G.has_edge(name_id2, node_id) and clip.G.has_edge(node_id, name_id1):
                self.triplets[self.tripl_counter] = {0: name2, 1:name1}
                self.tripl_counter += 1

        # if there is no pairs corresponding to the interaction -> at least one person should be
        if not self.triplets:
            for name_id in self.name2id.values():
                # print('bad thing with names, should be at least two')
                if clip.G.has_edge(name_id, node_id):
                    self.triplets[self.tripl_counter] = {0: self.id2names[name_id]}
                    self.tripl_counter += 1
                if clip.G.has_edge(node_id, name_id):
                    self.triplets[self.tripl_counter] = {1: self.id2names[name_id]}
                    self.tripl_counter += 1

    def add_ftracks(self, tracks):
        for idx, track in enumerate(tracks['ftracks']):
            for name_idx, name in self.id2names.items():
                if tracks['names'][idx] is not None and \
                        (tracks['names'][idx] in name.split() or tracks['names'][idx] == name):
                    break
            else: continue
            # if tracks['names'][idx] not in self.names: continue
            start_time = max(self.time_node['start'], track[0]['timestamp'])
            end_time = min(self.time_node['end'], track[-1]['timestamp'])
            if start_time >= end_time: continue
            left, right = 0, len(track) - 1

            def find_bound(left, right, point):
                while left < right:
                    mid = left + right >> 1
                    if track[mid]['timestamp'] >= point: right = mid
                    else: left = mid + 1
                return left

            start_idx = find_bound(left, right, start_time)
            end_idx = find_bound(left, right, end_time)
            self.ftracks[name] += track[start_idx: end_idx + 1]
            # self.ftracks_names[name_idx] = name

        if self.video_descr['movie'] == 'tt0467406' and self.video_descr['scene'] == [63]:
            print(self.video_descr)
            
        for name in self.id2names.values():
            if name not in self.ftracks:
                self.ftracks[name] = []

    def add_relationships(self, clip, node_id, dict_rel, rels_15, rels_opp):
        node_ids_rel = clip.get_node_ids_of_type('relationship')
        scene_idx = clip.video['scene'][0]
        for node_id_rel in node_ids_rel:
            rel_name = clip.G.node[node_id_rel]['name']
            rel_name = rels_15[rel_name]
            for tripl_id in self.triplets:
                if len(self.triplets[tripl_id]) == 2:
                    name1, name2 = self.triplets[tripl_id][0], self.triplets[tripl_id][1]
                    name_id1, name_id2 = self.name2id[name1], self.name2id[name2]
                    if clip.G.has_edge(name_id1, node_id_rel) and clip.G.has_edge(node_id_rel, name_id2):
                        # self.relships[tripl_id] = [rel_name]
                        names_key = tuple([name1, name2])
                        if names_key in dict_rel:
                            dict_rel[names_key].append_scene(rel_name, scene_idx)
                            dict_rel[tuple([name2, name1])].append_scene(rels_opp[rel_name], scene_idx)
                            # dict_rel[tuple([name2, name1])].append_scene(rel_name, scene_idx)
                        else:
                            dict_rel[names_key] = Relationship(rel_name, scene_idx)
                            dict_rel[tuple([name2, name1])] = Relationship(rels_opp[rel_name], scene_idx)
                            # dict_rel[tuple([name2, name1])] = Relationship(rel_name, scene_idx)
                    elif clip.G.has_edge(name_id2, node_id_rel) and clip.G.has_edge(node_id_rel, name_id1):
                        names_key = tuple([name2, name1])
                        if names_key in dict_rel:
                            dict_rel[names_key].append_scene(rel_name, scene_idx)
                            dict_rel[tuple([name1, name2])].append_scene(rels_opp[rel_name], scene_idx)
                            # dict_rel[tuple([name1, name2])].append_scene(rel_name, scene_idx)
                        else:
                            dict_rel[names_key] = Relationship(rel_name, scene_idx)
                            dict_rel[tuple([name1, name2])] = Relationship(rels_opp[rel_name], scene_idx)
                            # dict_rel[tuple([name1, name2])] = Relationship(rel_name, scene_idx)

        for track_pair_rels in dict_rel.values():
            if scene_idx in track_pair_rels.scenes: continue
            track_pair_rels.append_scene(rels_name=None, scene_idx=scene_idx)

        for tripl_id in self.triplets:
            if len(self.triplets[tripl_id]) == 1: continue
            # if tripl_id in self.relships: continue
            name1, name2 = self.triplets[tripl_id][0], self.triplets[tripl_id][1]
            names_key = tuple([name1, name2])
            if names_key in dict_rel:
                self.relships[tripl_id] = dict_rel[names_key]._scene2rel[scene_idx]
        return dict_rel

    def get_relship_by_id(self, triplet_id):
        if triplet_id in self.relships:
            return np.random.choice(self.relships[triplet_id])

        else:
            return 'None'


def join_data(data1, data2, f):
    """Simple use of numpy functions vstack and hstack even if data not a tuple

    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy

    Returns:
        Joined data with provided method.
    """
    if isinstance(data1, torch.Tensor):
        data1 = data1.numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.numpy()
    if isinstance(data2, tuple):
        data2 = f(data2)
    if data2 is None:
        data2 = data1
    elif data1 is not None:
        data2 = f((data1, data2))
    return data2


def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_model(name='mlp_text'):
    if opt.device == 'cpu':
        checkpoint = torch.load(opt.resume_str, map_location='cpu')
    else:
        checkpoint = torch.load(opt.resume_str)
    checkpoint = checkpoint['state_dict']
    print('loaded model: ' + ' %s' % opt.resume_str)
    return checkpoint


def load_optimizer():
    if opt.device == 'cpu':
        checkpoint = torch.load(opt.resume_str, map_location='cpu')
    else:
        checkpoint = torch.load(opt.resume_str)
    checkpoint = checkpoint['optimizer']
    print('loaded optimizer')
    return checkpoint


def timing(f):
    """Wrapper for functions to measure time"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
                     % (f, (time2-time1)*1000.0,
                        (time2-time1)/60.0,
                        (time2-time1)))
        return ret
    return wrap


def dir_check(path):
    """If folder given path does not exist it is created"""
    if path == '':
        return
    else:
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except FileNotFoundError:
            dir_check(os.path.split(path)[0])
            dir_check(path)


def load_set(mode='train'):
    if opt.sanity_check:
        if mode == 'test':
            return ['tt0120338']
        else:
        # return ['tt0478311']
            return ['tt0108160']
    with open(opt.split_path, 'r') as f:
        splits = json.load(f)
    return splits[mode]


def load_interaction_names(idx2inter_ret=False):
    '''
    Each line in the file in the form: interaction_name #_interactions interaction_class (t | v | m \ all)
    :return:
    '''
    interactions = defaultdict(list)
    with open(opt.labeled_interactions, 'r') as f:
        for line in f:
            line = line.strip().split()
            interactions[line[-1]].append(' '.join(line[:-2]))
            interactions['all'].append(' '.join(line[:-2]))
    inter2idx = {}
    idx2inter = {}
    idx = 0
    map_inter_class2idx = {'t': 0, 'v': 1, 'm': 2}
    for key_idx, (key, inter_list) in enumerate(interactions.items()):
        if key == 'all':
            continue
        for local_idx, inter in enumerate(inter_list):
            inter2idx[inter] = (idx, map_inter_class2idx[key], local_idx)
            idx2inter[(key, local_idx)] = inter
            idx += 1
    if idx2inter_ret:
        return interactions, inter2idx, idx2inter
    return interactions, inter2idx


def load_merged_interactions():
    mgd_inters = {}
    mgd2idx = {}
    inter2mgd = {}
    with open(opt.merged_interactions, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split('\t')
            key, all_possibilities = line[0], line[1:]
            mgd_inters[key] = all_possibilities
            mgd2idx[key] = line_idx
            for inter_name in all_possibilities:
                inter2mgd[inter_name] = key
    return inter2mgd, mgd2idx


def load_iou2_any():
    iou2 = {}
    with open(ops.join(opt.intersected, 'intersected_any.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            iou2[row[0]] = row[1:]
    return iou2


def load_iou2_movies():
    iou2_movies = {}
    pattern = r'intersected_(tt\d*).csv'
    for filename in os.listdir(opt.intersected):
        search = re.search(pattern, filename)
        if search is None: continue
        iou2 = {}
        with open(ops.join(opt.intersected, filename), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                iou2[row[0]] = row[1:]
        iou2_movies[search.group(1)] = iou2
    return iou2_movies

def load_iou2_clips():
    iou2_clips = {}
    pattern = r'intersected_(tt\d*)_(\d*).csv'
    for filename in os.listdir(opt.intersected):
        search = re.search(pattern, filename)
        if search is None: continue
        iou2 = {}
        with open(ops.join(opt.intersected, filename), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                iou2[row[0]] = row[1:]
        iou2_clips[(search.group(1), int(search.group(2)))] = iou2

    return iou2_clips


def load_annotations(movie_idx='all'):
    with open(opt.annotations, 'rb') as f:
        all_mg = pickle.load(f, encoding='latin1')
    if movie_idx == 'all':
        for movie in all_mg.values():
            yield movie
    elif type(movie_idx) == list:
        for movie in movie_idx:
            yield all_mg[movie]
    else:
        yield all_mg[movie_idx]


def load_relships():
    rels_to_15 = {}
    rels_opp = {}
    with open(opt.relships2_15, 'r') as f:
        for line in f:
            line = line.strip().split()
            rels_to_15[' '.join(line[:-1])] = line[-1]
    with open(opt.relships_opp, 'r') as f:
        for line in f:
            line = line.strip().split()
            rels_opp[line[0]] = line[1]
    return rels_to_15, rels_opp


def merged_clips_processing():
    merged_clips = defaultdict(dict)
    with open(opt.merged_videos, 'r') as fid:
        for line in fid:
            line = line.strip().split()  # movie id, scene id (-1), scene name
            if '---' in line[2]:
                continue
            # idx1-> movie id
            #   idx2 -> scene id, value = clip name
            merged_clips[line[0]][int(line[1]) + 1] = line[2]
    return merged_clips


def load_annotated_inter(movie_idxs='all',
                         node_types=('interactions', 'summary'),
                         inter_class='all'):
    output = []

    if movie_idxs is None:
        return output
    np.random.seed(opt.seed)
    counter = set()
    # inter_names = load_interaction_names()[0][inter_class]
    if inter_class == 'all':
        inter_names = list(load_interaction_names()[1].keys())
    else:
        inter_names = load_interaction_names()[0][inter_class]
    merged_clips = merged_clips_processing()
    ftracks = load_tracks(movie_idxs)
    # movie is MovieGraph object
    # movie.imdb_key gives its movie_id
    track_stat = defaultdict(int)
    track_stat_scene = {0:0, 1:0, 2:0}
    inter_stat = []
    tracks_checked = set()
    inter_id = 0
    rel_stat = defaultdict(int)
    rel_15_stat = defaultdict(int)
    dict_rels = defaultdict(dict)
    rels_15, rels_opp = load_relships()
    for movie in load_annotations(movie_idxs):
        # clip is ClipGraph object
        # clip.G is DiGraph
        # clip.video['movie']: 'tt0120338'
        # clip.video['fname']: <class 'list'>: ['scene-002.ss-0009.es-0020.mp4']
        # clip.video['scene']: [2]
        for clip in movie.clip_graphs.values():
            # process merged clips
            if len(clip.video['fname']) > 1:
                for scene_idx in clip.video['scene']:
                    if scene_idx in merged_clips[movie.imdb_key]:
                        clip.video['scene'] = [scene_idx]
                        clip.video['fname'] = [merged_clips[movie.imdb_key][scene_idx]]
                        break
                else:
                    continue
            node_ids_rel = clip.get_node_ids_of_type('relationship')
            for node_id_rel in node_ids_rel:
                rel_stat[clip.G.node[node_id_rel]['name']] += 1
                # if clip.G.node[node_id_rel]['name'] == '___attribute___':
                #     print('alaaaarm')
                # if clip.G.node[node_id_rel]['name'] == 'colleague':
                    # print('colleague')


            for node_type in node_types:
                node_ids = clip.get_node_ids_of_type(node_type)
                for node_id in node_ids:
                    if clip.G.node[node_id]['name'] not in inter_names:
                        continue
                    # if clip.G.node[node_id]['type'] not in node_types: continue
                    # for neighbor in clip.G.neighbors(node_id):
                        # if clip.G.node[neighbor]['type'] == 'time':
                    annotated_inter = AnnotatedInter(clip, node_id)
                    inter_stat.append(clip.G.node[node_id]['name'])
                    # annotated_inter.time_node = clip.G.node[neighbor]

                    # add tracks
                    scene_idx = ops.splitext(clip.video['fname'][0])[0]
                    try:
                        tracks = ftracks[(movie.imdb_key, scene_idx)]
                    except KeyError:
                        tracks = {'ftracks': [], 'names': [], 'check': ['-', (movie.imdb_key, scene_idx,
                                                                              'no file')]}
                    if (movie.imdb_key, scene_idx) not in tracks_checked:
                        a = set()
                        for ai in tracks['names']:
                            if ai is not None: a.add(ai)
                        track_stat_scene[0] += len(a)
                        track_stat_scene[1] += 1
                        if len(a) == 0: track_stat_scene[2] += 1
                        tracks_checked.add((movie.imdb_key, scene_idx))

                    if tracks['check'][0].startswith('-'):
                        counter.add(tracks['check'][1])
                        # counter.update(tuple(tracks['check'][1]))
                        # print(tracks['check'][1])

                    annotated_inter.order_names(clip, node_id)
                    annotated_inter.add_ftracks(tracks)
                    dict_rels[movie.imdb_key] = annotated_inter.add_relationships(clip, node_id,
                                                                                  dict_rels[movie.imdb_key],
                                                                                  rels_15, rels_opp)

                    annotated_inter.id = inter_id
                    inter_id += 1
                    output.append(annotated_inter)

    for elem in counter:
        print(elem)
    print(len(counter))
    print('track stat', track_stat)
    print('track stat scene', track_stat_scene, opt.inter_class)
    print('0: how many tracks in total\n1: how many scenes\n2: how many scenes wihtout tracks')
    c = Counter(inter_stat)
    print(c)
    # for rel_key in sorted(rel_stat.keys(), key=lambda x: -rel_stat[x]):
    #     print('%s\t%s' % (rel_key, rel_stat[rel_key]))
    print('relationships: ', rel_stat)
    for imdb_rels in dict_rels.values():
        for r in imdb_rels.values():
            rel_15_stat[r.rels_name] += 1
    print('again relationships: ', rel_15_stat)
    rels_opp['None'] = None
    if opt.rels or opt.rels_multitask:
        return output, dict_rels, list(rels_opp.keys()), rels_opp
    return output


def load_tracks(movie_idxs: list):
    ftracks_all = {}
    for movie_idx in movie_idxs:
        with open(ops.join(opt.ftack_ids, '%s.json' % movie_idx), 'r') as f:
            ftracks_ids = json.load(f)
        for scene_ftrack in os.listdir(ops.join(opt.ftracks, movie_idx)):
            with open(ops.join(opt.ftracks, movie_idx, scene_ftrack), 'r') as f:
                ftracks = json.load(f)['ftracks']
                scene_idx = ops.splitext(scene_ftrack)[0]
                try:
                    names = []
                    for name in ftracks_ids[scene_idx]:
                        if name is not None:
                            names.append(name.lower())
                        else: names.append(None)
                    ftracks_all[(movie_idx, scene_idx)] = {'ftracks':ftracks,
                                                           'names': names,
                                                           'check': ['+', ()]}
                    # print('++++++++++')
                except KeyError:
                    if len(ftracks) == 0:
                        ftracks_all[(movie_idx, scene_idx)] = {'ftracks': ftracks,
                                                               'names': [],
                                                               'check': ['+', ()]}
                    else:
                        ftracks_all[(movie_idx, scene_idx)] = {'ftracks': ftracks,
                                                               'names': ['other'] * len(ftracks),
                                                               'check': ['-', (movie_idx, scene_idx, len(ftracks))]}
                    # if ftracks:
                    #     print('error', len(ftracks))
    return ftracks_all


def load_orig_resol():
    resol = {}
    with open(opt.orig_res, 'r') as f:
        for line in f:
            movie_idx, h, w = line.strip().split()
            resol[movie_idx] = [int(h), int(w)]
    return resol



