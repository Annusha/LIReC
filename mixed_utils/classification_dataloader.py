#!/usr/bin/env python

"""
"""
from plotly.graph_objs.layout import scene

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'


from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import permutations

from utils.util_functions import *
from utils.arg_pars import opt
from mixed_utils.mixed_features import MixedFeatures
from utils.util_functions import load_iou2_clips, join_data


class Pair2Scene:
    def __init__(self):
        self.scenes2inters = defaultdict(list)

    def update(self, scene, inter_id):
        self.scenes2inters[scene].append(inter_id)


class MixedFeaturesDataset(Dataset):
    def __init__(self, mode='train'):
        print('load mixed features. mode: %s' % mode)
        interactions, self.inter2idx = load_interaction_names()
        self.inter2mgd, self.mgd2idx = load_merged_interactions()
        self.interidx2mgdidx = None
        self.merged2inter()
        self.test_rels_multi_clip = False
        if opt.merged:
            self.n_classes = len(self.mgd2idx)
        else:
            self.n_classes = len(interactions[opt.inter_class])
        self.mode = mode
        if mode == 'train':
            self.tracks = opt.tracks
        else:
            self.tracks = True
        self.triplets = opt.tr_maximize
        self._max_n_tripl = 0
        self.rels_n_clips = 0
        self.movie_idxs = load_set(mode=mode)
        if opt.rels or opt.rels_multitask:
            self.interactions, self.rels, self.rels_list, self.rels_opp = load_annotated_inter(movie_idxs=self.movie_idxs,
                                                                                               inter_class=opt.inter_class)
        else:
            self.rels, self.rels_list, self.rels_opp = {}, [], {}
            self.interactions = load_annotated_inter(movie_idxs=self.movie_idxs,
                                                     inter_class=opt.inter_class)

        # self.interactions = list([i for i in load_annotated_inter(movie_idxs=self.movie_idxs,
        #                                                           inter_class=opt.inter_class)])

        print('will be loaded %d feature vectors' % len(self.interactions))
        # load features
        self.features = {}
        movie_scene = set()
        self.idxs_with_triplets = []
        self.mv_sc_tr2triplidx = {}
        self.mv2sc2intersid = {}
        self.pair2scenes = defaultdict(Pair2Scene)
        for inter in tqdm(self.interactions):
            movie_idx = inter.video_descr['movie']
            scene_idx = inter.video_descr['scene'][0]
            if movie_idx not in self.mv2sc2intersid:
                self.mv2sc2intersid[movie_idx] = defaultdict(list)
            self.mv2sc2intersid[movie_idx][scene_idx].append(inter.id)
            if not self.tracks or len(inter.triplets) == 0:
                self.idxs_with_triplets.append((inter.id, 0))
                self.mv_sc_tr2triplidx[(movie_idx, scene_idx, 0)] = len(self.idxs_with_triplets) - 1
            else:
                for triplet_idx in inter.triplets:
                    self.idxs_with_triplets.append((inter.id, triplet_idx))
                    if len(inter.triplets[triplet_idx]) == 2:
                        name1, name2 =inter.triplets[triplet_idx][0], inter.triplets[triplet_idx][1]
                        sn = tuple((name1, name2))
                        self.pair2scenes[(movie_idx, sn[0], sn[1])].update(scene_idx, inter.id)
                        self.pair2scenes[(movie_idx, sn[1], sn[0])].update(scene_idx, inter.id)
                        if opt.rels_multi_clip:
                            if sn in self.rels[movie_idx] and scene_idx not in self.rels[movie_idx][sn].scenes:
                                if len(self.rels[movie_idx][sn].rel2scenes) == 1:
                                    self.rels[movie_idx][sn].append_scene(None, scene_idx)
                                    self.rels[movie_idx][(sn[1], sn[0])].append_scene(None, scene_idx)
                                    inter.relships[triplet_idx] = [self.rels[movie_idx][sn].rels_name]
                                else:
                                    min_dist = float('inf')
                                    update_rels_name = None
                                    for rels in self.rels[movie_idx][sn].rel2scenes:
                                        distance = min(abs(np.min(self.rels[movie_idx][sn].rel2scenes[rels]) - scene_idx),
                                                       abs(np.max(self.rels[movie_idx][sn].rel2scenes[rels]) - scene_idx))
                                        if distance < min_dist:
                                            update_rels_name = rels
                                    self.rels[movie_idx][sn].append_scene(update_rels_name, scene_idx)
                                    self.rels[movie_idx][(sn[1], sn[0])].append_scene(self.rels_opp[update_rels_name], scene_idx)
                                    inter.relships[triplet_idx] = [update_rels_name]

            if (movie_idx, scene_idx) not in movie_scene:
                features = MixedFeatures(video_idx=movie_idx,
                                         scene_idx=scene_idx,
                                         fname=inter.video_descr['fname'][0])
                self.features[(movie_idx, scene_idx)] = features
                movie_scene.add((movie_idx, scene_idx))
        self.iou2_clips = load_iou2_clips()
        self.rels2idx, self.idx2rels = {}, {}
        self.n_rels = 0

        assert len(self.idxs_with_triplets) >= len(self.interactions)

    def merged2inter(self):
        self.interidx2mgdidx = np.zeros(len(self.inter2idx), dtype=int)
        idx = 0 if opt.inter_class == 'all' else -1
        for inter, inter_idx in self.inter2idx.items():
            mgd = self.inter2mgd[inter]
            mgd_idx = self.mgd2idx[mgd]
            self.interidx2mgdidx[inter_idx[idx]] = mgd_idx

    def init_relships(self):
        # if rels_list is not None: self.rels_list = list(set(rels_list))
        self.rels_list = list(reversed(sorted(self.rels_list)))
        for idx, relship in enumerate(self.rels_list):
            self.rels2idx[relship] = idx
            self.idx2rels[idx] = relship
        if opt.rels:
            self.rels_hash = np.eye(len(self.rels_list))
            opt.rels_dim = len(self.rels_list)
            print('updated opt.rels_dim: %d' % opt.rels_dim)
        assert self.rels2idx['None'] == len(self.rels_list) - 1
        self.n_rels = len(self.rels_list)


    @timing
    def cache(self):
        print('cache features')
        max_n_tripl = 0
        distr_ints = defaultdict(int)
        distr_tracks = defaultdict(int)
        distr_tracks_empty = defaultdict(int)
        for idx, triplet_idx in tqdm(self.idxs_with_triplets):
            inter = self.interactions[idx]
            movie_idx = inter.video_descr['movie']
            scene_idx = inter.video_descr['scene'][0]
            if len(inter.triplets[triplet_idx]) == 2:
                if triplet_idx not in inter.relships:
                    name1, name2 = inter.triplets[triplet_idx][0], inter.triplets[triplet_idx][1]
                    if opt.rels_multi_clip:
                        if (name1, name2) in self.rels[movie_idx]:
                            if scene_idx in self.rels[movie_idx][(name1, name2)].scenes:
                                rels_name = self.rels[movie_idx][(name1, name2)]._scene2rel[scene_idx]
                                inter.relships[triplet_idx] = rels_name

            feature_cl = self.features[(movie_idx, scene_idx)]
            distr_ints[inter.inter_node['name']] += 1
            distr_tracks[len(inter.triplets[triplet_idx])] += 1
            # just to cache features
            feature_cl.get_features_by_time(inter.time_node, idx=idx)
            if opt.tracks and len(inter.triplets):
                full = 2
                for positional_idx, track_name in inter.triplets[triplet_idx].items():
                    track = feature_cl.get_features_by_track(inter.ftracks[track_name],
                                                             idx=(idx, track_name),
                                                             name=track_name)
                    if np.sum(track) == 0: full -= 1
                # # of people with ints, # of tracks exist
                distr_tracks_empty['#ch_%d.#tr(nonzero)_%d'%(len(inter.triplets[triplet_idx]), full)] += 1

            # multiplication by 2 to take into account both directions
            max_n_tripl += len(list(permutations(inter.id2names.values(), 2)))
            max_n_tripl += 2 * len(inter.id2names)
            self._max_n_tripl = max_n_tripl if max_n_tripl > self._max_n_tripl else self._max_n_tripl
            self._max_n_tripl = 20
        # free space after precomputing each feature
        for feature_cl in self.features.values():
            feature_cl.free()

        print('max # triplets per interaction is %d' % self._max_n_tripl)
        if opt.rels_multi_clip:
            # number of clips which would be considered for the relationship classification
            self.rels_n_clips = opt.rels_n_clips
            self.cache_relationships()

    def cache_relationships(self):
        self.movie_ch1_ch2_rel = {}
        self.movie_ch1_ch2_rel_inter = {}
        hash_idx = 0
        self.hashidx_rels = {}
        self.hashrels_idx = {}
        distr = defaultdict(int)
        self.context_idxs = {}
        cashed_pairs = set()
        for movie_id in self.rels:
            for pair in self.rels[movie_id]:
                for pair_rel, pair_scenes in self.rels[movie_id][pair].rel2scenes.items():
                    dict_key = (movie_id, pair[0], pair[1], pair_rel)
                    dict_val = None
                    inter_classes = []
                    hash_key = (movie_id, pair[0], pair[1], pair_rel)
                    cashed_pairs.add(pair)
                    if hash_key not in self.hashidx_rels:
                        self.hashidx_rels[hash_key] = hash_idx
                        self.hashrels_idx[hash_idx] = hash_key
                        hash_idx += 1
                    # for each scene where both ch are presented
                    for scene_id in pair_scenes:

                        clips = []
                        features_cl = self.features[(movie_id, scene_id)]
                        # for all interactions which are in this scene
                        for inter_id in self.mv2sc2intersid[movie_id][scene_id]:
                            inter = self.interactions[inter_id]
                            # check if both persons are use this interaction, if not -> do not take it
                            if pair[0] in inter.name2id and pair[1] in inter.name2id:
                                clips.append([inter_id, (inter_id, pair[0]), (inter_id, pair[1])])
                                inter_classes.append(self.mgd2idx[self.inter2mgd[inter.inter_node['name']]])
                                distr[pair_rel] += 1
                        mat = features_cl.create_ch1_ch2_rel_mat(clips)
                        if mat is not None:
                            dict_val = join_data(dict_val, mat, np.vstack)
                    self.movie_ch1_ch2_rel[dict_key] = dict_val
                    self.movie_ch1_ch2_rel_inter[dict_key] = np.array(inter_classes, dtype=int)
                    if self.mode != 'train':
                        if len(dict_val) > self.rels_n_clips:
                            self.context_idxs[dict_key] = list(range(0, len(dict_val), len(dict_val) // self.rels_n_clips))[:self.rels_n_clips]
                            assert len(self.context_idxs[dict_key]) == self.rels_n_clips


        self.check_rels()
        self.cache_None_rels(cashed_pairs)


    def cache_None_rels(self, cashed_pairs):
        self.movie_ch1_ch2_none = {}
        self.movie_ch1_ch2_none_inter = {}
        self.context_idxs_none = {}
        for key, val in self.pair2scenes.items():
            movie_idx, name1, name2 = key
            if (name1, name2) in cashed_pairs:
                continue
            dict_val = None
            inter_classes = []
            for pair_scene_id in val.scenes2inters:
                clips = []
                features_cl = self.features[(movie_idx, pair_scene_id)]
                for inter_id in val.scenes2inters[pair_scene_id]:
                    inter = self.interactions[inter_id]
                    clips.append([inter_id, (inter_id, name1), (inter_id, name2)])
                    inter_classes.append(self.mgd2idx[self.inter2mgd[inter.inter_node['name']]])
                mat = features_cl.create_ch1_ch2_rel_mat(clips)
                if mat is not None:
                    dict_val = join_data(dict_val, mat, np.vstack)
            self.movie_ch1_ch2_none[key] = dict_val
            self.movie_ch1_ch2_none_inter[key] = np.array(inter_classes, dtype=int)
            if self.mode != 'train':
                if len(dict_val) > self.rels_n_clips:
                    self.context_idxs_none[key] = list(range(0, len(dict_val), len(dict_val) // self.rels_n_clips))[:self.rels_n_clips]
                    assert len(self.context_idxs_none[key]) == self.rels_n_clips

    def check_rels(self):
        total_n = 0
        inters_without_rels = 0
        uncertanty = defaultdict(int)
        for idx_pair in range(len(self.idxs_with_triplets)):
            idx, triplet_idx = self.idxs_with_triplets[idx_pair]
            inter = self.interactions[idx]
            if len(inter.triplets[triplet_idx]) == 2:
                if triplet_idx in inter.relships:
                    uncertanty[len(inter.relships[triplet_idx])] += 1
                else:
                    uncertanty['None'] += 1
                total_n += 1
            else:
                inters_without_rels += 1


        for key, val in uncertanty.items():
            print('%s\t%d' % (str(key), val))
        print('%d inter #rels in total' % total_n)
        print('%d inters wihtout any rels' % inters_without_rels)

    def __len__(self):
        if self.test_rels_multi_clip:
            return len(self.hashidx_rels)
        return len(self.idxs_with_triplets)

    def __getitem__(self, idx_pair):
        output = {}
        if self.test_rels_multi_clip:
            movie_idx, name1, name2, rel_name = self.hashrels_idx[idx_pair]

            shape = self.movie_ch1_ch2_rel[self.hashrels_idx[idx_pair]].shape

            output['rels_label'] = self.rels2idx[rel_name]

            output['features'] = np.zeros((shape[0] + 1, shape[1]))
            output['features'][1:] = self.movie_ch1_ch2_rel[self.hashrels_idx[idx_pair]]
            output['rels_mask'] = np.ones((shape[0], 1), dtype=int)
            return output

        idx, triplet_idx = self.idxs_with_triplets[idx_pair]
        inter = self.interactions[idx]
        movie_idx = inter.video_descr['movie']
        scene_idx = inter.video_descr['scene'][0]
        feature_cl = self.features[(movie_idx, scene_idx)]

        # get id of the text based interaction
        if opt.inter_class == 'all':
            label = self.inter2idx[inter.inter_node['name']][0]  # global index among all interaction classes
            if opt.merged:
                label = self.interidx2mgdidx[label]
        else:
            label = self.inter2idx[inter.inter_node['name']][2]  # local index for one of the 3 classes v/t/m
            if opt.merged:
                label = self.interidx2mgdidx[label]
        output['labels'] = label

        mg_time_node = inter.time_node
        features = feature_cl.get_features_by_time(mg_time_node, idx=idx)

        if opt.tracks and len(inter.triplets):
            if opt.tr_maximize:
                # reserve space for all triples
                if opt.rels_multitask:
                    mem_features = np.zeros((self._max_n_tripl, self.rels_n_clips+1, opt.mlp_dim))
                    mem_counter = 0
                else:
                    mem_features = np.zeros((self._max_n_tripl, opt.mlp_dim))
                    mem_counter = 0
                    mem_features[:, :features.shape[1]] = np.tile(features, (self._max_n_tripl, 1))

            track_features = np.zeros((1, opt.track_dim * 2))
            triplet_name = ['', '']
            #  compute ground truth tracks for the interaction, it's index is always 0 in the triplets
            for positional_idx, track_name in inter.triplets[triplet_idx].items():
                if positional_idx == 0:
                    track_features[0, :opt.track_dim] = feature_cl.get_features_by_track(inter.ftracks[track_name],
                                                                                         idx=(idx, track_name),
                                                                                         name=track_name)
                    triplet_name[0] = track_name
                else:
                    triplet_name[1] = track_name
                    track_features[0, opt.track_dim:] = feature_cl.get_features_by_track(inter.ftracks[track_name],
                                                                                         idx=(idx, track_name),
                                                                                         name=track_name)

            if np.min(track_features) != 0 or np.max(track_features) != 0: just_zeros = False
            else: just_zeros = True
            output['just_zeros'] = just_zeros
            track_names = {}
            track_rels = {}
            for j in range(self._max_n_tripl):
                track_names[j] = ['','']
                track_rels[j] = ''
            track_names[0] = triplet_name

            if opt.rels_multitask:
                output['rels_label'] = self.rels2idx[inter.get_relship_by_id(triplet_idx)]
                if opt.rels_multi_clip:
                    # create context of the current features
                    rels_mask = np.zeros((self.rels_n_clips, 1), dtype=int)
                    if len(inter.triplets[triplet_idx]) == 2:
                        name1, name2 = inter.triplets[triplet_idx][0], inter.triplets[triplet_idx][1]
                        rel_name = self.idx2rels[output['rels_label']]
                        track_rels[0] = rel_name
                        if rel_name == 'None':
                            output['hash_rel'] = -1
                            dict_key = (movie_idx, name1, name2)

                            context_len = len(self.movie_ch1_ch2_none[dict_key])
                            context = np.zeros((self.rels_n_clips+1, features.shape[-1] + track_features.shape[-1]))
                            context_gt = np.zeros((self.rels_n_clips+1, 1), dtype=int)
                            if context_len == 0:
                                rels_mask[0] = 1
                                context[1] = np.hstack((features, track_features))
                                context_gt[1] = label
                            elif context_len <= self.rels_n_clips:
                                rels_mask[:context_len] = 1
                                context[1:context_len+1] = self.movie_ch1_ch2_none[dict_key]
                                context_gt[1:context_len+1] = self.movie_ch1_ch2_none_inter[dict_key].reshape(-1,1)
                            else:
                                if self.mode == 'train':
                                    context_idxs = np.random.choice(np.arange(context_len), self.rels_n_clips, replace=False)
                                else:
                                    context_idxs = self.context_idxs_none[dict_key]
                                context[1:] = self.movie_ch1_ch2_none[dict_key][context_idxs]
                                context_gt[1:] = self.movie_ch1_ch2_none_inter[dict_key][context_idxs].reshape(-1,1)
                                rels_mask[:] = 1
                        else:
                            output['hash_rel'] = self.hashidx_rels[(movie_idx, name1, name2, rel_name)]
                            dict_key = (movie_idx, name1, name2, rel_name)
                            context_len = len(self.movie_ch1_ch2_rel[dict_key])
                            context = np.zeros((self.rels_n_clips+1, features.shape[-1] + track_features.shape[-1]))
                            context_gt = np.zeros((self.rels_n_clips+1,1), dtype=int)
                            if context_len <= self.rels_n_clips:
                                rels_mask[:context_len] = 1
                                context[1:context_len+1] = self.movie_ch1_ch2_rel[dict_key]
                                context_gt[1:context_len+1] = self.movie_ch1_ch2_rel_inter[dict_key].reshape(-1, 1)
                            else:
                                if self.mode == 'train':
                                    context_idxs = np.random.choice(np.arange(context_len), self.rels_n_clips, replace=False)
                                else:
                                    context_idxs = self.context_idxs[dict_key]
                                context[1:] = self.movie_ch1_ch2_rel[dict_key][context_idxs]
                                context_gt[1:] = self.movie_ch1_ch2_rel_inter[dict_key][context_idxs].reshape(-1, 1)
                                rels_mask[:] = 1

                    else:
                        output['hash_rel'] = -1
                        context = np.tile(np.hstack((features, track_features)), (self.rels_n_clips+1, 1))
                        context_gt = np.ones((self.rels_n_clips+1, 1), dtype=int) * label
                        rels_mask[0] = 1
                    # put gt at the first position

                    context[0, :] = np.hstack((features, track_features))
                    context_gt[0] = label

            # put ground truth features
            if opt.tr_maximize:
                if opt.rels_multitask:
                    mem_features[0] = context
                    output['rels_mask'] = rels_mask

                else:
                    mem_features[0, features.shape[1]:] = track_features
                mem_counter = 1
            elif opt.rels_multi_clip:
                output['features'] = context
                output['labels'] = context_gt
                output['rels_mask'] = rels_mask
            else:
                output['features'] = np.hstack((features, track_features))

            if self.triplets:
                # if case of bidirectional interaction there can be other pair of track which could be correct
                gt_tracks = [0, 0]
                just_zeros = True
                if opt.rels_multitask:
                    rels_labs = np.zeros(self._max_n_tripl, dtype=int)
                    rels_labs[0] = output['rels_label']
                    rels_masks = np.zeros((self._max_n_tripl, self.rels_n_clips), dtype=int)
                    rels_masks[0] = output['rels_mask'].reshape(-1)
                # two persons participate in the interaction
                for name1, name2 in permutations(inter.id2names.values(), 2):
                    if len(inter.triplets[triplet_idx]) == 2:
                        if name1 == inter.triplets[triplet_idx][0] and name2 == inter.triplets[triplet_idx][1]: continue
                        if inter.bi and name1 == inter.triplets[triplet_idx][1] and name2 == inter.triplets[triplet_idx][0]:
                            if opt.tr_maximize:
                                gt_tracks[1] = mem_counter-1
                            else:
                                # what is this????
                                gt_tracks[1] = len(output['features'])
                    track_features = np.zeros((1, opt.track_dim * 2))
                    track_features[0, :opt.track_dim] = feature_cl.get_features_by_track(inter.ftracks[name1],
                                                                                         idx=(idx, name1),
                                                                                         name=name1)

                    track_features[0, opt.track_dim:] = feature_cl.get_features_by_track(inter.ftracks[name2],
                                                                                         idx=(idx, name2),
                                                                                         name=name2)
                    if np.min(track_features) != 0 or np.max(track_features) != 0: just_zeros = False


                    if opt.tr_maximize:
                        if mem_counter < self._max_n_tripl:
                            sorted_names = tuple([name1, name2])
                            if opt.rels_multitask:
                                # if the relationship between name1 and name2 exists, add it to the features
                                rels_mask = np.zeros((self.rels_n_clips, 1), dtype=int)
                                if sorted_names in self.rels[movie_idx]:
                                    rel_name = self.rels[movie_idx][sorted_names].scene2rel(scene_idx)
                                    if rel_name == 'None':
                                        context_tripl = np.tile(np.hstack((features, track_features)), (self.rels_n_clips+1, 1))
                                        rels_mask[0] = 1
                                    else:
                                        dict_key = (movie_idx, name1, name2, rel_name)
                                        context_len = len(self.movie_ch1_ch2_rel[dict_key])
                                        context_tripl = np.zeros((self.rels_n_clips + 1, features.shape[-1] + track_features.shape[-1]))
                                        if context_len <= self.rels_n_clips:
                                            rels_mask[:context_len] = 1
                                            context_tripl[1:context_len + 1] = self.movie_ch1_ch2_rel[dict_key]
                                        else:
                                            if self.mode == 'train':
                                                context_idxs = np.random.choice(np.arange(context_len),
                                                                                self.rels_n_clips, replace=False)
                                            else:
                                                context_idxs = self.context_idxs[dict_key]
                                            context_tripl[1:] = self.movie_ch1_ch2_rel[dict_key][context_idxs]
                                            rels_mask[:] = 1
                                else:
                                    rel_name = 'None'
                                    context_tripl = np.tile(np.hstack((features, track_features)), (self.rels_n_clips+1, 1))
                                    rels_mask[0] = 1

                                mem_features[mem_counter] = context_tripl
                                rels_labs[mem_counter] = self.rels2idx[rel_name]
                                rels_masks[mem_counter] = rels_mask.reshape(-1)
                                track_rels[mem_counter] = rel_name
                                track_names[mem_counter] = sorted_names
                            else:
                                mem_features[mem_counter, features.shape[1]:] = track_features

                            mem_counter += 1
                    else:
                        output['features'] = np.vstack((output['features'], np.hstack((features, track_features))))

                # one person participate in the interaction
                # features with ground truth person but changed direction of the interaction
                if len(inter.triplets[triplet_idx]) == 1:
                    track_features = np.zeros((1, opt.track_dim * 2))
                    position, gt_name = list(inter.triplets[triplet_idx].items())[0]
                    # put features to the wrong position
                    if position == 0:
                        track_features[0, opt.track_dim:] = feature_cl.get_features_by_track(inter.ftracks[gt_name],
                                                                                             idx=(idx, gt_name),
                                                                                             name=gt_name)
                    else:
                        track_features[0, :opt.track_dim] = feature_cl.get_features_by_track(inter.ftracks[gt_name],
                                                                                             idx=(idx, gt_name),
                                                                                             name=gt_name)
                    if np.min(track_features) != 0 or np.max(track_features) != 0: just_zeros = False
                    if opt.tr_maximize:
                        if mem_counter < self._max_n_tripl:
                            if inter.bi: gt_tracks[1] = mem_counter
                            if opt.rels_multitask:
                                rels_labs[mem_counter] = self.rels2idx['None']
                                context_tripl = np.tile(np.hstack((features, track_features)), (self.rels_n_clips+1, 1))
                                mem_features[mem_counter] = context_tripl
                                rels_masks[mem_counter, 0] = 1
                            else:
                                mem_features[mem_counter, features.shape[1]:] = track_features
                            mem_counter += 1
                    else:
                        if inter.bi: gt_tracks[1] = len(output['features'])
                        output['features'] = np.vstack((output['features'],
                                                        np.hstack((features, track_features))))

                # all other possible features where only one person track is appear for the interaction
                for name1 in inter.id2names.values():
                    if len(inter.triplets[triplet_idx]) == 1 and name1 == gt_name: continue
                    track_features1 = np.zeros((1, opt.track_dim * 2))
                    track_features2 = np.zeros((1, opt.track_dim * 2))
                    track_features1[0, :opt.track_dim] = feature_cl.get_features_by_track(inter.ftracks[name1],
                                                                                          idx=(idx, name1),
                                                                                          name=name1)
                    track_features2[0, opt.track_dim:] = feature_cl.get_features_by_track(inter.ftracks[name1],
                                                                                          idx=(idx, name1),
                                                                                          name=name1)
                    if np.min(track_features) != 0 or np.max(track_features) != 0: just_zeros = False
                    if opt.tr_maximize:
                        if mem_counter < self._max_n_tripl-1:
                            if opt.rels_multitask:
                                rels_labs[mem_counter] = self.rels2idx['None']
                                context_tripl = np.tile(np.hstack((features, track_features1)), (self.rels_n_clips+1, 1))
                                mem_features[mem_counter] = context_tripl
                                rels_masks[mem_counter, 0] = 1

                                rels_labs[mem_counter+1] = self.rels2idx['None']
                                context_tripl = np.tile(np.hstack((features, track_features2)), (self.rels_n_clips+1, 1))
                                mem_features[mem_counter+1] = context_tripl
                                rels_masks[mem_counter+1, 0] = 1
                            else:
                                mem_features[mem_counter, features.shape[1]:] = track_features1
                                mem_features[mem_counter+1, features.shape[1]:] = track_features2
                            mem_counter += 2
                    else:
                        output['features'] = np.vstack((output['features'],
                                                        np.hstack((features, track_features1)),
                                                        np.hstack((features, track_features2))))
                output['just_zeros'] = just_zeros
                output['gt_tracks'] = np.array(gt_tracks)
                output['n_names'] = len(inter.id2names)
                if opt.tr_maximize:
                    output['features'] = mem_features
                    mem_mask = np.zeros(self._max_n_tripl)
                    mem_mask[:mem_counter] = 1
                    output['mem_mask'] = mem_mask
                    if opt.rels_multitask:
                        output['rels_label'] = rels_labs
                        output['rels_mask'] = rels_masks
        elif opt.tracks:
            raise EnvironmentError
        else:
            output['features'] = features

        if opt.multilab_weights:
            multilab_weights = np.ones(self.n_classes)
            multilab_weights_axl = np.ones(len(self.interidx2mgdidx))
            for soft_name in self.iou2_clips[(movie_idx, scene_idx)][inter.inter_node['name']]:
                if opt.inter_class != 'all' and ['t', 'v', 'm'][self.inter2idx[soft_name][1]] != opt.inter_class:
                    continue
                inter_idx = self.inter2idx[soft_name][0 if opt.inter_class == 'all' else 2]
                multilab_weights_axl[inter_idx] = 0
                inter_idx = self.interidx2mgdidx[inter_idx]
                multilab_weights[inter_idx] = 0
            output['multilab_weights'] = multilab_weights
            output['multilab_weights_axl'] = multilab_weights_axl

        if opt.soft_gt:
            soft_labels, sf_idx = np.ones(self.n_classes) * -1, 1
            multilab_weights = np.ones(self.n_classes)
            soft_labels[0] = label
            for soft_name in self.iou2_clips[(movie_idx, scene_idx)][inter.inter_node['name']]:
                if opt.inter_class != 'all' and ['t', 'v', 'm'][self.inter2idx[soft_name][1]] != opt.inter_class:
                    continue
                inter_idx = self.inter2idx[soft_name][0 if opt.inter_class == 'all' else 2]
                inter_idx = self.interidx2mgdidx[inter_idx]
                soft_labels[sf_idx] = inter_idx
                sf_idx += 1
                multilab_weights[inter_idx] = 0
            output['soft_labels'] = soft_labels
        return output

    def _one_hot_encoding(self, position):
        label = np.zeros(self.n_classes)
        label[position] = 1
        return label

def f_dataloader(mode='train'):
    print('load mixed features. mode: %s' % mode)
    dataset = MixedFeaturesDataset(mode)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=(mode == 'train'),
                                             num_workers=opt.num_workers)
    return dataloader, dataset.n_classes

