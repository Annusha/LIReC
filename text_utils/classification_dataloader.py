#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'


from torch.utils.data import Dataset
from tqdm import tqdm

from utils.util_functions import *
from text_utils.text_features import TextFeatures


class TextFeaturesDataset(Dataset):
    def __init__(self, mode='train', html=False):
        print('load text features. mode: %s' % mode)
        if html:
            interactions, self.inter2idx, self.idx2inter = load_interaction_names(idx2inter_ret=True)
        else:
            interactions, self.inter2idx = load_interaction_names()
        self.n_classes = len(interactions[opt.inter_class])
        self.movie_idxs = load_set(mode=mode)
        self.html = html
        # to test on reduced amout of data
        # self.movie_idxs = self.movie_idxs[:len(self.movie_idxs)//4]
        # if not ops.isdir('/sequoia/data2/'):
        if opt.sanity_check:
            self.movie_idxs = 'tt1454029'
        node_types = ['interaction', 'summary']
        self.interactions = list([i for i in load_annotated_inter(self.movie_idxs,
                                                                  node_types,
                                                                  inter_class=opt.inter_class)])
        print('will be loaded %d feature vectors' % len(self.interactions))
        # load features
        self.features = {}
        movie_scene = set()
        for inter in tqdm(self.interactions):
            movie_idx = inter.video_descr['movie']
            scene_idx = inter.video_descr['scene'][0]
            if (movie_idx, scene_idx) not in movie_scene:
                text_feature = TextFeatures(video_idx=movie_idx,
                                            scene_idx=scene_idx,
                                            fname=inter.video_descr['fname'][0])
                self.features[(movie_idx, scene_idx)] = text_feature
                movie_scene.add((movie_idx, scene_idx))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        inter = self.interactions[idx]
        movie_idx = inter.video_descr['movie']
        scene_idx = inter.video_descr['scene'][0]
        features = self.features[(movie_idx, scene_idx)]

        if self.html:
            features, dialog = features.get_features_by_time(inter.time_node, html=True)
        else:
            features = features.get_features_by_time(inter.time_node)
        # get id of the text based interaction
        label = self.inter2idx[inter.inter_node['name']][2]  # local index for one of the 3 classes v/t/m
        if opt.pool_features in ['max', 'mix']:
            features = np.max(features, axis=0).reshape(1, -1)
        elif opt.pool_features == 'sum':
            features = np.sum(features, axis=0).reshape(1, -1)
        elif opt.pool_features == 'avg':
            features = np.mean(features, axis=0, keepdims=True)
        if self.html:
            meta = {'dialog': dialog,
                    'fname': '%s_%s' % (movie_idx, scene_idx)}
            return features, label, meta
        return features, label



def f_dataloader(mode='train', html=False):
    print('load text features. mode: %s' % mode)
    dataset = TextFeaturesDataset(mode, html=html)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=(mode == 'train'),
                                             num_workers=opt.num_workers)
    if html:
        return dataloader, dataset.n_classes, dataset.idx2inter
    return dataloader, dataset.n_classes


if __name__ == '__main__':
    test_loader = TextFeaturesDataset()
    test_elem = test_loader[1]
    print(0)
