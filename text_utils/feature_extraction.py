#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'May 2019'


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

import os
import os.path as ops
import sys
import re

from utils.arg_pars import opt
from utils.util_functions import dir_check
from text_utils import update_arg_pars

########################################
# load dialogs for the clips


def load_text():
    clip2dialog = {}
    # p_file = '/sequoia/data1/akukleva/projects/inter_recog/mixed_up_files'
    # names_mixed = []
    # with open(p_file, 'r') as f:
    #     for line in f:
    #         names_mixed.append(line[:-4])
    for root, dirs, files in os.walk(opt.dialogs_path):
        for filename in files:
            if not filename.endswith(opt.ext_dialog): continue
            text_name = clip_name(root, filename)
            # tt0119822 scene-170  -- three dots
            # tt0114924 scene-127  -- multi conversation
            # tt0106918 scene-298  -- super wierd behaviour

            # for name_mixed in names_mixed:
            #     movie_name, scene_name = name_mixed.split('_')
            #     if movie_name in root and 'scene-%s' % scene_name in filename:
            #         break
            # else:
            #     continue
            # if 'tt0307987' not in root: continue
            # if 'scene-119' not in filename: continue
            # try:
            #     os.remove(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.npy'))
            # except FileNotFoundError:
            #     pass

            # allows to extract features on several machines simultaneously
            if ops.exists(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.npy')) and \
                    ops.exists(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.token2idx')):
                continue
            with open(ops.join(root, filename), 'rb') as f:
                try:
                    binary_text = f.read()
                    text = binary_text.decode('unicode_escape')
                except UnicodeDecodeError:
                    print(ops.join(root, filename))
                    raise UnicodeDecodeError
            preprocessed_text = []
            for subtext in preprocess_file(text):
                try:
                    preprocessed_text.append(preprocess_text(subtext))
                except IndexError:
                    print(ops.join(root, filename))
                    raise IndexError
            # check if the file is already processed
            clip2dialog[text_name] = preprocessed_text
            print('%d %s' % (len(clip2dialog), text_name))
            #
            # if len(clip2dialog) == 10:
            #     return clip2dialog

    return clip2dialog


def preprocess_file(text: str):
    flag = False
    subtext = ['']
    text = text.strip().split('\n')
    for line in text:
        #
        if line == '' and flag:
            # even there is nothing the function return empty line and then
            # for bert model the sentense is [CLS] [SEP]
            # => number of timestamps will correspond to the number of processed sentences
            # which is specified by number of [CLS]
            if subtext[-1].strip().endswith('...'):
                subtext[-1] = re.sub(r'\.\.\.', ' ', subtext[-1].strip())
                flag = False
            else:
                yield subtext
                flag, subtext = False, ['']

        if flag:
            if line.startswith('-'):
                if not subtext[0]: subtext = []
                subtext.append(line)
            else:
                subtext[-1] += line + ' '
        # for each new time stamp
        if '-->' in line:
            flag = True
    yield subtext


def preprocess_text(subtext: list):
    ''' Parse dialogs to exclude redundant lines and put special tokens.
    :param subtext: raw dialogs from the file
    :return: (str) processed sentences with special tokens
    '''
    start = ['[CLS]']
    sep = ['[SEP]']

    def _erase_special_symbols(narration):
        narration = narration.strip()
        # Get rid of text in parentheses or square brackets
        narration = re.sub(r"\([^\)]+\)", "", narration)
        narration = re.sub(r"\[[^\]]+\]", "", narration)
        narration = re.sub(r"<i>", "", narration)
        narration = re.sub(r"</i>", "", narration)
        narration = re.sub('<.+?>', "", narration)
        return narration

    for idx, narration in enumerate(subtext):
        narration = _erase_special_symbols(narration)
        if narration.startswith('-'): narration = narration[1:]
        if narration == '': return ''
        narration = narration.split() + sep
        subtext[idx] = narration

    if len(subtext) <= 1:
        return [' '.join(start + subtext[0])]

    multi_conversation = []
    for i in range(len(subtext) - 1):
        marked_sentence = ' '.join(start + subtext[i] + subtext[i+1])
        multi_conversation.append(marked_sentence)
    return multi_conversation


def clip_name(root, filename):
    clip = re.search(r'(tt\d*)', root).group(1)
    scene = re.search(r'scene-(\d*)\.', filename).group(1)
    dir_check(ops.join(opt.text_path, clip))
    return clip + '_' + scene


########################################
# extract features

def bert_features(clip2dialog: dict):
    tokenizer = BertTokenizer.from_pretrained(opt.bert_model)
    model = BertModel.from_pretrained(opt.bert_model).eval()

    text_names = np.array(list(clip2dialog.keys()))
    np.random.shuffle(text_names)
    for text_idx, text_name in enumerate(text_names):
        file_text = clip2dialog[text_name]
        # if already processed
        if ops.exists(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.npy')) and \
                ops.exists(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.token2idx')):
            continue
        token2embedding = None
        total_token_idx = 0
        tokens = []
        for marked_text_idx, marked_text in enumerate(file_text):
            for sentense_idx, marked_sentense in enumerate(marked_text):
                tokenized_text = tokenizer.tokenize(marked_sentense)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids = [0] * (indexed_tokens.index(indexed_tokens[-1]) + 1)
                segments_ids += [1] * (len(tokenized_text) - len(segments_ids))

                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensor = torch.tensor([segments_ids])
                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor, segments_tensor)

                # define active len of text to write at the moment
                if sentense_idx == 0:  # write full text
                    text_len = len(tokenized_text)
                else:  # if more than 2 persons in one conversation
                    # then treat conversation by two sentanses each time and write
                    # only what hasn't been written yet
                    text_len = np.sum(segments_ids, dtype=int)

                # matrix where each row corresponds to concatenation of tokens
                # embeddings from all the layers
                subtoken2embedding = np.zeros((text_len, opt.text_dim * opt.text_layers))
                for token_idx, token in enumerate(tokenized_text[-text_len:]):
                    hidden_layers = []
                    layer_idx = token_idx + len(tokenized_text) - text_len

                    for layer in encoded_layers:
                        hidden_layers.append(layer[0][layer_idx].numpy())

                    subtoken2embedding[token_idx] = np.array(hidden_layers).flatten()
                if token2embedding is None:
                    token2embedding = subtoken2embedding
                else:
                    token2embedding = np.vstack((token2embedding, subtoken2embedding))

                if opt.save_model:
                    # save token <-> token_idx correspondences for the current text
                    mode = 'w' if marked_text_idx == 0 and sentense_idx == 0 else 'a'
                    with open(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.token2idx'), mode) as f:
                        for token_idx, token in enumerate(tokenized_text[-text_len:]):
                            f.write('%s %d\n' % (token, token_idx + total_token_idx))
                    extra_path = '/sequoia/data2/akukleva/moviegraph/features/bert/bert_base'
                    with open(ops.join(extra_path, text_name.split('_')[0], text_name + '.token2idx'), mode) as f:
                        for token_idx, token in enumerate(tokenized_text[-text_len:]):
                            f.write('%s %d\n' % (token, token_idx + total_token_idx))
                total_token_idx += text_len
                tokens += tokenized_text[-text_len:]

        if opt.save_model:
            # save matrix
            np.save(ops.join(opt.text_path, text_name.split('_')[0], text_name + '.npy'),
                    token2embedding)
        print('%d / %d  %s ' % (text_idx, len(text_names), text_name))


def pipeline():
    update_arg_pars.update()

    clip2dialog = load_text()
    bert_features(clip2dialog)


if __name__ == '__main__':
    pipeline()
