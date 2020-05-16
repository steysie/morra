#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate conjoint models.
"""
from corpuscula.corpus_utils import download_ud, UniversalDependencies, \
                                    AdjustedForSpeech
from morra import MorphParser3

###
import sys
sys.path.append('../')
###
from scripts.local_methods_syntagrus import guess_pos, guess_lemma, guess_feat

MODEL_FN = 'model.pickle'

# we use UD Taiga corpus only as example. For real model training comment
# Taiga and uncomment SynTagRus
corpus_name = 'UD_Russian-Taiga'
#corpus_name = 'UD_Russian-SynTagRus'

download_ud(corpus_name, overwrite=False)                           
train_corpus = dev_corpus = test_corpus = UniversalDependencies(corpus_name)
#train_corpus = dev_corpus = test_corpus = \
#                         AdjustedForSpeech(UniversalDependencies(corpus_name))

mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                  guess_feat=guess_feat)
mp.load(MODEL_FN)

print()
print('== 1s ==')
mp.evaluate(test_corpus, pos_rev=False, feats_joint=False, feats_rev=False)
print('== 1s-rev ==')
mp.evaluate(test_corpus, pos_rev=True, feats_joint=False, feats_rev=True)
print('== 1j ==')
mp.evaluate(test_corpus, pos_rev=False, feats_joint=True, feats_rev=False)
print('== 1j-rev ==')
mp.evaluate(test_corpus, pos_rev=True, feats_joint=True, feats_rev=True)
print()
print('== 2s ==')
mp.evaluate2(test_corpus, pos_backoff=True,
             feats_joint=False, feats_backoff=True)
print('== 2s:0: ==')
mp.evaluate2(test_corpus, pos_backoff=False, pos_repeats=0,
             feats_joint=False, feats_backoff=True)
print('== 2s::0 ==')
mp.evaluate2(test_corpus, pos_backoff=True,
             feats_joint=False, feats_backoff=False, feats_repeats=0)
for _s in [0, 1, 2]:
    for _j in [0, 1, 2]:
        print('== 2s:{}:{} =='.format(_s, _j))
        mp.evaluate2(test_corpus, pos_backoff=False, pos_repeats=_s,
                     feats_joint=False, feats_backoff=False, feats_repeats=_j)
print()
print('== 2j ==')
mp.evaluate2(test_corpus, pos_backoff=True,
             feats_joint=True, feats_backoff=True)
print('== 2j:0: ==')
mp.evaluate2(test_corpus, pos_backoff=False, pos_repeats=0,
             feats_joint=True, feats_backoff=True)
print('== 2j::0 ==')
mp.evaluate2(test_corpus, pos_backoff=True,
             feats_joint=True, feats_backoff=False, feats_repeats=0)
for _s in [0, 1, 2]:
    for _j in [0, 1, 2]:
        print('== 2j:{}:{} =='.format(_s, _j))
        mp.evaluate2(test_corpus, pos_backoff=False, pos_repeats=_s,
                     feats_joint=True, feats_backoff=False, feats_repeats=_j)
print()
print('== 3 ==')
mp.evaluate3(test_corpus, pos_backoff=True,
             feats_s_backoff=True, feats_j_backoff=True)
print('== 3:0: ==')
mp.evaluate3(test_corpus, pos_backoff=False, pos_repeats=0,
             feats_s_backoff=True, feats_j_backoff=True)
print('== 3::0_0 ==')
mp.evaluate3(test_corpus, pos_backoff=True,
             feats_s_backoff=False, feats_s_repeats=0,
             feats_j_backoff=False, feats_j_repeats=0)

pos_repeats = 0
feats_s_repeats = 0
feats_j_repeats = 0
print('== 3:{}:{}_{} =='
          .format(pos_repeats, feats_s_repeats, feats_j_repeats))
mp.evaluate3(test_corpus, pos_backoff=False, pos_repeats=pos_repeats,
             feats_s_backoff=False, feats_s_repeats=feats_s_repeats,
             feats_j_backoff=False, feats_j_repeats=feats_j_repeats)
