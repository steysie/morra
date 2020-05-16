#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate one exact Morra model you want to check. Set its
parameters below.
"""
from corpuscula.corpus_utils import download_syntagrus, syntagrus, \
                                    AdjustedForSpeech
from morra import MorphParser3

###
import sys
sys.path.append('../')
###
from scripts.local_methods_syntagrus import guess_pos, guess_lemma, guess_feat

MODEL_FN = 'model.pickle'
# set params you want to check
pos_backoff=False
pos_repeats = 0
feats_s_backoff=False
feats_s_repeats = 0
feats_j_backoff=False
feats_j_repeats = 0

download_syntagrus(overwrite=False)
test_corpus = syntagrus
#test_corpus = AdjustedForSpeech(syntagrus)

mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                  guess_feat=guess_feat)
mp.load(MODEL_FN)

print()
print('== 3:{}:{}_{} =='
          .format(pos_repeats, feats_s_repeats, feats_j_repeats))
mp.evaluate3(test_corpus, pos_backoff=pos_backoff, pos_repeats=pos_repeats,
             feats_s_backoff=feats_s_backoff, feats_s_repeats=feats_s_repeats,
             feats_j_backoff=feats_j_backoff, feats_j_repeats=feats_j_repeats)
