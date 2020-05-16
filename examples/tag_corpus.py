#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Tag a corpus.
"""
from morra import MorphParser3

###
import sys
sys.path.append('../')
###
from local_methods import guess_pos, guess_lemma, guess_feats

CORPUS_IN_FN = 'corpus.conllu'
CORPUS_OUT_FN = 'corpus_tagged.conllu'
MODEL_FN = 'model.pickle'

corpus = CORPUS_IN_FN

mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                  guess_feats=guess_feats)
mp.load(MODEL_FN)
# set parameters you need
mp.predict3_sents(corpus, pos_backoff=False, pos_repeats=0,
                  feats_s_backoff=False, feats_s_repeats=0,
                  feats_j_backoff=False, feats_j_repeats=0,
                  save_to=CORPUS_OUT_FN)

# set parameters you need
#corpus = mp.predict_pos2_sents(corpus, with_backoff=False, max_repeats=0)
#corpus = mp.predict_lemma_sents(corpus)
#corpus = mp.predict_feats3_sents(corpus,
#                                 with_s_backoff=False, max_s_repeats=0,
#                                 with_j_backoff=False, max_j_repeats=0,
#                                 save_to=CORPUS_OUT_FN)
