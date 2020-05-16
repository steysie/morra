#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Tag a corpus with NE tagger. The corpus should be already tagged with
POS and LEMMA taggers. If not, uncomment corresponding lines below.
"""
from morra import MorphParser3, MorphParserNE

###
import sys
sys.path.append('../')
###
from local_methods import guess_pos, guess_lemma, guess_ne

CORPUS_IN_FN = 'corpus.conllu'
CORPUS_OUT_FN = 'corpus_tagged_ner.conllu'
MODEL_FN = 'model.pickle'
MODEL_NE_FN = 'model_ne.pickle'

corpus = CORPUS_IN_FN

# uncomment lines below if you need POS and LEMMA tagging
'''
mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma)
mp.load(MODEL_FN)
# set parameters you need
corpus = mp.predict_pos2_sents(corpus, with_backoff=False, max_repeats=1)
corpus = mp.predict_lemma_sents(corpus)
'''

mp = MorphParserNE(guess_ne=guess_ne)
mp.load(MODEL_FN)
# set parameters you need
mp.predict_ne3_sents(corpus,
                     with_s_backoff=False, max_s_repeats=0,
                     with_j_backoff=False, max_j_repeats=1,
                     save_to=CORPUS_OUT_FN)
