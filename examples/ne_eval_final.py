#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate one exact Morra NER model you want to check. Set its
parameters below.
"""
from morra import MorphParserNE

###
import sys
sys.path.append('../')
###
from scripts.ne_local_methods import guess_ne

MODEL_FN = 'model_ne.pickle'
test_corpus = 'corpus_test.conllu'
# set params you want to check
with_j_backoff=False
with_j_repeats = 0
with_s_backoff=False
with_s_repeats = 0

mp = MorphParserNE(guess_ne=guess_ne)
mp.load(MODEL_FN)

print()
print('== 3:{}:{} =='.format(with_j_repeats, with_s_repeats))
mp.evaluate_ne3(
    test_corpus,
    feats_j_backoff=with_j_backoff, feats_j_repeats=with_j_repeats,
    feats_s_backoff=with_s_backoff, feats_s_repeats=with_s_repeats
)
