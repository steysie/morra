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
import _get_names_db
from scripts.ne_local_methods import guess_ne

MODEL_FN = 'model_ne.pickle'
test_corpus = 'corpus_test.conllu'
# set params you want to check
with_j_backoff=False
max_j_repeats = 0
with_s_backoff=False
max_s_repeats = 0

mp = MorphParserNE(guess_ne=guess_ne)
mp.load(MODEL_FN)

print()
print('== 3:{}:{} =='.format(max_j_repeats, max_s_repeats))
mp.evaluate_ne3(
    test_corpus,
    with_j_backoff=with_j_backoff, max_j_repeats=max_j_repeats,
    with_s_backoff=with_s_backoff, max_s_repeats=max_s_repeats
)
