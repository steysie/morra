#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate Morra NER models.
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

mp = MorphParserNE(guess_ne=guess_ne)
mp.load(MODEL_FN)

print()
print('== ne 1j ==')
mp.evaluate_ne(test_corpus, joint=True, rev=False)
print('== ne 1j-rev ==')
mp.evaluate_ne(test_corpus, joint=True, rev=True)
print('== ne 2j ==')
mp.evaluate_ne2(test_corpus, joint=True, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== ne 2j:{} =='.format(_r))
    mp.evaluate_ne2(test_corpus, joint=True, with_backoff=False,
                    max_repeats=_r)
print()
print('== ne 1s ==')
mp.evaluate_ne(test_corpus, joint=False, rev=False)
print('== ne 1s-rev ==')
mp.evaluate_ne(test_corpus, joint=False, rev=True)
print('== ne 2s ==')
mp.evaluate_ne2(test_corpus, joint=False, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== ne 2j:{} =='.format(_r))
    mp.evaluate_ne2(test_corpus, joint=False, with_backoff=False,
                    max_repeats=_r)
print()
for max_s in [None, 0, 1, 2]:
    for max_j in [None, 0, 1, 2]:
        print('== ne 3:{}:{} =='.format('' if max_s is None else max_s,
                                        '' if max_j is None else max_j))
        mp.evaluate_ne3(test_corpus,
                        with_s_backoff=max_s is None, max_s_repeats=max_s,
                        with_j_backoff=max_j is None, max_j_repeats=max_j)
