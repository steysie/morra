#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Split corpus into 'train', 'dev' and 'test parts.
"""
from morra import MorphParser3

SEED=42

CORPUS_FN = 'corpus.conllu'
CORPUS_TRAIN_FN = 'corpus_train.conllu'
CORPUS_DEV_FN = 'corpus_dev.conllu'
CORPUS_TEST_FN = 'corpus_test.conllu'

MorphParser3.split_corpus(
    CORPUS_FN,
    split=[.8, .1, .1],
    save_split_to=[CORPUS_TRAIN_FN, CORPUS_DEV_FN, CORPUS_TEST_FN],
    seed=SEED
)
