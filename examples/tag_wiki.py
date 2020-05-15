#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Tag Wikipedia with UPOS and LEMMA fields. You need Wikipedia already
tokenized and saved in CONLL-U format. You could do it with
examples/tokenize_wiki.py from our Toxine project
(https://github.com/fostroll/toxine).
"""
from morra import MorphParser3

###
import sys
sys.path.append('../')
###
from scripts.local_methods import guess_pos, guess_lemma, guess_feat

WIKI_IN_FN = 'wiki.conllu'
MODEL_FN = 'model.pickle'
WIKI_OUT_FN = 'wiki.conllu'

mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                  guess_feat=guess_feat)
mp.load(MODEL_FN)
mp.predict_lemma_sents(
    mp.predict_pos2_sents(
        WIKI_IN_FN,
#        mp.load_conllu(WIKI_IN_FN, fix=True,
#                       adjust_for_speech=True, log_file=None),
        with_backoff=False, max_repeats=1),
    save_to=WIKI_OUT_FN
)
