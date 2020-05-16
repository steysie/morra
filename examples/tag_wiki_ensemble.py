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
from scripts.local_methods_syntagrus import guess_pos, guess_lemma, guess_feat

WIKI_IN_FN = 'wiki.conllu'
WIKI_OUT_FN = 'wiki.conllu'
# all models should be in MODEL_DIR directory
MODEL_DIR = 'ensemble'
# their names should be model.<seed>.pickle
MODEL_FN_TPL = 'model.{}.pickle'
SEEDS = [2, 4, 24, 42]

def get_model(seed):
    mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                      guess_feat=guess_feat)
    fn = os.path.join(MODEL_DIR, MODEL_FN_TPL.format(seed))
    mp.load(fn)
    return mp

def remove_pos(model):
    model._pos_model     = None
    model._pos_rev_model = None
    model._pos2_model    = None

def remove_feats(model):
    model._feats_model      = None
    model._feats_rev_model  = None
    model._feats2_model     = None
    model._feats_models     = {}
    model._feats_rev_models = {}
    model._feats2_models    = {}

me = mp = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    # remove excess models for to save memory
    remove_feats(mp)
    if not me:
        me = MorphEnsemble(mp._cdict)
    me.add(mp.predict_pos2, with_backoff=True)

mp.predict_lemma_sents(
    me.predict_sents(
        'UPOS',
        WIKI_IN_FN,
#        mp.load_conllu(WIKI_IN_FN, fix=True,
#                       adjust_for_speech=True, log_file=None),
    ),
    save_to=WIKI_OUT_FN
)
