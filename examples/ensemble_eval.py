#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate ensemble of models. Here, we evaluate only POS-tagges.
"""
from corpuscula.corpus_utils import download_syntagrus, syntagrus, \
                                    AdjustedForSpeech
from morra import MorphParser3
from morra.morph_ensemble import MorphEnsemble
import os
import sys

###
import sys
sys.path.append('../')
###
from scripts.local_methods_syntagrus import guess_pos, guess_lemma, guess_feat

download_syntagrus(overwrite=False)
test_corpus = syntagrus
#test_corpus = AdjustedForSpeech(syntagrus)

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

me = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    # remove excess models for to save memory
    remove_feats(mp)
    if not me:
        me = MorphEnsemble(mp._cdict)
    #me.add(mp.predict_pos, rev=True)
    #me.add(mp.predict_pos, rev=False)
    me.add(mp.predict_pos2, with_backoff=True)
me.evaluate('UPOS', test_corpus)

me = None
for _r in [1, 0, -1]:
    for seed in SEEDS:
        mp = get_model(load_corpuses=False, load_model=True, seed=seed)
        remove_feats(mp)
        if not me:
            me = MorphEnsemble(mp._cdict)
        # add each model 3 times with different params:
        if _r < 0:
            me.add(mp.predict_pos2, with_backoff=True)
        else:
            me.add(mp.predict_pos2, with_backoff=False, max_repeats=_r)
me.evaluate('UPOS', test_corpus)
