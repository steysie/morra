#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Disassemble a complete model to it's parts.
"""
from morra import MorphParser3

MODEL_FN = 'model.pickle'

mp = MorphParser3()
mp.load(MODEL_FN)

if mp._cdict:
    mp._save_cdict('_cdict.pickle')
if mp._lemma_model:
    mp._save_lemma_model('_model.lemma.pickle')
if mp._pos_model:
    mp._save_pos_model('_model.pos.pickle')
if mp._pos_rev_model:
    mp._save_pos_rev_model('_model.pos_rev.pickle')
if mp._pos2_model:
    mp._save_pos2_model('_model.pos2.pickle')
if mp._feats_models:
    mp._save_feats_models('_models.feats.pickle')
if mp._feats_rev_models:
    mp._save_feats_rev_models('_models.feats_rev.pickle')
if mp._feats2_models:
    mp._save_feats2_models('_models.feats2.pickle')
if mp._feats_model:
   mp._save_feats_model('_model.feats.pickle')
if mp._feats_rev_model:
   mp._save_feats_rev_model('_model.feats_rev.pickle')
if mp._feats2_model:
   mp._save_feats2_model('_model.feats2.pickle')
