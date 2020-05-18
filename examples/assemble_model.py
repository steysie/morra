#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Assemble a complete model from it's parts.
"""
from morra import MorphParser3

MODEL_FN = 'model.pickle'

mp = MorphParser3()
# If you already have a partially trained model, uncomment the line below
#mp.load(MODEL_FN)

mp._load_cdict('_cdict.pickle')
# comment partial models that you don't need
mp._load_pos_model('_model.pos.pickle')
mp._load_pos_rev_model('_model.pos_rev.pickle')
mp._load_pos2_model('_model.pos2.pickle')
mp._load_feats_models('_models.feats.pickle')
mp._load_feats_rev_models('_models.feats_rev.pickle')
mp._load_feats2_models('_models.feats2.pickle')
mp._load_feats_model('_model.feats.pickle')
mp._load_feats_rev_model('_model.feats_rev.pickle')
mp._load_feats2_model('_model.feats2.pickle')
mp.save(MODEL_FN)
