#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Assemble a complete Morra NER model from it's parts.
"""
from morra import MorphParserNE

MODEL_FN = 'model_ne.pickle'

mp = MorphParserNE()
# If you already have a partially trained model, uncomment the line below
#mp.load(MODEL_FN)

mp._load_cdict('_cdict.pickle')
# comment partial models that you don't need
mp._load_ne_model('_model.ne.pickle')
mp._load_ne_rev_model('_model.ne_rev.pickle')
mp._load_ne2_model('_model.ne2.pickle')
mp._load_ne_models('_models.ne.pickle')
mp._load_ne_rev_models('_models.ne_rev.pickle')
mp._load_ne2_models('_models.ne2.pickle')
mp.save(MODEL_FN)
