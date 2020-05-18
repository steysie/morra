#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Disassemble a complete Morra NER model to it's parts.
"""
from morra import MorphParserNE

MODEL_FN = 'model_ne.pickle'

mp = MorphParserNE()
mp.load(MODEL_FN)

if mp._ne_model:
    mp._save_ne_model('_model.ne.pickle')
if mp._ne_rev_model:
    mp._save_ne_rev_model('_model.ne_rev.pickle')
if mp._ne_model:
    mp._save_ne2_model('_model.ne2.pickle')
if mp._ne_models:
    mp._save_ne_models('_models.ne.pickle')
if mp._ne_rev_models:
    mp._save_ne_rev_models('_models.ne_rev.pickle')
if mp._ne2_models:
    mp._save_ne2_models('_models.ne2.pickle')
