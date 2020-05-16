# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Helpers for Morra parsers.
"""
import os
from pymorphy2 import MorphAnalyzer

from corpuscula.items import Items

POS_PROPN = 'PROPN'
MISC_NE = 'Entity'
MISC_NE_PERSON = 'Person'

names_db = os.environ.get('NAMES_DB') or 'names.pickle'
surnames_db = os.environ.get('SURNAMES_DB') or 'surnames.pickle'
_names = Items(restore_from=names_db)
_surnames = Items(restore_from=names_db)

def guess_ne (guess, coef, i, tokens, cdict):
    token = tokens[i]
    lemma, pos = token[1:3]
    # Person
    if pos == POS_PROPN:
        #if _names.item_isknown(lemma, 'patronym'):
        #    guess, coef = MISC_NE_PERSON, .5
        #elif _names.item_isknown(lemma, 'name'):
        #    guess, coef = MISC_NE_PERSON, 1.
        #elif _surnames.item_isknown(lemma, 'surname'):
        #    guess, coef = MISC_NE_PERSON, 1.
        pass
    return guess, coef

def fix_token ():
    pass
