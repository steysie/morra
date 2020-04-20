# -*- coding: utf-8 -*-

import os
import re
from pymorphy2 import MorphAnalyzer
from pymorphy2.tagset import OpencorporaTag

from corpuscula.items import Items

POS_PROPN = 'PROPN'
POS_NOUN = 'NOUN'
POS_VERB = 'VERB'
FEATS_GENDER = 'Gender'
FEATS_GENDER_M = 'Masc'
FEATS_GENDER_F = 'Fem'
MISC_NE = 'Entity'
MISC_NE_PERSON = 'Person'

items_db = os.environ.get('ITEMS_DB')
_it = Items(restore_from=items_db or 'items.pickle')
re_initial = re.compile('^[A-ZЁА-Я]\.$')
ma_parse = MorphAnalyzer().parse

def guess_pos (guess, coef, i, tokens, cdict):
    wform = tokens[i][0]

    # Сложные случаи
    #if re.match('мыл[аи]?', wform):
    #    guess, coef = POS_VERB, .3

    # Инициалы
    if re_initial.match(wform) \
   and (i > 0 or (len(tokens) > 1 and tokens[1][0].istitle())):
        guess, coef = POS_PROPN, 1.

    else:
        lemma, _ = cdict.predict_lemma(wform, POS_PROPN)
        if _it.item_isknown(lemma, 'patronym'):
            guess, coef = POS_PROPN, 1.
        elif _it.item_isknown(lemma, 'name'):
            if guess is None:
                guess, coef = POS_PROPN, 1.
            elif guess == POS_PROPN and coef < 1.:
                coef = .8 * coef + .2
        elif _it.item_isknown(lemma, 'surname'):
            if guess == POS_PROPN and coef < 1.:
                coef = .8 * coef + .2
    return guess, coef

MA_POS = {'ADJ'  : ['ADJF', 'COMP'],  # 112 (107, 3)
          'ADP'  : ['PREP'],          # 4
          'ADV'  : ['ADVB'],          # 18
          'AUX'  : ['VERB'],          # 1
          'CCONJ': ['CONJ'],          # 1
          'DET'  : ['NPRO'],          # 0-
          'INTJ' : ['INTJ'],          # 1
          'NOUN' : ['NOUN'],          # 408
          'NUM'  : ['NUMR'],          # 5
          'PART' : ['PRCL'],          # 1
          'PRON' : ['NPRO'],          # 30
          'PROPN': ['NOUN'],          # -
          'PUNCT': None,
          'SCONJ': ['CONJ'],          # 0-
          'SYM'  : None,
          'VERB' : ['VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'],  # -
          'X'    : None}
def guess_lemma (guess, coef, i, tokens, cdict):
    isfirst = i == 0
    token = tokens[i]
    wform, pos = token[:2]
    if coef == 0 and pos in [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ',
        'INTJ', 'NOUN', 'NUM', 'PART', 'PRON'
    ]:
        ma_tags = MA_POS[pos]
        if ma_tags:
            ma_guess = ma_parse(wform)
            for guess_ in ma_guess:
                ma_tag = guess_.tag
                #if isinstance(ma_tag, OpencorporaTag) and ma_tag.POS in ma_tags:
                if ma_tag.POS in ma_tags:
                    guess = guess_.normal_form.replace('ё', 'е')
                    coef = .1
                    break
    elif pos in ['PROPN']:
        if coef == 0:
            if _it.item_isknown(wform, 'patronym'):
                guess, coef = wform, 1.
            elif _it.item_isknown(wform, 'name'):
                guess, coef = wform, 1.
            else:
                guess_, coef = cdict.predict_lemma(wform, 'NOUN',
                                                   isfirst=isfirst)
                if coef > 0:
                    guess = guess_
        if wform.isupper():
            guess = wform
        elif not isfirst and wform.istitle():
            guess = guess.title()
    return guess, coef

def guess_feat (guess, coef, i, feat, tokens, cdict):
    guess_, coef_ = None, None
    token = tokens[i]
    wform, lemma, pos = token[:3]
    if pos == POS_PROPN and feat == FEATS_GENDER:
        item = _it.get(item_class='patronym', item=lemma, copy=False) \
            or _it.get(item_class='name', item=lemma, copy=False)
            #or _it.get(item_class='surname', item=lemma, copy=False)
        if item:
            gender = item['gender']
            guess_, coef_ = FEATS_GENDER_M if gender == 'M' else \
                            FEATS_GENDER_F if gender == 'F' else \
                            None, 1.
    if guess_ is None:
        guess_, coef_ = guess, coef
    return guess_, coef_

def guess_ne (guess, coef, i, tokens, cdict):
    token = tokens[i]
    lemma, pos = token[1:3]
    # Person
    if pos == POS_PROPN:
        #if _it.item_isknown(lemma, 'patronym'):
        #    guess, coef = MISC_NE_PERSON, .5
        #elif _it.item_isknown(lemma, 'name'):
        #    guess, coef = MISC_NE_PERSON, 1.
        #elif _it.item_isknown(lemma, 'surname'):
        #    guess, coef = MISC_NE_PERSON, 1.
        pass
    return guess, coef

def fix_token ():
    pass
