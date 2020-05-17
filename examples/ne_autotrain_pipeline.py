#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: A pipeline to train the Morra NER model with hyperparameters
selection. WARNING: It will work long enough.
"""
from morra import MorphParserNE

###
import sys
sys.path.append('../')
###
import _get_names_db
from scripts.ne_local_methods import guess_ne

MODEL_FN = 'model_ne.pickle'
SEED=42

# corpus must already be POS and LEMMA tagged and splitted into 'train',
# 'dev' and 'test' parts. NE tags should be placed in MISC field in 'NE'
# variable (e.g.: NE=Address)
train_corpus = 'corpus_train.conllu'
dev_corpus = 'corpus_dev.conllu'
test_corpus = 'corpus_test.conllu'

def get_model (load_corpuses=True, load_model=True):
    mp = MorphParserNE(guess_ne=guess_ne)
    if load_corpuses:
        mp.load_test_corpus(dev_corpus)
        mp.load_train_corpus(train_corpus)
    if load_model:
        mp.load(MODEL_FN)
    return mp

def reload_train_corpus ():
    mp._train_corpus = None
    mp.load_train_corpus(train_corpus)

mp = get_model(load_model=False)
mp.save(MODEL_FN)

print()
mp = get_model()
autotrain(mp.train_ne,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=False, seed=SEED, epochs=-3)
mp._save_ne_model('_model.ne.pickle')
mp.save(MODEL_FN)
print('== ne 1j ==')
mp.evaluate_ne(test_corpus, joint=True, rev=False)

print()
mp = get_model()
autotrain(mp.train_ne,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=True, seed=SEED, epochs=-3)
mp._save_ne_rev_model('_model.ne_rev.pickle')
mp.save(MODEL_FN)
print('== ne 1j-rev ==')
mp.evaluate_ne(test_corpus, joint=True, rev=True)

print()
mp = get_model()
autotrain(mp.train_ne2,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, seed=SEED, epochs=-3)
mp._save_ne2_model('_model.ne2.pickle')
mp.save(MODEL_FN)
print('== ne 2j ==')
mp.evaluate_ne2(test_corpus, joint=True, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== ne 2j:{} =='.format(_r))
    mp.evaluate_ne2(test_corpus, joint=True, with_backoff=False,
                    max_repeats=_r)

print()
mp = get_model()
ne_classes = sorted(x[0] for x in mp._ne_freq)
results = {}
for ne in ne_classes:
    score, kwargs, _ = \
        autotrain(mp.train_ne,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, rev=False, ne=ne, seed=SEED, epochs=-3)
    results[ne] = (score, kwargs)
for ne in ne_classes:
    print(ne + ':', results[ne][0], results[ne][1])
mp._save_ne_models('_models.ne.pickle')
mp.save(MODEL_FN)
print('== ne 1s ==')
mp.evaluate_ne(test_corpus, joint=False, rev=False)

print()
mp = get_model()
ne_classes = sorted(x[0] for x in mp._ne_freq)
results = {}
for ne in ne_classes:
    score, kwargs, _ = \
        autotrain(mp.train_ne,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, rev=True, ne=ne, seed=SEED, epochs=-3)
    results[ne] = (score, kwargs)
for ne in ne_classes:
    print(ne + ':', results[ne][0], results[ne][1])
mp._save_ne_rev_models('_models.ne.pickle')
mp.save(MODEL_FN)
print('== ne 1s-rev ==')
mp.evaluate_ne(test_corpus, joint=False, rev=True)

print()
mp = get_model()
ne_classes = sorted(x[0] for x in mp._ne_freq)
results = {}
for ne in ne_classes:
    score, kwargs, _ = \
        autotrain(mp.train_ne2,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, ne=ne, seed=SEED, epochs=-3)
    results[ne] = (score, kwargs)
for ne in ne_classes:
    print(ne + ':', results[ne][0], results[ne][1])
mp._save_ne2_models('_models.ne2.pickle')
mp.save(MODEL_FN)
print('== ne 2s ==')
mp.evaluate_ne2(test_corpus, joint=False, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== ne 2s:{} =='.format(_r))
    mp.evaluate_ne2(test_corpus, joint=False, with_backoff=False,
                    max_repeats=_r)

print()
mp = get_model()
for max_s in [None, 0, 1, 2]:
    for max_j in [None, 0, 1, 2]:
        print('== ne 3:{}:{} =='.format('' if max_s is None else max_s,
                                        '' if max_j is None else max_j))
        mp.evaluate_ne3(test_corpus,
                        with_s_backoff=max_s is not None, max_s_repeats=max_s,
                        with_j_backoff=max_s is not None, max_j_repeats=max_j)
