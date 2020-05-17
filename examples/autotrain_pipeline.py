#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: A pipeline to train the Morra model with hyperparameters selection.
WARNING: It will work long. Especially, training joint FEATS models.
"""
import os
from corpuscula.corpus_utils import download_ud, UniversalDependencies, \
                                    AdjustedForSpeech
from morra import autotrain, MorphParser3

###
import sys
sys.path.append('../')
###
from corpuscula.utils import download_file
import _get_names_db
from scripts.local_methods_syntagrus import guess_pos, guess_lemma, guess_feat

MODEL_FN = 'model.pickle'
SEED=42

# we use UD Taiga corpus only as example. For real model training comment
# Taiga and uncomment SynTagRus
corpus_name = 'UD_Russian-Taiga'
#corpus_name = 'UD_Russian-SynTagRus'

download_ud(corpus_name, overwrite=False)
train_corpus = dev_corpus = test_corpus = UniversalDependencies(corpus_name)
#train_corpus = dev_corpus = test_corpus = \
#                         AdjustedForSpeech(UniversalDependencies(corpus_name))

def get_model(load_corpuses=True, load_model=True):
    mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                      guess_feat=guess_feat)
    if load_corpuses:
        mp.load_test_corpus(dev_corpus)
        mp.load_train_corpus(train_corpus)
    if load_model:
        mp.load(MODEL_FN)
    return mp

def reload_train_corpus():
    mp._train_corpus = None
    mp.load_train_corpus(train_corpus)

mp = get_model(load_model=False)
mp.parse_train_corpus()
mp._save_cdict('_cdict.pickle')
mp.save(MODEL_FN)

print()
mp = get_model()
mp.train_lemma(epochs=-3)
mp._save_lemma_model('_model.lemma.pickle')
mp.save(MODEL_FN)
print('== lemma ==')
mp.evaluate_lemma(test_corpus)

print()
mp = get_model()
autotrain(mp.train_pos,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          rev=False, seed=SEED, epochs=-3)
mp._save_pos_model('_model.pos.pickle')
mp.save(MODEL_FN)
print('== pos 1 ==')
mp.evaluate_pos(test_corpus, rev=False)

print()
mp = get_model()
autotrain(mp.train_pos,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          rev=True, seed=SEED, epochs=-3)
mp._save_pos_rev_model('_model.pos_rev.pickle')
mp.save(MODEL_FN)
print('== pos 1-rev ==')
mp.evaluate_pos(test_corpus, rev=True)

print()
mp = get_model()
autotrain(mp.train_pos2,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          seed=SEED, epochs=-3, test_max_repeats=0)
mp._save_pos2_model('_model.pos2.pickle')
mp.save(MODEL_FN)
print('== pos 2 ==')
mp.evaluate_pos2(test_corpus, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== pos 2:{} =='.format(_r))
    mp.evaluate_pos2(test_corpus, with_backoff=False, max_repeats=_r)

print()
mp = get_model()
feats = sorted(mp._cdict.get_feats())
results = {}
for feat in feats:
    score, kwargs, _ = \
        autotrain(mp.train_feats,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, rev=False, feat=feat, seed=SEED, epochs=-3)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
mp._save_feats_models('_models.feats.pickle')
mp.save(MODEL_FN)
print('== feats 1s ==')
mp.evaluate_feats(test_corpus, joint=False, rev=False)

print()
mp = get_model()
feats = sorted(mp._cdict.get_feats())
results = {}
for feat in feats:
    score, kwargs, _ = \
        autotrain(mp.train_feats,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, rev=True, feat=feat, seed=SEED, epochs=-3)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
mp._save_feats_rev_models('_models.feats_rev.pickle')
mp.save(MODEL_FN)
print('== feats 1s-rev ==')
mp.evaluate_feats(test_corpus, joint=False, rev=True)

print()
mp = get_model()
feats = sorted(mp._cdict.get_feats())
results = {}
for feat in feats:
    score, kwargs, _ = \
        autotrain(mp.train_feats2,
                  backup_model_func=mp.backup, restore_model_func=mp.restore,
                  reload_trainset_func=reload_train_corpus,
                  fit_params={'dropout': [None, .005, .01],
                              'context_dropout': [None, .05, .1]},
                  joint=False, seed=SEED, epochs=-3, test_max_repeats=0,
                  feat=feat)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
mp._save_feats2_models('_models_.feats2.pickle')
mp.save(MODEL_FN)
print('== feats 2s ==')
mp.evaluate_feats2(test_corpus, joint=False, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== feats 2s:{} =='.format(_r))
    mp.evaluate_feats2(test_corpus, joint=False, with_backoff=False,
                       max_repeats=_r)

print()
mp = get_model()
autotrain(mp.train_feats,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=False, seed=SEED, epochs=-3)
mp._save_feats_model('_model.feats.pickle')
mp.save(MODEL_FN)
print('== feats 1j ==')
mp.evaluate_feats(test_corpus, joint=True, rev=False)

print()
mp = get_model()
autotrain(mp.train_feats,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=True, seed=SEED, epochs=-3)
mp._save_feats_rev_model('_model.feats_rev.pickle')
mp.save(MODEL_FN)
print('== feats 1j-rev ==')
mp.evaluate_feats(test_corpus, joint=True, rev=True)

print()
mp = get_model()
autotrain(mp.train_feats2,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01]},
          joint=True, seed=SEED, epochs=-3, test_max_repeats=0)
mp._save_feats2_model('_model.feats2.pickle')
mp.save(MODEL_FN)
print('== feats 2j ==')
mp.evaluate_feats2(test_corpus, joint=True, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== feats 2j:{} =='.format(_r))
    mp.evaluate_feats2(test_corpus, joint=True, with_backoff=False,
                       max_repeats=_r)

print()
mp = get_model()
for max_s in [None, 0, 1, 2]:
    for max_j in [None, 0, 1, 2]:
        print('== feats 3:{}:{} =='.format('' if max_s is None else max_s,
                                           '' if max_j is None else max_j))
        mp.evaluate_feats3(test_corpus,
                           with_s_backoff=max_s is None, max_s_repeats=max_s,
                           with_j_backoff=max_j is None, max_j_repeats=max_j)
