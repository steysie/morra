#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

from corpuscula.corpus_utils import download_syntagrus, syntagrus
from local_methods import guess_pos, guess_lemma, guess_feat
from morra import MorphParser3, autotrain

download_syntagrus(overwrite=False)
train_corpus = dev_corpus = test_corpus = syntagrus

MODEL_DIR = '___ensemble'
def get_model (load_corpuses=True, load_model=True, seed=None):
    mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                      guess_feat=guess_feat)
    if load_corpuses:
        mp.load_test_corpus(dev_corpus)
        mp.load_train_corpus(train_corpus)
    if load_model:
        fn = os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed))
        mp.load(fn)
    return mp

def reload_train_corpus ():
    mp._train_corpus = None
    mp.load_train_corpus(train_corpus)    
'''
mp = get_model(load_model=False)
mp.parse_train_corpus()
mp._save_cdict('_model_.cdict.pickle')
mp.save('model_.pickle')

mp = get_model()
mp.train_lemma(epochs=-3)
#mp._save_lemma_model('_model_.lemma.pickle')
mp.save('model_.pickle')
mp.evaluate_lemma(test_corpus)
'''
SEEDS = [4, 21, 42, 333, 777]
'''
for seed in SEEDS:
    print('\nseed =', seed, file=sys.stderr)
    mp = get_model(seed=seed)
    #autotrain(mp.train_pos,
    #          backup_model_func=mp.backup, restore_model_func=mp.restore,
    #          reload_trainset_func=reload_train_corpus,
    #          fit_params={'dropout': [None, .005, .01],
    #                      'context_dropout': [None, .05, .1]},
    #          rev=False, seed=seed, epochs=-3)
    mp.train_pos(rev=False, seed=seed, epochs=-3)
    mp._save_pos_model(os.path.join(MODEL_DIR, '_model_.pos{}.pickle'.format(seed)))
    mp.save(os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed)))
    mp.evaluate_pos(test_corpus)

for seed in SEEDS:
    print('\nseed =', seed, file=sys.stderr)
    mp = get_model(seed=seed)
    #autotrain(mp.train_pos,
    #          backup_model_func=mp.backup, restore_model_func=mp.restore,
    #          reload_trainset_func=reload_train_corpus,
    #          fit_params={'dropout': [None, .005, .01],
    #                      'context_dropout': [None, .05, .1]},
    #          rev=True, seed=seed, epochs=-3)
    mp.train_pos(rev=True, seed=seed, epochs=-3)
    mp._save_pos_model(os.path.join(MODEL_DIR, '_model_.pos_rev{}.pickle'.format(seed)))
    mp.save(os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed)))
    mp.evaluate_pos(test_corpus, rev=True)

for seed in SEEDS:
    print('\nseed =', seed, file=sys.stderr)
    mp = get_model(seed=seed)
    #autotrain(mp.train_pos2,
    #          backup_model_func=mp.backup, restore_model_func=mp.restore,
    #          reload_trainset_func=reload_train_corpus,
    #          fit_params={'dropout': [None, .005, .01],
    #                      'context_dropout': [None, .05, .1]},
    #          seed=seed, epochs=-3)
    mp.train_pos2(seed=seed, epochs=-3)
    mp._save_pos2_model(os.path.join(MODEL_DIR, '_model_.pos2{}.pickle'.format(seed)))
    mp.save(os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed)))
    mp.evaluate_pos2(test_corpus)
    mp.evaluate_pos2(test_corpus, with_backoff=False)
'''
'''
for seed in SEEDS:
    mp = get_model(seed=seed)
    #mp._load_pos_model(os.path.join(MODEL_DIR, '_model_.pos{}.pickle'.format(seed)))
    #mp.save(os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed)))
    mp.evaluate_pos(test_corpus)
for seed in SEEDS:
    mp = get_model(seed=seed)
    #mp._load_pos_model(os.path.join(MODEL_DIR, '_model_.pos_rev{}.pickle'.format(seed)))
    #mp.save(os.path.join(MODEL_DIR, 'model_.{}.pickle'.format(seed)))
    mp.evaluate_pos(test_corpus, rev=True)
'''
mp = get_model(load_corpuses=False, load_model=True, seed=4)
mp.evaluate_pos2(test_corpus, with_backoff=False)
'''
from morra.morph_ensemble import MorphEnsemble
me = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    if not me:
        me = MorphEnsemble(mp._cdict)
    me.add(mp.predict_pos2)
    #me.add(mp.predict_pos, rev=True)
    #me.add(mp.predict_pos, rev=False)
me.evaluate('UPOS', test_corpus)
me = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    if not me:
        me = MorphEnsemble(mp._cdict)
    me.add(mp.predict_pos2, with_backoff=False)
me.evaluate('UPOS', test_corpus)
me = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    if not me:
        me = MorphEnsemble(mp._cdict)
    me.add(mp.predict_pos2)
    me.add(mp.predict_pos2, with_backoff=False)
me.evaluate('UPOS', test_corpus)
me = None
for seed in SEEDS:
    mp = get_model(load_corpuses=False, load_model=True, seed=seed)
    if not me:
        me = MorphEnsemble(mp._cdict)
    me.add(mp.predict_pos2, with_backoff=False)
    me.add(mp.predict_pos2)
me.evaluate('UPOS', test_corpus)
'''
'''
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
                  joint=False, rev=False, feat=feat, seed=21, epochs=-3)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
#mp.train_feats(joint=False, rev=False, seed=21, epochs=-3)
mp._save_feats_models('_models_.feats.pickle')
mp.save('model_.pickle')
mp.evaluate_feats(test_corpus, joint=False, rev=False)

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
                  joint=False, rev=True, feat=feat, seed=21, epochs=-3)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
#mp.train_feats(joint=False, rev=True, seed=21, epochs=-3)
mp._save_feats_rev_models('_models_.feats_rev.pickle')
mp.save('model_.pickle')
mp.evaluate_feats(test_corpus, joint=False, rev=True)

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
                  joint=False, feat=feat, seed=21, epochs=-3)
    results[feat] = (score, kwargs)
for feat in feats:
    print(feat + ':', results[feat][0], results[feat][1])
#mp.train_feats2(joint=False, seed=21, epochs=-3)
mp._save_feats2_models('_models_.feats2.pickle')
mp.save('model_.pickle')
mp.evaluate_feats2(test_corpus, joint=False)
mp.evaluate_feats2(test_corpus, joint=False, with_backoff=False)

mp = get_model()
autotrain(mp.train_feats,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=False, seed=21, epochs=-3)
#mp.train_feats(joint=True, rev=False, seed=21, epochs=-3)
mp._save_feats_model('_model_.feats.pickle')
mp.save('model_.pickle')
mp.evaluate_feats(test_corpus, joint=True, rev=False)

mp = get_model()
autotrain(mp.train_feats,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01],
                      'context_dropout': [None, .05, .1]},
          joint=True, rev=True, seed=21, epochs=-3)
#mp.train_feats(joint=True, rev=True, seed=21, epochs=-3)
mp._save_feats_rev_model('_model_.feats_rev.pickle')
mp.save('model_.pickle')
mp.evaluate_feats(test_corpus, joint=True, rev=True)

mp = get_model()
autotrain(mp.train_feats2,
          backup_model_func=mp.backup, restore_model_func=mp.restore,
          reload_trainset_func=reload_train_corpus,
          fit_params={'dropout': [None, .005, .01]},
          joint=True, seed=21, epochs=-3)
#mp.train_feats2(joint=True, seed=21, epochs=-3)
mp._save_feats2_model('_model_.feats2.pickle')
mp.save('model_.pickle')
mp.evaluate_feats2(test_corpus, joint=True)
mp.evaluate_feats2(test_corpus, joint=True, with_backoff=False)
print()
mp.evaluate_feats3(test_corpus)
mp.evaluate_feats3(test_corpus, with_s_backoff=False, with_j_backoff=False)

print()
print()
print('FINAL TESTS')
mp = get_model(load_corpuses=False)
print()
print('== lemma ==')
mp.evaluate_lemma(test_corpus)
print()
print('== pos 1 ==')
mp.evaluate_pos(test_corpus)
print('== pos 1-rev ==')
mp.evaluate_pos(test_corpus, rev=True)
print('== pos 2 ==')
mp.evaluate_pos2(test_corpus)
print('== pos 2- ==')
mp.evaluate_pos2(test_corpus, with_backoff=False)
print()
print('== feats 1s ==')
mp.evaluate_feats(test_corpus, joint=False, rev=False)
print('== feats 1s-rev ==')
mp.evaluate_feats(test_corpus, joint=False, rev=True)
print('== feats 2s ==')
mp.evaluate_feats2(test_corpus, joint=False)
print('== feats 2s- ==')
mp.evaluate_feats2(test_corpus, joint=False, with_backoff=False)
print()
print('== feats 1j ==')
mp.evaluate_feats(test_corpus, joint=True, rev=False)
print('== feats 1j-rev ==')
mp.evaluate_feats(test_corpus, joint=True, rev=True)
print('== feats 2j ==')
mp.evaluate_feats2(test_corpus, joint=True)
print('== feats 2j- ==')
mp.evaluate_feats2(test_corpus, joint=True, with_backoff=False)
print()
print('== feats 3 ==')
mp.evaluate_feats3(test_corpus)
print('== feats 3- ==')
mp.evaluate_feats3(test_corpus, with_s_backoff=False, with_j_backoff=False)
print()
print('== 1s ==')
mp.evaluate(test_corpus)
print('== 1s-rev ==')
mp.evaluate(test_corpus, pos_rev=True, feats_rev=True)
print('== 1j ==')
mp.evaluate(test_corpus, feats_joint=True)
print('== 1j-rev ==')
mp.evaluate(test_corpus, pos_rev=True, feats_joint=True, feats_rev=True)
print()
print('== 2s ==')
mp.evaluate2(test_corpus, feats_joint=False)
print('== 2s-p ==')
mp.evaluate2(test_corpus, pos_backoff=False, feats_joint=False)
print('== 2s-f ==')
mp.evaluate2(test_corpus, feats_joint=False, feats_backoff=False)
print('== 2s-pf ==')
mp.evaluate2(test_corpus, pos_backoff=False, feats_joint=False, feats_backoff=False)
print()
print('== 2j ==')
mp.evaluate2(test_corpus, feats_joint=True)
print('== 2j-p ==')
mp.evaluate2(test_corpus, pos_backoff=False, feats_joint=True)
print('== 2j-f ==')
mp.evaluate2(test_corpus, feats_joint=True, feats_backoff=False)
print('== 2j-pf ==')
mp.evaluate2(test_corpus, pos_backoff=False, feats_joint=True, feats_backoff=False)
print()
print('== 3 ==')
mp.evaluate3(test_corpus)
print('== 3-p ==')
mp.evaluate3(test_corpus, pos_backoff=False)
print('== 3-f ==')
mp.evaluate3(test_corpus, feats_s_backoff=False, feats_j_backoff=False)
print('== 3-pf ==')
mp.evaluate3(test_corpus, pos_backoff=False, feats_s_backoff=False, feats_j_backoff=False)
'''
