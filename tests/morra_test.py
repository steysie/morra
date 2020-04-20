#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import isclose
import filecmp
import os

###
import sys
sys.path.append('../')
###

from _tests._test_support import bprint, eprint, safe_run, check_res
from morra import MorphParser3

WORK_DIR = os.path.dirname(os.path.realpath(__file__))
NO_TRAIN_EVALS = True
train_path = os.path.join(WORK_DIR, 'test.conllu')
model_path = os.path.join(WORK_DIR, 'model.pickle')

def get_model ():
    mp = MorphParser3()
    mp.load(model_path)
    mp.load_train_corpus(train_path, test=.1, seed=42)
    mp.remove_rare_feats(abs_thresh=1000, rel_thresh=.1)
    return mp

def f ():
    mp = MorphParser3()
    mp.load_train_corpus(train_path, test=.1, seed=42)
    mp.remove_rare_feats(abs_thresh=1000, rel_thresh=.1)
    mp.parse_train_corpus()
    mp._save_cdict(os.path.join(WORK_DIR, '_model.cdict.pickle'))
    mp.save(model_path)
    return True
check_res(safe_run(f, 'Creating cdict'))

def f ():
    mp = get_model()
    mp.train_lemma(epochs=2)
    mp._save_lemma_model(os.path.join(WORK_DIR, '_model.lemma.pickle'))
    mp.save(model_path)
    res = mp.evaluate_lemma()
    return res > 0.95
check_res(safe_run(f, 'Testing LEMMA'))

def f ():
    mp = get_model()
    mp.train_pos(seed=42, epochs=2, no_train_evals=NO_TRAIN_EVALS)
    mp._save_pos_model(os.path.join(WORK_DIR, '_model.pos.pickle'))
    mp.save(model_path)
    res = mp.evaluate_pos()
    return res > 0.98
check_res(safe_run(f, 'Testing POS'))

def f ():
    mp = get_model()
    mp.train_pos(rev=True, seed=42, epochs=2, no_train_evals=NO_TRAIN_EVALS)
    mp._save_pos_rev_model(os.path.join(WORK_DIR, '_model.pos_rev.pickle'))
    mp.save(model_path)
    res = mp.evaluate_pos(rev=True)
    return res > 0.98
check_res(safe_run(f, 'Testing Reversed POS'))

def f ():
    mp = get_model()
    mp.train_pos2(seed=42, epochs=2, no_train_evals=NO_TRAIN_EVALS)
    mp._save_pos2_model(os.path.join(WORK_DIR, '_model.pos2.pickle'))
    mp.save(model_path)
    res = mp.evaluate_pos2()
    return res > 0.982
check_res(safe_run(f, 'Testing POS-2'))

def f ():
    mp = get_model()
    mp.train_feats(joint=False, rev=False, seed=42, epochs=2,
                   no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats_models(os.path.join(WORK_DIR, '_models.feats.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats(joint=False, rev=False)
    return res[0] > 0.84 and res[1] > 0.94
check_res(safe_run(f, 'Testing FEATS'))

def f ():
    mp = get_model()
    mp.train_feats(joint=False, rev=True, seed=42, epochs=2,
                   no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats_rev_models(os.path.join(WORK_DIR,
                                           '_models.feats_rev.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats(joint=False, rev=True)
    return res[0] > 0.85 and res[1] > 0.94
check_res(safe_run(f, 'Testing Reversed FEATS'))

def f ():
    mp = get_model()
    mp.train_feats2(joint=False, seed=42, epochs=2,
                    no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats2_models(os.path.join(WORK_DIR, '_models.feats2.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats2(joint=False)
    return res[0] > 0.86 and res[1] > 0.94
check_res(safe_run(f, 'Testing FEATS-2'))

def f ():
    mp = get_model()
    mp.train_feats(joint=True, rev=False, seed=42, epochs=2,
                   no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats_model(os.path.join(WORK_DIR, '_model.feats.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats(joint=True, rev=False)
    return res[0] > 0.87 and res[1] > 0.94
check_res(safe_run(f, 'Testing FEATS-j'))

def f ():
    mp = get_model()
    mp.train_feats(joint=True, rev=True, seed=42, epochs=2,
                   no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats_rev_model(os.path.join(WORK_DIR, '_model.feats_rev.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats(joint=True, rev=True)
    return res[0] > 0.87 and res[1] > 0.93
check_res(safe_run(f, 'Testing Reversed FEATS-j'))

def f ():
    mp = get_model()
    mp.train_feats2(joint=True, seed=42, epochs=2,
                    no_train_evals=NO_TRAIN_EVALS)
    mp._save_feats2_model(os.path.join(WORK_DIR, '_model.feats2.pickle'))
    mp.save(model_path)
    res = mp.evaluate_feats2(joint=True)
    return res[0] > 0.88 and res[1] > 0.94
check_res(safe_run(f, 'Testing FEATS-2-j'))

def _eval ():
    mp = get_model()
    print('== 1 ==')
    res_ = mp.evaluate()
    res = res_[0] > 0.87 and res_[1] > 0.94
    print('== 1+prev ==')
    res_ = mp.evaluate(pos_rev=True)
    res = res and res_[0] > 0.87 and res_[1] > 0.94
    print('== 1+frev ==')
    res_ = mp.evaluate(feats_rev=True)
    res = res and res_[0] > 0.88 and res_[1] > 0.94
    print('== 1+rev ==')
    res_ = mp.evaluate(pos_rev=True, feats_rev=True)
    res = res and res_[0] > 0.88 and res_[1] > 0.94
    print('== 2-s ==')
    res_ = mp.evaluate2()
    res = res and res_[0] > 0.88 and res_[1] > 0.94
    print('== 2-j ==')
    res_ = mp.evaluate2(feats_joint=True)
    res = res and res_[0] > 0.89 and res_[1] > 0.94
    print('== 3 ==')
    res_ = mp.evaluate3()
    res = res and res_[0] > 0.88 and res_[1] > 0.94
    print('== 3-- ==')
    res_ = mp.evaluate3(pos_backoff=False,
                        feats_s_backoff=False, feats_j_backoff=False)
    res = res and res_[0] > 0.88 and res_[1] > 0.94
    return res
check_res(safe_run(_eval, 'FINAL TESTS'))

def f ():
    mp = MorphParser3()
    mp.load_train_corpus(train_path, test=.1, seed=42)
    mp.remove_rare_feats(abs_thresh=1000, rel_thresh=.1)
    mp._load_cdict(os.path.join(WORK_DIR, '_model.cdict.pickle'))
    mp._load_pos_model(os.path.join(WORK_DIR, '_model.pos.pickle'))
    mp._load_pos_rev_model(os.path.join(WORK_DIR, '_model.pos_rev.pickle'))
    mp._load_pos2_model(os.path.join(WORK_DIR, '_model.pos2.pickle'))
    mp._load_lemma_model(os.path.join(WORK_DIR, '_model.lemma.pickle'))
    mp._load_feats_model(os.path.join(WORK_DIR, '_model.feats.pickle'))
    mp._load_feats_rev_model(os.path.join(WORK_DIR, '_model.feats_rev.pickle'))
    mp._load_feats2_model(os.path.join(WORK_DIR, '_model.feats2.pickle'))
    mp._load_feats_models(os.path.join(WORK_DIR, '_models.feats.pickle'))
    mp._load_feats_rev_models(os.path.join(WORK_DIR, '_models.feats_rev.pickle'))
    mp._load_feats2_models(os.path.join(WORK_DIR, '_models.feats2.pickle'))
    mp.save(model_path)
    return True
check_res(safe_run(f, 'Recreate model'))

check_res(safe_run(_eval, 'Repeat FINAL TESTS'))

def f ():
    fname = os.path.join(WORK_DIR, 'test$.conllu')
    mp = get_model()
    mp.predict3_sents(pos_backoff=False,
                      feats_s_backoff=False, feats_j_backoff=False,
                      inplace=True, save_to=fname)
    res_ = mp.evaluate3(test=fname, pos_backoff=False,
                        feats_s_backoff=False, feats_j_backoff=False)
    res = res_[0] > 0.88 and res_[1] > 0.94
    res = res and filecmp.cmp(fname, os.path.join(WORK_DIR, 'test1.conllu'))
    if res:
        os.remove(fname)
    return res
check_res(safe_run(f, 'Final FINAL TESTS'))

os.remove(os.path.join(WORK_DIR, '_model.cdict.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.pos.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.pos_rev.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.pos2.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.lemma.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.feats.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.feats_rev.pickle'))
os.remove(os.path.join(WORK_DIR, '_model.feats2.pickle'))
os.remove(os.path.join(WORK_DIR, '_models.feats.pickle'))
os.remove(os.path.join(WORK_DIR, '_models.feats_rev.pickle'))
os.remove(os.path.join(WORK_DIR, '_models.feats2.pickle'))
os.remove(model_path)
