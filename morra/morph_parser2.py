# -*- coding: utf-8 -*-
# Morra project: Morphological parser 2
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Get the results of forward and backward parsers and make refining parse on the
ground of them.
"""
from collections import OrderedDict
from copy import deepcopy
import pickle
import random
from random import randint, random as rand
import sys

from corpuscula.utils import LOG_FILE, print_progress
from morra.base_parser import _AveragedPerceptron
from morra.features2 import Features2
from morra.morph_parser import MorphParser


class MorphParser2(MorphParser):

    def __init__(self, features='RU',
                 guess_pos=None, guess_lemma=None, guess_feat=None):
        super().__init__(
            guess_pos=guess_pos, guess_lemma=guess_lemma, guess_feat=guess_feat
        )
        self.features = Features2(lang=features) \
            if isinstance(features, str) else features

        self._pos2_model    = None
        self._feats2_model  = None
        self._feats2_models = {}

    def backup(self):
        """Get current state"""
        o = super().backup()
        o.update({'pos2_model_weights'   : self._pos2_model.weights
                                               if self._pos2_model   else
                                           None,
                  'feats2_model_weights' : self._feats2_model.weights
                                               if self._feats2_model else
                                           None,
                  'feats2_models_weights': {
                      x: y.weights for x, y in self._feats2_models.items()
                  }})
        return o

    def restore(self, o):
        """Restore current state from backup object"""
        super().restore(o)
        (pos2_model_weights   ,
         feats2_model_weights ,
         feats2_models_weights) = [o.get(x) for x in ['pos2_model_weights'   ,
                                                      'feats2_model_weights' ,
                                                      'feats2_models_weights']]
        if pos2_model_weights:
            self._pos2_model = _AveragedPerceptron()
            self._pos2_model.weights = pos2_model_weights
        else:
            self._pos2_model = None
        if feats2_model_weights:
            self._feats2_model = _AveragedPerceptron()
            self._feats2_model.weights = feats2_model_weights
        else:
            self._feats2_model = None
        self._feats2_models = {}
        if feats2_models_weights:
            for feat, weights in feats2_models_weights.items():
                model = self._feats2_models[feat] = _AveragedPerceptron()
                model.weights = weights

    def _save_pos2_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._pos2_model.weights if self._pos2_model else
                        None, f, 2)

    def _load_pos2_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._pos2_model = _AveragedPerceptron()
                self._pos2_model.weights = weights
            else:
                self._pos2_model = None

    def _save_feats2_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._feats2_model.weights if self._feats2_model else
                        None, f, 2)

    def _load_feats2_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._feats2_model = _AveragedPerceptron()
                self._feats2_model.weights = weights
            else:
                self._feats2_model = None

    def _save_feats2_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump(
                (feat, self._feats2_models[feat].weights) if feat else
                {x: y.weights for x, y in self._feats2_models.items()},
                f, 2
            )

    def _load_feats2_models(self, file_path):
        with open(file_path, 'rb') as f:
            o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._feats2_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._feats2_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def predict_pos2(self, sentence, with_backoff=True, max_repeats=0,
                     inplace=True):
        """Tag the *sentence* with the POS-2 tagger.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where POS-1 taggers
                            don't concur
        :type max_repeats: int
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        cdict = self._cdict
        model = self._pos2_model
        assert model, 'ERROR: Use train_pos2() prior to prepare POS-2 tagger'
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        predict = self.predict_pos_ if hasattr(self, 'predict_pos_') else \
                      self.predict_pos
        sent = predict(sent, rev=False, inplace=True)
        sent_rev = predict(sent, rev=True, inplace=False)
        tokens_straight = [(x['FORM'], x['UPOS'])
                               for x in sent
                                   if x['FORM'] and x['UPOS']
                                                and '-' not in x['ID']]
        tokens_rev = [(x['FORM'], x['UPOS'])
                          for x in sent_rev
                              if x['FORM'] and x['UPOS']
                                           and '-' not in x['ID']]
        context, pos_context_straight = \
            [list(x) for x in zip(*[t for t in tokens_straight])] \
                if tokens_straight else \
            [[]] * 2
        pos_context_rev = [t[1] for t in tokens_rev]
        ## Rev model is better for initial word (with capital letter?)
        tokens_ = [[t[0], None] for t in tokens_rev][:2] \
                + [[t[0], None] for t in tokens_straight][2:]
        pos_context = pos_context_rev[:2] + pos_context_straight[2:]
        ###
        changes = len(sent) + 1
        i_ = 1
        while True:
            changes_prev = changes
            changes = 0
            pos_context_straight_i = iter(pos_context_straight)
            pos_context_rev_i      = iter(pos_context_rev     )
            i = 0
            for token in sent:
                wform = token['FORM']
                if wform and '-' not in token['ID']:
                    pos_straight = next(pos_context_straight_i)
                    pos_rev      = next(pos_context_rev_i     )
                    if pos_straight != pos_rev or (not with_backoff
                                               and max_repeats > 0):
                        guess, coef = cdict.predict_tag(wform, isfirst=i == 0)
                        if self._guess_pos:
                            guess, coef = self._guess_pos(guess, coef, i,
                                                          tokens_, cdict)
                        if guess is None or coef < 1.:
                            features = self.features.get_pos2_features(
                                i, context, pos_context
                            )
                            guess = model.predict(
                                features#, suggest=guess, suggest_coef=coef
                            )
                            if with_backoff and guess not in [pos_straight,
                                                              pos_rev]:
                                guess = pos_context[i]
                        if guess != token['UPOS']:
                            changes += 1
                        token['UPOS'] = tokens_[i][1] = pos_context[i] = guess
                    i += 1
            if with_backoff or changes == 0:
                break
            elif changes > changes_prev:
                for token, token_prev in zip(sent, sent_prev):
                    token['UPOS'] = token_prev['UPOS']
                break
            if i_ >= max_repeats:
                break
            sent_prev = deepcopy(sent)
            i_ += 1
        return sentence

    def predict_feats2(self, sentence, joint=False, with_backoff=True,
                       max_repeats=0, feat=None, inplace=True):
        """Tag the *sentence* with the FEATS-2 tagger.

        :param sentence: sentence in Parsed CONLL-U format; UPOS and LEMMA
                         fields must be already filled
        :type sentence: list(dict)
        :param joint: if True, use joint FEATS-2 model; elsewise, use separate
                      models (default)
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where FEATS-1 taggers
                            don't concur
        :type max_repeats: int
        :param feat: name of the feat to tag; if None, then all possible feats
                     will be tagged
        :type feat: str
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        return (
            self._predict_feats2_joint if joint else
            self._predict_feats2_separate
        )(
            sentence, with_backoff=with_backoff, max_repeats=max_repeats,
            feat=feat, inplace=inplace
        )

    def _predict_feats2_separate(self, sentence, with_backoff=True,
                                 max_repeats=0, feat=None, inplace=True):
        cdict = self._cdict
        models = self._feats2_models
        assert models, \
               'ERROR: Use train_feats2(joint=False) prior to prepare ' \
               'FEATS-2 tagger'
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        if not feat:
            for token in sent:
                token['FEATS'] = OrderedDict()
            for feat in cdict.get_feats():
                self._predict_feats2_separate(sent, with_backoff=with_backoff,
                                              max_repeats=max_repeats,
                                              feat=feat, inplace=True)
        else:
            default_val = '_'
            model = models[feat]
            val_cnt = len(cdict.get_feats()[feat]) - 1
            sent = self._predict_feats_separate(
                sent, rev=False, feat=feat, inplace=True
            )
            sent_rev = self._predict_feats_separate(
                sent, rev=True, feat=feat, inplace=False
            )
            tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                          for x in sent
                              if x['FORM'] and x['LEMMA'] and x['UPOS']
                             and '-' not in x['ID']]
            tokens_rev = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                              for x in sent_rev
                                  if x['FORM'] and x['LEMMA'] and x['UPOS']
                                 and '-' not in x['ID']]
            context, lemma_context, pos_context, feats_context_straight = \
                [list(x) for x in zip(*[t for t in tokens])] if tokens else \
                [[]] * 4
            feats_context_rev = [t[3] for t in tokens_rev]
            # Get straight version as backoff
            tokens_ = [[*t[:3], None] for t in tokens]
            feats_context = deepcopy(feats_context_straight)
            ###
            changes = len(tokens) + 1
            i_ = 1
            while True:
                changes_prev = changes
                changes = 0
                feats_context_straight_i = iter(feats_context_straight)
                feats_context_rev_i      = iter(feats_context_rev     )
                for i, (wform, lemma, pos, feats) in enumerate(tokens):
                    feat_val_straight = next(feats_context_straight_i) \
                                             .get(feat, default_val)
                    feat_val_rev      = next(feats_context_rev_i     ) \
                                             .get(feat, default_val)
                    if feat_val_straight != feat_val_rev or (not with_backoff
                                                         and max_repeats > 0):
                        guess, coef = \
                            cdict.predict_feat(feat, wform, lemma, pos)
                        if self._guess_feat:
                            guess, coef = \
                                self._guess_feat(guess, coef, i, feat,
                                                 tokens_, cdict)
                        if coef is not None and guess is None:
                            guess = default_val
                        if coef != 1.:
                            features = self.features.get_feat2_features(
                                    i, feat, context,
                                    lemma_context, pos_context, feats_context,
                                    False, val_cnt
                                )
                            guess = model.predict(
                                features, suggest=guess, suggest_coef=coef
                            )
                            if with_backoff and guess not in [feat_val_rev,
                                                              feat_val_straight]:
                                guess = feats_context[i].get(feat, default_val)
                        if guess != feats.get(feat, default_val):
                            changes += 1
                        tokens_[i][3] = guess
                        if guess != default_val:
                            feats[feat] = feats_context[i][feat] = guess
                        else:
                            feats.pop(feat, None)
                            feats_context[i].pop(feat, None)
                if with_backoff or changes == 0:
                    break
                elif changes > changes_prev:
                    for token, token_prev in zip(tokens, tokens_prev):
                        tokens[3] = token_prev[3].copy()
                    break
                if i_ >= max_repeats:
                    break
                tokens_prev = deepcopy(tokens)
                i_ += 1
        return sentence

    def _predict_feats2_joint(self, sentence, with_backoff=True, feat=None,
                              max_repeats=0, inplace=True):
        assert not feat, 'ERROR: feat must be None with joint=True'
        cdict = self._cdict
        model = self._feats2_model
        assert model, \
               'ERROR: Use train_feats2(joint=True) prior to prepare ' \
               'FEATS-2 tagger'
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        sent = self._predict_feats_joint(sent, rev=False, inplace=True)
        sent_rev = self._predict_feats_joint(sent, rev=True, inplace=False)
        tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                      for x in sent
                          if x['FORM'] and x['LEMMA'] and x['UPOS']
                         and '-' not in x['ID']]
        tokens_rev = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                          for x in sent_rev
                              if x['FORM'] and x['LEMMA'] and x['UPOS']
                             and '-' not in x['ID']]
        context, lemma_context, pos_context, feats_context_straight = \
            [list(x) for x in zip(*[t for t in tokens])] if tokens else \
            [[]] * 4
        feats_context_rev = [t[3] for t in tokens_rev]
        # Rev model is better for initial word (with capital letter?)
        feats_context = deepcopy(feats_context_rev[:1]
                               + feats_context_straight[1:])
        ###
        changes = len(feats_context_straight) + 1
        i_ = 1
        while True:
            changes_prev = changes
            changes = 0
            feats_context_rev_i      = iter(feats_context_rev     )
            feats_context_straight_i = iter(feats_context_straight)
            for i, feats in enumerate(feats_context_straight):
                feats_rev = next(feats_context_rev_i)
                feat_vals_rev = \
                    '|'.join('='.join((x, feats_rev[x]))
                                 for x in sorted(feats_rev))
                feats_straight = next(feats_context_straight_i)
                feat_vals_straight = \
                    '|'.join('='.join((x, feats_straight[x]))
                                 for x in sorted(feats_straight))
                if feat_vals_rev != feat_vals_straight or (not with_backoff
                                                       and max_repeats > 0):
                    features = self.features.get_feat2_features(
                        i, None,
                        context, lemma_context, pos_context, feats_context,
                        True, 0
                    )
                    guess = model.predict(features)
                    feats_ctx = '|'.join('='.join((x, feats_context[i][x]))
                                             for x in sorted(feats_context[i]))
                    if with_backoff and guess not in [feat_vals_rev,
                                                      feat_vals_straight]:
                        guess = feats_ctx
                    elif guess != feats_ctx:
                        changes += 1
                    feats.clear()
                    feats_ctx = feats_context[i]
                    feats_ctx.clear()
                    if guess:
                        for feat, val in [t.split('=') for t in guess.split('|')]:
                            feats[feat] = feats_ctx[feat] = val
            if with_backoff or changes == 0:
                break
            elif changes > changes_prev:
                for feats, feats_prev in zip(feats_context_straight,
                                             feats_context_straight_prev):
                    feats.clear()
                    feats.update(feats_prev)
                break
            if i_ >= max_repeats:
                break
            feats_context_straight_prev = deepcopy(feats_context_straight)
            i_ += 1
        return sentence

    def predict2(self, sentence, pos_backoff=True, pos_repeats=0,
                 feats_joint=False, feats_backoff=True, feats_repeats=0,
                 inplace=True):
        """Tag the *sentence* with the all available taggers.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one
                            repeat - only for tokens where POS-1 taggers
                            don't concur
        :type pos_repeats: int
        :param feats_joint: if True, use joint model; elsewise, use separate
                            models (default)
        :type feats_backoff: if result of FEATS-2 tagger differs from both its
                             base taggers, get one of the bases on the ground
                             of some heuristics
        :param feats_repeats: repeat a prediction step based on the previous
                              one while changes in prediction are diminishing
                              and ``max_repeats`` is not reached. 0 (default)
                              means one repeat - only for tokens where FEATS-1
                              taggers don't concur
        :type feats_repeats: int
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        return \
            self.predict_feats2(
                self.predict_lemma(
                    self.predict_pos2(
                        sentence, with_backoff=pos_backoff,
                        max_repeats=pos_repeats, inplace=inplace
                    ),
                    inplace=inplace
                ),
                joint=feats_joint, with_backoff=feats_backoff,
                max_repeats=feats_repeats, inplace=inplace
            )

    def predict_pos2_sents(self, sentences=None, with_backoff=True,
                           max_repeats=0, inplace=True, save_to=None):
        """Apply ``self.predict_pos2()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where POS-1 taggers
                            don't concur
        :type max_repeats: int
        :param inplace: if True, method changes and returns the given
                        sentences themselves; elsewise, new list of sentences
                        will be created
        :param save_to: if not None then the result will be saved to the file
                        with a specified name
        :type save_to: str
        """
        return self._predict_sents(
            sentences,
            lambda sentences:
                (self.predict_pos2(
                     s, with_backoff=with_backoff, max_repeats=max_repeats,
                     inplace=inplace
                 )
                     for s in sentences),
            save_to=save_to
        )

    def predict_feats2_sents(self, sentences=None, joint=False,
                             with_backoff=True, max_repeats=0, feat=None,
                             inplace=True, save_to=None):
        """Apply ``self.predict_feats2()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param joint: if True, use joint FEATS-2 model; elsewise, use separate
                      models (default)
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where FEATS-1 taggers
                            don't concur
        :type max_repeats: int
        :param feat: name of the feat to tag; if None, then all feats will be
                     tagged
        :type feat: str
        :param inplace: if True, method changes and returns the given
                        sentences themselves; elsewise, the new list of
                        sentences will be created
        :param save_to: if not None then the result will be saved to the file
                        with a specified name
        :type save_to: str
        """
        return self._predict_sents(
            sentences,
            lambda sentences:
                (self.predict_feats2(
                     s, joint=joint, with_backoff=with_backoff,
                     max_repeats=max_repeats, feat=feat, inplace=inplace
                 )
                     for s in sentences),
            save_to=save_to
        )

    def predict2_sents(self, sentences=None, pos_backoff=True, pos_repeats=0,
                       feats_joint=False, feats_backoff=True, feats_repeats=0,
                       inplace=True, save_to=None):
        """Apply ``self.predict2()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one
                            repeat - only for tokens where POS-1 taggers
                            don't concur
        :type pos_repeats: int
        :param feats_joint: if True, use joint model; elsewise, use separate
                            models (default)
        :type feats_backoff: if result of FEATS-2 tagger differs from both its
                            base taggers, get one of the bases on the ground
                            of some heuristics
        :param feats_repeats: repeat a prediction step based on the previous
                              one while changes in prediction are diminishing
                              and ``max_repeats`` is not reached. 0 (default)
                              means one repeat - only for tokens where FEATS-1
                              taggers don't concur
        :type feats_repeats: int
        :param inplace: if True, method changes and returns the given
                        sentences themselves; elsewise, new list of sentences
                        will be created
        :param save_to: if not None then the result will be saved to the file
                        with a specified name
        :type save_to: str
        """
        return self._predict_sents(
            sentences,
            lambda sentences:
                (self.predict2(
                     s, pos_backoff=pos_backoff, pos_repeats=pos_repeats,
                     feats_joint=feats_joint, feats_backoff=feats_backoff,
                     feats_repeats=feats_repeats, inplace=inplace
                 )
                     for s in sentences),
            save_to=save_to
        )

    def evaluate_pos2(self, gold=None, test=None, with_backoff=True,
                      max_repeats=0, pos=None, unknown_only=False,
                      silent=False):
        """Score the accuracy of the POS tagger against the *gold* standard.
        Remove POS tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where POS-1 taggers
                            don't concur
        :type max_repeats: int
        :param pos: name of the tag to evaluate the tagger; if None, then
                    tagger will be evaluated for all tags
        :type pos: str
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        self.predict_pos_ = self.predict_pos
        self.predict_pos = \
            lambda sentence, rev=None, inplace=True: \
                self.predict_pos2(sentence, with_backoff=with_backoff,
                                  max_repeats=max_repeats, inplace=inplace)
        res = self.evaluate_pos(gold=gold, test=test, pos=pos,
                                unknown_only=unknown_only, silent=silent)
        self.predict_pos = self.predict_pos_
        del self.predict_pos_
        return res

    def evaluate_feats2(self, gold=None, test=None, joint=False,
                        with_backoff=True, max_repeats=0,
                        feat=None, unknown_only=False, silent=False):
        """Score the accuracy of the FEATS-2 tagger against the *gold*
        standard. Remove feats (or only one specified feat) from the *gold*
        standard text, generate new feats using the tagger, then compute the
        accuracy score. If *test* is not None, compute the accuracy of the
        *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param joint: if True, use joint FEATS-2 model; elsewise, use separate
                      models (default)
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 (default) means
                            one repeat - only for tokens where FEATS-1 taggers
                            don't concur
        :type max_repeats: int
        :param feat: name of the feat to evaluate the tagger; if None, then
                     tagger will be evaluated for all feats
        :type feat: str
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
        :param silent: suppress log
        :return: accuracy scores of the tagger against the gold:
                 1. by tokens: the tagging of the whole token may be either
                    correct or not;
                 2. by tags: sum of correctly detected feats to sum of all
                    feats that are non-empty in either gold or retagged 
                    sentence
        :rtype: tuple(float, float)
        """
        f = self.predict_feats
        self.predict_feats = \
            lambda sentence, joint=joint, rev=None, feat=feat, inplace=True: \
                self.predict_feats2(sentence, joint=joint,
                                    with_backoff=with_backoff,
                                    max_repeats=max_repeats, feat=feat,
                                    inplace=inplace)
        res = self.evaluate_feats(gold=gold, test=test, joint=joint, feat=feat,
                                  unknown_only=unknown_only, silent=silent)
        self.predict_feats = f
        return res

    def evaluate2(self, gold=None, test=None, pos_backoff=True, pos_repeats=0,
                  feats_joint=False, feats_backoff=True, feats_repeats=0,
                  feat=None, unknown_only=False, silent=False):
        """Score a joint accuracy of the all available taggers against the
        *gold* standard. Extract wforms from the *gold* standard text, retag it
        using all the taggers, then compute a joint accuracy score. If *test*
        is not None, compute the accuracy of the *test* corpus with respect to
        the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one
                            repeat - only for tokens where POS-1 taggers
                            don't concur
        :type pos_repeats: int
        :param feats_joint: if True, use joint model; elsewise, use separate
                            models (default)
        :type feats_backoff: if result of FEATS-2 tagger differs from both its
                            base taggers, get one of the bases on the ground
                            of some heuristics
        :param feats_repeats: repeat a prediction step based on the previous
                              one while changes in prediction are diminishing
                              and ``max_repeats`` is not reached. 0 (default)
                              means one repeat - only for tokens where FEATS-1
                              taggers don't concur
        :type feats_repeats: int
        :param feat: name of the feat to evaluate the tagger; if None, then
                     tagger will be evaluated for all feats
        :type feat: str
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
        :param silent: suppress log
        :return: joint accuracy scores of the taggers against the gold:
                 1. by tokens: the tagging of the whole token may be either
                    correct or not
                 2. by tags: sum of correctly detected tags to sum of all tags
                    that are non-empty in either gold or retagged sentences
        :rtype: tuple(float, float)
        """
        f = self.predict
        self.predict = \
            lambda sentence, pos_rev=None, \
                   feats_joint=feats_joint, feats_rev=None, inplace=False: \
                self.predict2(
                    sentence, pos_backoff=pos_backoff, pos_repeats=pos_repeats,
                    feats_joint=feats_joint, feats_backoff=feats_backoff,
                    feats_repeats=feats_repeats, inplace=inplace
                )
        res = self.evaluate(gold=gold, test=test, feats_joint=feats_joint,
                            feat=feat, unknown_only=unknown_only,
                            silent=silent)
        self.predict = f
        return res

    def train_pos2(self, epochs=5, test_max_repeats=0, no_train_evals=True,
                   seed=None, dropout=None, context_dropout=None):
        """Train a POS-2 tagger from ``self._train_corpus``.

        :param epochs: number of training iterations. If epochs < 0, then the
                       best model will be searched based on evaluation of test
                       corpus. The search will be stopped when the result of
                       next |epochs| iterations will be worse than the best
                       one. It's allowed to specify epochs as tuple of both
                       variants (positive and negative)
        :type epochs: int|tuple(int, int)
        :param test_max_repeats: parameter for ``evaluate_pos2()``
        :type test_max_repeats: int
        :param no_train_evals: don't make interim and final evaluations on the
                               training set (save time)
        :param seed: init value for the random number generator
        :type seed: int
        :param dropout: a fraction of weiths to be randomly set to 0 at each
                        predict to prevent overfitting
        :type dropout: float
        :param context_dropout: a fraction of POS tags to be randomly replaced
                                after predict to random POS tags to prevent
                                overfitting
        :type context_dropout: float
        """
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)

        assert self._pos_model, \
               'ERROR: Use train_pos() prior to prepare POS tagger'
        assert self._pos_rev_model, \
               'ERROR: Use train_pos(rev=True) prior to prepare ' \
               'Reversed POS tagger'

        model = self._pos2_model = \
            _AveragedPerceptron(default_class=cdict.most_common_tag())

        header = 'POS-2'
        tags = sorted(cdict.get_tags())
        last_tag_idx = len(tags) - 1
        print(tags, file=LOG_FILE)
        best_epoch, best_score, best_weights, eqs, bads, score = \
                                                        -1, -1, None, 0, 0, -1
        epoch = 0
        while True:
            n = c = 0
            td = fd = td2 = fd2 = tp = fp = 0
            random.shuffle(self._train_corpus)
            print('{} Epoch {}'.format(header, epoch), file=LOG_FILE)
            for sent_no, sentence in enumerate(self._train_corpus):
                if not sent_no % progress_check_step:
                    print_progress(sent_no, end_value=corpus_len,
                                   step=progress_step)

                tokens = [(x['FORM'], x['UPOS'])
                              for x in sentence
                                  if x['FORM'] and '-' not in x['ID']]
                context, pos_context = \
                    [list(x) for x in zip(*[t for t in tokens])] \
                        if tokens else \
                    [[]] * 2
                tokens_ = [[t[0], None] for t in tokens]

                for i, (wform, pos) in enumerate(tokens):
                    guess, coef = cdict.predict_tag(wform, isfirst=i == 0)
                    if self._guess_pos:
                        guess, coef = self._guess_pos(guess, coef, i,
                                                      tokens_, cdict)
                    if guess is not None:
                        if guess == pos:
                            td2 += 1
                        else:
                            fd2 += 1
                    if guess is None or coef < 1.:
                        features = self.features.get_pos2_features(
                            i, context, pos_context
                        )
                        guess = model.predict(
                            features,# suggest=guess, suggest_coef=coef,
                            dropout=dropout
                        )
                        if guess == pos:
                            tp += 1
                        else:
                            fp += 1
                        model.update(pos, guess, features)
                    elif guess == pos:
                        td += 1
                    else:
                        fd += 1
                    n += 1
                    c += guess == pos

                    tokens_[i][1] = pos_context[i] = \
                        guess if not context_dropout \
                              or rand() >= context_dropout else \
                        tags[randint(0, last_tag_idx)]

            print_progress(sent_no + 1, end_value=corpus_len,
                           step=progress_step)
            epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                self._train_eval(
                    model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    self.evaluate_pos2,
                    {'with_backoff': False, 'max_repeats': test_max_repeats}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals, self.evaluate_pos2,
            {'with_backoff': False, 'max_repeats': test_max_repeats}
        )

    def train_feats2(self, joint=False, feat=None, epochs=5,
                     test_max_repeats=0, no_train_evals=True, seed=None,
                     dropout=None, context_dropout=None):
        """Train FEATS-2 taggers from ``self._train_corpus``.

        :param joint: if True, use joint FEATS-2 model; elsewise, train
                      separate models (default)
        :param feat: name of the feat to evaluate the tagger; if None, then
                     tagger will be evaluated for all feats
        :type feat: str
        :param epochs: number of training iterations. If epochs < 0, then the
                       best model will be searched based on evaluation of test
                       corpus. The search will be stopped when the result of
                       next |epochs| iterations will be worse than the best
                       one. It's allowed to specify epochs as tuple of both
                       variants (positive and negative)
        :type epochs: int|tuple(int, int)
        :param test_max_repeats: parameter for ``evaluate_feats2()``
        :type test_max_repeats: int
        :param no_train_evals: don't make interim and final evaluations on the
                               training set (save time)
        :param seed: init value for the random number generator
        :type seed: int
        :param dropout: a fraction of weiths to be randomly set to 0 at each
                        predict to prevent overfitting
        :type dropout: float
        :param context_dropout: a fraction of FEATS tags to be randomly replaced
                                after predict to random FEATS tags to prevent
                                overfitting
        :type context_dropout: float
        """
        return (
            self._train_feats2_joint if joint else
            self._train_feats2_separate
        )(
            feat=feat, epochs=epochs,
            no_train_evals=no_train_evals, test_max_repeats=test_max_repeats,
            seed=seed, dropout=dropout, context_dropout=context_dropout
        )

    def _train_feats2_separate(self, feat=None, epochs=5,
                               no_train_evals=True, test_max_repeats=0,
                               seed=None, dropout=None, context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)

        assert self._feats_models, \
               'ERROR: Use train_feats() prior to prepare FEATS tagger'
        assert self._feats_rev_models, \
               'ERROR: Use train_feats(rev=True) prior to prepare ' \
               'Reversed FEATS tagger'

        if feat:
            models = self._feats2_models
        else:
            models = self._feats2_models = {}

        default_val = '_'
        feat_vals = cdict.get_feats()
        if feat:
            feat_vals = {feat: feat_vals[feat]}
        for feat in sorted(feat_vals):
            header = 'FEAT-2<<{}>>'.format(feat)
            model = models[feat] = \
                _AveragedPerceptron(default_class=default_val)
            vals = sorted(feat_vals[feat])
            last_val_idx = len(vals) - 1
            print([x for x in vals if x != default_val], file=LOG_FILE)
            best_epoch, best_score, best_weights, eqs, bads, score = \
                                                        -1, -1, None, 0, 0, -1
            epoch = 0
            while True:
                n = c = 0
                td = fd = td2 = fd2 = tp = fp = 0
                random.shuffle(self._train_corpus)
                print('{} Epoch {}'.format(header, epoch), file=LOG_FILE)
                for sent_no, sentence in enumerate(self._train_corpus):
                    if not sent_no % progress_check_step:
                        print_progress(sent_no, end_value=corpus_len,
                                       step=progress_step)

                    tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                                  for x in sentence
                                      if x['FORM'] and x['LEMMA'] and x['UPOS']
                                     and '-' not in x['ID']]
                    context, lemma_context, pos_context, feats_context = \
                        [list(x) for x in zip(*[t for t in tokens])] \
                            if tokens else \
                        [[]] * 4
                    tokens_ = [[*t[:3], None] for t in tokens]

                    for i, (wform, lemma, pos, feats) in enumerate(tokens):
                        gold_val = feats.get(feat, default_val)
                        guess, coef = \
                            self._cdict.predict_feat(feat,
                                                     wform, lemma, pos)
                        if self._guess_feat:
                            guess, coef = self._guess_feat(guess, coef, i,
                                                           feat, tokens_,
                                                           cdict)
                        if coef is not None:
                            if guess is None:
                                guess = default_val
                            if guess == gold_val:
                                td2 += 1
                            else:
                                fd2 += 1
                        if coef == 1.:
                            if guess == gold_val:
                                td += 1
                            else:
                                fd += 1
                        else:
                            features = self.features.get_feat2_features(
                                i, feat, context,
                                lemma_context, pos_context, feats_context,
                                False, last_val_idx
                            )
                            guess = model.predict(
                                features, suggest=guess, suggest_coef=coef,
                                dropout=dropout
                            )
                            if guess == gold_val:
                                tp += 1
                            else:
                                fp += 1
                            model.update(gold_val, guess, features)
                        if guess != default_val or gold_val != default_val:
                            n += 1
                            c += guess == gold_val

                        tokens_[i][3] = \
                            guess if not context_dropout \
                                  or rand() >= context_dropout else \
                            vals[randint(0, last_val_idx)]

                print_progress(sent_no + 1, end_value=corpus_len,
                               step=progress_step)
                epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                    self._train_eval(
                        model, epoch, epochs, epochs_,
                        best_epoch, best_score, best_weights,
                        eqs, bads, score,
                        td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                        lambda **kwargs: self.evaluate_feats2(**kwargs)[1],
                        {'joint': False, 'with_backoff': False,
                         'max_repeats': test_max_repeats, 'feat': feat}
                    )
                if eqs == -1:
                    break

            res = self._train_done(
                header, model, eqs, no_train_evals,
                lambda **kwargs: self.evaluate_feats2(**kwargs)[1],
                {'joint': False, 'with_backoff': False,
                 'max_repeats': test_max_repeats, 'feat': feat}
            )

        return res if feat else \
               f_evaluate(joint=False, rev=rev, feat=feat, silent=True)

    def _train_feats2_joint(self, feat=None, epochs=5,
                            no_train_evals=True, test_max_repeats=0,
                            seed=None, dropout=None, context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)
        assert not feat, 'ERROR: feat must be None with joint=True'
        assert not context_dropout, \
               'ERROR: context_dropout must be None with joint=True'

        assert self._feats_model, \
               'ERROR: Use train_feats(joint=True) prior to prepare ' \
               'joint FEATS tagger'
        assert self._feats_rev_model, \
               'ERROR: Use train_feats(joint=True, rev=True) prior ' \
               'to prepare Reversed joint FEATS tagger'

        model = self._feats2_model = _AveragedPerceptron(default_class='')

        header = 'FEATS-2'
        best_epoch, best_score, best_weights, eqs, bads, score = \
                                                        -1, -1, None, 0, 0, -1
        epoch = 0
        while True:
            n = c = 0
            td = fd = td2 = fd2 = tp = fp = 0
            random.shuffle(self._train_corpus)
            print('{} Epoch {}'.format(header, epoch), file=LOG_FILE)
            for sent_no, sentence in enumerate(self._train_corpus):
                if not sent_no % progress_check_step:
                    print_progress(sent_no, end_value=corpus_len,
                                   step=progress_step)

                tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                              for x in sentence
                                  if x['FORM'] and x['LEMMA'] and x['UPOS']
                                 and '-' not in x['ID']]
                context, lemma_context, pos_context, feats_context = \
                    [list(x) for x in zip(*[t for t in tokens])] \
                        if tokens else \
                    [[]] * 4

                for i, feats in enumerate(feats_context):
                    gold = '|'.join('='.join((x, feats[x]))
                                        for x in sorted(feats))
                    features = self.features.get_feat2_features(
                        i, None,
                        context, lemma_context, pos_context, feats_context,
                        True, 0
                    )
                    guess = model.predict(features, dropout=dropout)
                    model.update(gold, guess, features)
                    n += 1
                    c += guess == gold

            print_progress(sent_no + 1, end_value=corpus_len,
                           step=progress_step)
            epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                self._train_eval(
                    model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    lambda **kwargs: self.evaluate_feats2(**kwargs)[1],
                    {'joint': True, 'with_backoff': False,
                     'max_repeats': test_max_repeats}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals,
            lambda **kwargs: self.evaluate_feats2(**kwargs)[1],
            {'joint': True, 'with_backoff': False,
            'max_repeats': test_max_repeats}
        )
