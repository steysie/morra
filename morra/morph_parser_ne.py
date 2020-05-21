# -*- coding: utf-8 -*-
# Morra project: Named Entity tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Adjusting Morphological parser's models for Named Entity Recognition. It
includes all modification of the algorithm: forward and backward, joint and
separate, combining joint and separate, and, finally, the NER-3 algorithm
that make prediction on the ground of all its predecessors.
"""
from copy import deepcopy
import pickle
import random
import sys

from corpuscula.corpus_utils import _AbstractCorpus
from corpuscula.utils import LOG_FILE, print_progress, vote
from morra.base_parser import _AveragedPerceptron, BaseParser
from morra.features_ne import FeaturesNE


class MorphParserNE(BaseParser):
    """Named Entity parser"""

    def __init__(self, features='RU', guess_ne=None):
        super().__init__()
        self.features = FeaturesNE(lang=features) \
            if isinstance(features, str) else features

        self._guess_ne = guess_ne

        self._ne_freq       = None  # [(ne, cnt, freq)], sorted by freq
        self._ne_model      = None
        self._ne_rev_model  = None
        self._ne2_model     = None
        self._ne_models     = {}
        self._ne_rev_models = {}
        self._ne2_models    = {}

    def backup(self):
        """Get current state"""
        o = super().backup()
        o.update({'ne_freq'              : self._ne_freq,
                  'ne_model_weights'     : self._ne_model.weights
                                               if self._ne_model else
                                           None,
                  'ne_rev_model_weights' : self._ne_rev_model.weights
                                               if self._ne_rev_model else
                                           None,
                  'ne2_model_weights'    : self._ne2_model.weights
                                               if self._ne2_model else
                                           None,
                  'ne_models_weights'    : {
                      x: y.weights for x, y in self._ne_models.items()
                  },
                  'ne_rev_models_weights': {
                      x: y.weights for x, y in self._ne_rev_models.items()
                  },
                  'ne2_models_weights'   : {
                      x: y.weights for x, y in self._ne2_models.items()
                  }})
        return o

    def restore(self, o):
        """Restore current state from backup object"""
        super().restore(o)
        (self._ne_freq        ,
         ne_model_weights     ,
         ne_rev_model_weights ,
         ne2_model_weights    ,
         ne_models_weights    ,
         ne_rev_models_weights,
         ne2_models_weights   ) = [o.get(x) for x in ['ne_freq'              ,
                                                      'ne_model_weights'     ,
                                                      'ne_rev_model_weights' ,
                                                      'ne2_model_weights'    ,
                                                      'ne_models_weights'    ,
                                                      'ne_rev_models_weights',
                                                      'ne2_models_weights'   ]]
        if ne_model_weights:
            self._ne_model = _AveragedPerceptron()
            self._ne_model.weights = ne_model_weights
        else:
            self._ne_model = None
        if ne_rev_model_weights:
            self._ne_rev_model = _AveragedPerceptron()
            self._ne_rev_model.weights = ne_rev_model_weights
        else:
            self._ne_rev_model = None
        if ne2_model_weights:
            self._ne2_model = _AveragedPerceptron()
            self._ne2_model.weights = ne2_model_weights
        else:
            self._ne2_model = None
        models = self._ne_models = {}
        if ne_models_weights:
            for ne, weights in ne_models_weights.items():
                model = models[ne] = _AveragedPerceptron()
                model.weights = weights
        models = self._ne_rev_models = {}
        if ne_rev_models_weights:
            for ne, weights in ne_rev_models_weights.items():
                model = models[ne] = _AveragedPerceptron()
                model.weights = weights
        models = self._ne2_models = {}
        if ne2_models_weights:
            for ne, weights in ne2_models_weights.items():
                model = models[ne] = _AveragedPerceptron()
                model.weights = weights

    def _save_ne_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self._ne_freq,
                         self._ne_model.weights if self._ne_model else
                         None), f, 2)

    def _load_ne_model(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, weights = pickle.load(f)
            if weights:
                self._ne_model = _AveragedPerceptron()
                self._ne_model.weights = weights
            else:
                self._ne_model = None

    def _save_ne_rev_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self._ne_freq,
                         self._ne_rev_model.weights if self._ne_rev_model else
                         None), f, 2)

    def _load_ne_rev_model(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, weights = pickle.load(f)
            if weights:
                self._ne_rev_model = _AveragedPerceptron()
                self._ne_rev_model.weights = weights
            else:
                self._ne_rev_model = None

    def _save_ne2_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self._ne_freq,
                         self._ne2_model.weights if self._ne2_model else
                         None), f, 2)

    def _load_ne2_model(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, weights = pickle.load(f)
            if weights:
                self._ne2_model = _AveragedPerceptron()
                self._ne2_model.weights = weights
            else:
                self._ne2_model = None

    def _save_ne_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump((
                self._ne_freq,
                (feat, self._ne_models[feat].weights) if feat else
                {x: y.weights for x, y in self._ne_models.items()}
            ), f, 2)

    def _load_ne_models(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._ne_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._ne_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def _save_ne_rev_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump((
                self._ne_freq,
                (feat, self._ne_rev_models[feat].weights) if feat else
                {x: y.weights for x, y in self._ne_rev_models.items()}
            ), f, 2)

    def _load_ne_rev_models(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._ne_rev_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._ne_rev_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def _save_ne2_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump((
                self._ne_freq,
                (feat, self._ne2_models[feat].weights) if feat else
                {x: y.weights for x, y in self._ne2_models.items()}
            ), f, 2)

    def _load_ne2_models(self, file_path):
        with open(file_path, 'rb') as f:
            self._ne_freq, o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._ne2_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._ne2_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def predict_ne(self, sentence, joint=True, rev=False, ne=None,
                   inplace=True, no_final_clean=False):
        """Tag the *sentence* with the NE tagger.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param joint: if True, use joint NE model (default); elsewise, use
                      separate models
        :param rev: if True, use Reversed NE tagger instead of generic
                    straight one
        :param ne: predict only specified NE. Can be used only with joint=False
        :type ne: str
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :param no_final_clean: don't search and remove NE from empty nodes
        :return: tagged sentence in Parsed CONLL-U format
        """
        return (self._predict_ne_joint if joint else
                self._predict_ne_separate)(sentence, rev=rev, ne=ne,
                                           inplace=inplace)

    def predict_ne2(self, sentence, joint=True, with_backoff=True,
                    max_repeats=0, ne=None, inplace=True, no_final_clean=False):
        """Tag the *sentence* with the NE-2 tagger.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param joint: if True, use joint NE-2 model (default); elsewise, use
                      separate models
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one repeat
                            only for tokens where NE-1 taggers don't concur
        :type max_repeats: int
        :param ne: predict only specified NE. Can be used only with joint=False
        :type ne: str
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :param no_final_clean: don't search and remove NE from empty nodes
        :return: tagged sentence in Parsed CONLL-U format
        """
        return (
            self._predict_ne2_joint if joint else
            self._predict_ne2_separate
        )(
            sentence, with_backoff=with_backoff, max_repeats=max_repeats,
            ne=ne, inplace=inplace
        )

    def _predict_ne_joint(self, sentence, rev=False,
                          ne=None, inplace=True, no_final_clean=False):
        assert not ne, 'ERROR: ne must be None with joint=True'
        cdict = self._cdict
        model = self._ne_rev_model if rev else self._ne_model
        assert model, 'ERROR: Use train_ne(joint=True{}) prior ' \
                      'to prepare {}NE tagger' \
                          .format(*((', rev=True', 'Reversed') if rev else
                                    ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        default_ne = '_'
        tokens = [(x['FORM'], x['LEMMA'],
                   x['UPOS'], x['FEATS'], x['MISC'])
                      for x in sent
                          if x['FORM'] and x['LEMMA'] and x['UPOS']
                         and '-' not in x['ID']]
        tokens_ = [[*x[:4], None] for x in tokens]
        if rev:
            tokens = tokens[::-1]
        context, lemma_context, pos_context, feats_context = \
            [list(x) for x in zip(*[t[:4] for t in tokens])] if tokens else \
            [[]] * 4
        prev, prev2 = self.features.START
        max_i = len(tokens) - 1
        for i, (wform, lemma, pos, feats, misc) in enumerate(tokens):
            i_ = max_i - i if rev else i
            guess, coef = None, None
            if self._guess_ne:
                guess, coef = self._guess_ne(None, None, i_,
                                             tokens_, cdict)
            if coef is not None and guess is None:
                guess = default_ne
            if coef != 1.:
                features = self.features.get_ne_features(
                    i, context, lemma_context, pos_context,
                    feats_context, prev, prev2
                )
                guess = model.predict(
                    features, suggest=guess, suggest_coef=coef
                )
            prev2 = prev
            tokens_[i_][4] = prev = guess
            if guess != default_ne:
                misc['NE'] = guess
            else:
                misc.pop('NE', None)
        if not no_final_clean:
            for misc in [
                x['MISC']
                    for x in sent
                        if not x['FORM'] or not x['LEMMA'] or not x['UPOS']
                       or '-' in x['ID']
            ]:
                misc.pop('NE', None)
        return sentence

    def _predict_ne_separate(self, sentence, rev=False,
                             ne=None, inplace=True, no_final_clean=None):
        cdict = self._cdict
        models = self._ne_rev_models if rev else self._ne_models
        assert models, 'ERROR: Use train_ne(joint=False{}) prior ' \
                       'to prepare {}NE tagger' \
                           .format(*((', rev=True', 'Reversed') if rev else
                                     ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        for token in sent:
            misc = token['MISC']
            if 'NE' in misc and (not ne or misc['NE'] == ne):
                del misc['NE']
        if not ne:
            for ne, _, _ in self._ne_freq:
                self._predict_ne_separate(sent, rev=rev, ne=ne, inplace=True)
        else:
            model = models[ne]
            default_ne = '_'
            tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'], x['MISC'])
                          for x in sent
                              if x['FORM'] and x['LEMMA'] and x['UPOS']
                             and '-' not in x['ID']]
            tokens_ = [[*x[:4], None] for x in tokens]
            if rev:
                tokens = tokens[::-1]
            context, lemma_context, pos_context, feats_context = \
                [list(x) for x in zip(*[t[:4] for t in tokens])] \
                    if tokens else \
                [[]] * 4
            prev, prev2 = self.features.START
            max_i = len(tokens) - 1
            for i, (wform, lemma, pos, feats, misc) in enumerate(tokens):
                i_ = max_i - i if rev else i
                guess, coef = None, None
                if self._guess_ne:
                    guess, coef = self._guess_ne(None, None, i_,
                                                 tokens_, cdict)
                if coef is not None:
                    guess = guess == ne
                if coef != 1.:
                    features = \
                        self.features.get_ne_features(
                            i, context, lemma_context, pos_context,
                            feats_context, prev, prev2
                        )
                    guess = model.predict(
                        features, suggest=guess, suggest_coef=coef
                    )
                prev2 = prev
                tokens_[i_][4] = prev = guess
                if guess:
                    misc['NE'] = ne
        return sentence

    def _predict_ne2_joint(self, sentence, with_backoff=True, max_repeats=0,
                           ne=None, inplace=True, no_final_clean=False):
        assert not ne, 'ERROR: ne must be None with joint=True'
        cdict = self._cdict
        model = self._ne2_model
        assert model, 'ERROR: Use train_ne2(joint=True) prior ' \
                      'to prepare NE-2 tagger'
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        default_ne = '_'
        sent = self._predict_ne_joint(sent, rev=True, ne=ne,
                                      inplace=True,
                                      no_final_clean=no_final_clean)
        sent_straight = self._predict_ne_joint(sent, rev=False, ne=ne,
                                               inplace=False,
                                               no_final_clean=True)
        tokens = [(x['FORM'], x['LEMMA'],
                   x['UPOS'], x['FEATS'], x['MISC'])
                      for x in sent
                          if x['FORM'] and x['LEMMA'] and x['UPOS']
                         and '-' not in x['ID']]
        tokens_straight = [(x['FORM'], x['LEMMA'],
                            x['UPOS'], x['FEATS'], x['MISC'])
                               for x in sent_straight
                                   if x['FORM'] and x['LEMMA'] and x['UPOS']
                                  and '-' not in x['ID']]
        context, lemma_context, pos_context, feats_context, ne_context_rev = \
            [list(x) for x in zip(*[[*t[:4], t[4].get('NE', default_ne)]
                                        for t in tokens])] if tokens else \
            [[]] * 5

        ne_context_straight = [t[4].get('NE', default_ne)
                                   for t in tokens_straight]
        # Rev model is better for initial word (with capital letter?)
        tokens_ = [[*t[:4], None] for t in tokens][:1] \
                + [[*t[:4], None] for t in tokens_straight][1:]
        ne_context = ne_context_rev[:1] + ne_context_straight[1:]
        ###
        changes = len(tokens) + 1
        i_ = 1
        while True:
            changes_prev = changes
            changes = 0
            ne_context_rev_i      = iter(ne_context_rev     )
            ne_context_straight_i = iter(ne_context_straight)
            for i, (wform, lemma, pos, feats, misc) in enumerate(tokens):
                ne_rev      = next(ne_context_rev_i     )
                ne_straight = next(ne_context_straight_i)
                if ne_rev != ne_straight or (not with_backoff
                                         and max_repeats > 0):
                    guess, coef = self._guess_ne(
                        None, None, i, tokens_, cdict
                    ) if self._guess_ne else (None, None)
                    if coef is not None and guess is None:
                        guess = default_ne
                    if coef != 1.:
                        features = self.features.get_ne2_features(
                            i, context, lemma_context, pos_context,
                            feats_context, ne_context
                        )
                        guess = model.predict(
                            features, suggest=guess, suggest_coef=coef
                        )
                        if with_backoff and guess not in [ne_rev, ne_straight]:
                            guess = ne_context[i]
                    if guess != misc.get('NE', default_ne):
                        changes += 1
                    tokens_[i][4] = ne_context[i] = guess
                    if guess != default_ne:
                        misc['NE'] = guess
                    else:
                        misc.pop('NE', None)
            if with_backoff or changes == 0:
                break
            elif changes > changes_prev:
                for token, token_prev in zip(tokens, tokens_prev):
                    tokens[4] = token_prev[4].copy()
                break
            if i_ >= max_repeats:
                break
            tokens_prev = deepcopy(tokens)
            i_ += 1
        return sentence

    def _predict_ne2_separate(self, sentence, with_backoff=True,
                              max_repeats=0, ne=None, inplace=True,
                              no_final_clean=None):
        cdict = self._cdict
        models = self._ne2_models
        assert models, \
               'ERROR: Use train_ne2(joint=False) prior ' \
               'to prepare NE-2 tagger' \
                   .format(*(('rev=True', 'Reversed') if rev else ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        if not ne:
            for ne, _, _ in self._ne_freq:
                self._predict_ne2_separate(sent, with_backoff=with_backoff,
                                           max_repeats=max_repeats,
                                           ne=ne, inplace=True)
        else:
            model = models[ne]
            default_ne = '_'
            sent = self._predict_ne_separate(sent, rev=True, ne=ne,
                                             inplace=True)
            sent_straight = self._predict_ne_separate(sent, rev=False, ne=ne,
                                                      inplace=False)
            tokens = [(x['FORM'], x['LEMMA'],
                       x['UPOS'], x['FEATS'], x['MISC'])
                          for x in sent
                              if x['FORM'] and x['LEMMA'] and x['UPOS']
                             and '-' not in x['ID']]
            tokens_straight = [(x['FORM'], x['LEMMA'],
                                x['UPOS'], x['FEATS'], x['MISC'])
                                   for x in sent_straight
                                       if x['FORM'] and x['LEMMA']
                                      and x['UPOS'] and '-' not in x['ID']]
            context, lemma_context, pos_context, feats_context, \
                                                            ne_context_rev = \
                [list(x) for x in zip(*[[*t[:4], t[4].get('NE', default_ne)]
                                            for t in tokens])] if tokens else \
                [[]] * 5
            ne_context_straight = [t[4].get('NE', default_ne)
                                       for t in tokens_straight]
            # Rev model is better for initial word (with capital letter?)
            tokens_ = [[*t[:4], None] for t in tokens][:1] \
                    + [[*t[:4], None] for t in tokens_straight][1:]
            ne_context = ne_context_rev[:1] + ne_context_straight[1:]
            ###
            changes = len(tokens) + 1
            i_ = 1
            while True:
                changes_prev = changes
                changes = 0
                ne_context_rev_i      = iter(ne_context_rev     )
                ne_context_straight_i = iter(ne_context_straight)
                for i, (wform, lemma, pos, feats, misc) in enumerate(tokens):
                    ne_rev      = next(ne_context_rev_i     )
                    ne_straight = next(ne_context_straight_i)
                    if ne_rev != ne_straight or (not with_backoff
                                             and max_repeats > 0):
                        guess, coef = None, None
                        if self._guess_ne:
                            guess, coef = self._guess_ne(None, None, i,
                                                         tokens_, cdict)
                        if coef is not None:
                            guess = guess == ne
                        if coef != 1.:
                            features = self.features.get_ne2_features(
                                i, context, lemma_context, pos_context,
                                feats_context, ne_context
                            )
                            guess = model.predict(
                                features, suggest=guess, suggest_coef=coef
                            )
                            if with_backoff and guess not in [ne_rev,
                                                              ne_straight]:
                                guess = ne_context[i]
                        if guess != misc.get('NE', default_ne):
                            changes += 1
                        tokens_[i][4] = ne_context[i] = guess
                        if guess:
                            misc['NE'] = ne_context[i] = ne
                if with_backoff or changes == 0:
                    break
                elif changes > changes_prev:
                    for token, token_prev in zip(tokens, tokens_prev):
                        tokens[4] = token_prev[4].copy()
                    break
                if i_ >= max_repeats:
                    break
                tokens_prev = deepcopy(tokens)
                i_ += 1
        return sentence

    def predict_ne3(self, sentence,
                    with_s_backoff=True, max_s_repeats=0,
                    with_j_backoff=True, max_j_repeats=0,
                    inplace=True):
        """Tag the *sentence* with the NE-3 tagger.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param with_s_backoff: if result of the separate NE-1 tagger differs
                               from both base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_ne2(joint=False)``
        :type max_s_repeats: int
        :param with_j_backoff: if result of the joint NE-1 tagger differs
                               from both base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_ne2(joint=True)``
        :type max_j_repeats: int
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged sentence in Parsed CONLL-U format
        """
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        sent = self._predict_ne2_separate(
            sent, with_backoff=with_s_backoff, max_repeats=max_s_repeats,
            inplace=True
        )
        sent_j = iter(self._predict_ne2_joint(
            sent, with_backoff=with_j_backoff, max_repeats=max_j_repeats,
            inplace=False, no_final_clean=True))
        for i, token in enumerate(sent):
            ne = token['MISC'].get('NE')
            ne_j = next(sent_j)['MISC'].get('NE')
            if not ne:
                if ne_j:
                    token['MISC']['NE'] = ne_j
            elif not ne_j:
                token['MISC'].pop('NE', None)
        return sentence

    def predict_ne_sents(self, sentences=None, joint=True, rev=False,
                         ne=None, inplace=True, save_to=None):
        """Apply ``self.predict_ne()`` to each element of *sentences*.

        :param sentences: a name of a file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param joint: if True, use joint NE model (default); elsewise, use
                      separate models
        :param rev: if True, use Reversed NE tagger instead of generic
                    straight one
        :param ne: name of the NE to tag; if None, then all possible NEs will
                   be tagged. Must be None if joint=True
        :type ne: str
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
                (self.predict_ne(
                     s, joint=joint, rev=rev, ne=ne, inplace=inplace
                 )
                     for s in sentences),
            save_to=save_to
        )

    def predict_ne2_sents(self, sentences=None, joint=True, with_backoff=True,
                          max_repeats=0, ne=None, inplace=True, save_to=None):
        """Apply ``self.predict_ne2()`` to each element of *sentences*.

        :param sentences: a name of a file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param joint: if True, use joint NE-2 model (default); elsewise, use
                      separate models
        :param with_backoff: if result of the tagger differs from both base
                             taggers, get one of the bases on the ground of
                             some heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one repeat
                            only for tokens where NE-1 taggers don't concur
        :type max_repeats: int
        :param ne: name of the NE to tag; if None, then all possible NEs will
                   be tagged. Must be None if joint=True
        :type ne: str
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
                (self.predict_ne2(
                     s, joint=joint, with_backoff=with_backoff,
                     max_repeats=max_repeats, ne=ne, inplace=inplace
                 )
                     for s in sentences),
            save_to=save_to
        )

    def predict_ne3_sents(self, sentences=None, inplace=True,
                          with_s_backoff=True, max_s_repeats=0,
                          with_j_backoff=True, max_j_repeats=0,
                          save_to=None):
        """Apply ``self.predict_ne3()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param with_s_backoff: if result of the separate NE-1 tagger differs
                               from both base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_ne3()``
        :type max_s_repeats: int
        :param with_j_backoff: if result of the joint NE-1 tagger differs
                               from both base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_ne3()``
        :type max_j_repeats: int
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
                (self.predict_ne3(s,
                                  with_s_backoff=with_s_backoff,
                                  max_s_repeats=max_s_repeats,
                                  with_j_backoff=with_j_backoff,
                                  max_j_repeats=max_j_repeats,
                                  inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def evaluate_ne(self, gold=None, test=None, joint=True, rev=False,
                    ne=None, silent=False):
        """Score the accuracy of the NE tagger against the *gold* standard.
        Remove NE tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param joint: if True, use joint NE model (default); elsewise, use
                      separate models
        :param rev: if True, use Reversed NE tagger instead of generic
                    straight one
        :param ne: name of the NE to evaluate; if None, then all possible NEs
                   will be evaluated
        :type ne: str
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        nn = cc = n = c = w = w1 = w2 = 0
        if gold is None:
            gold = self._test_corpus
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
          or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = self._get_corpus(gold)
        if test:
            test = self._get_corpus(test, silent=True)
        header = '{}NE{}'.format('Reversed ' if rev else '',
                                 '<<{}>>'.format(ne) if ne else '')
        if not silent:
            print('Evaluate ' + header, file=LOG_FILE)
        progress_step = max(int(corpus_len / 60), 1000) \
                            if corpus_len else 1000
        progress_check_step = min(int(corpus_len / 100), 1000) \
                                  if corpus_len else 100
        i = -1
        for i, gold_sent in enumerate(gold):
            if not silent and not i % progress_check_step:
                print_progress(i, end_value=corpus_len, step=progress_step,
                               file=LOG_FILE)
            test_sent = next(test) if test else \
                        self.predict_ne(gold_sent, joint=joint, rev=rev,
                                        ne=None if joint else ne,
                                        inplace=False)
            for j, gold_token in enumerate(gold_sent):
                if gold_token['FORM'] and '-' not in gold_token['ID']:
                    gold_ne = gold_token['MISC'].get('NE')
                    test_ne = test_sent[j]['MISC'].get('NE')
                    nn += 1
                    if (ne and (gold_ne == ne or test_ne == ne)) \
                    or (not ne and (gold_ne or test_ne)):
                        n += 1
                        if gold_ne == test_ne:
                            c += 1
                            cc += 1
                        elif not gold_ne or (ne and gold_ne != ne):
                            w1 += 1
                        elif not test_ne \
                          or (ne and test_ne != ne):
                            w2 += 1
                        else:
                            w += 1
                    else:
                        cc += 1
        if not silent:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                print_progress(i + 1,
                               end_value=corpus_len if corpus_len else 0,
                               step=progress_step, file=LOG_FILE)
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {}'.format(n), file=LOG_FILE)
                print(sp   + ' correct: {}'.format(c), file=LOG_FILE)
                print(sp   + '   wrong: {} [{} excess / {} absent{}]'
                                  .format(n - c, w1, w2, '' if ne else
                                                         ' / {} wrong type'
                                                             .format(w)),
                      file=LOG_FILE)
                print(sp   + 'Accuracy: {}'.format(c / n if n > 0 else 1.))
                print('[Total accuracy: {}]'.format(cc / nn if nn > 0 else 1.))
        return c / n if n > 0 else 1.

    def evaluate_ne2(self, gold=None, test=None, joint=True,
                     with_backoff=True, max_repeats=0, ne=None, silent=False):
        """Score the accuracy of the NE-2 tagger against the *gold* standard.
        Remove NE tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param joint: if True, use joint NE-2 model (default); elsewise, use
                      separate models
        :type with_backoff: if result of the tagger differs from both base
                            taggers, get one of the bases on the ground of some
                            heuristics
        :param max_repeats: repeat a prediction step based on the previous one
                            while changes in prediction are diminishing and
                            ``max_repeats`` is not reached. 0 means one repeat
                            only for tokens where NE-1 taggers don't concur
        :type max_repeats: int
        :param ne: name of the NE to evaluate; if None, then all possible NEs
                   will be evaluated
        :type ne: str
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        f = self.predict_ne
        self.predict_ne = \
            lambda sentence, joint=joint, rev=None, ne=ne, inplace=True, \
                   no_final_clean=False: \
                self.predict_ne2(sentence, joint=joint,
                                 with_backoff=with_backoff,
                                 max_repeats=max_repeats, ne=ne,
                                 inplace=inplace,
                                 no_final_clean=no_final_clean)
        res = self.evaluate_ne(gold=gold, test=test, joint=joint, ne=ne,
                               silent=silent)
        self.predict_ne_sents = f
        return res

    def evaluate_ne3(self, gold=None, test=None,
                     with_s_backoff=True, max_s_repeats=0,
                     with_j_backoff=True, max_j_repeats=0, silent=False):
        """Score the accuracy of the NE-3 tagger against the *gold* standard.
        Remove NE tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param with_s_backoff: if result of separate NE-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_ne3()``
        :type max_s_repeats: int
        :param with_j_backoff: if result of joint NE-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_ne3()``
        :type max_j_repeats: int
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        f = self.predict_ne
        self.predict_ne = \
            lambda sentence, joint=None, rev=None, ne=None, inplace=True, \
                   no_final_clean=None: \
                self.predict_ne3(sentence,
                                 with_s_backoff=with_s_backoff,
                                 max_s_repeats=max_s_repeats,
                                 with_j_backoff=with_j_backoff,
                                 max_j_repeats=max_j_repeats,
                                 inplace=inplace)
        res = self.evaluate_ne(gold=gold, test=test, silent=silent)
        self.predict_ne_sents = f
        return res

    def train_ne(self, joint=True, rev=False, ne=None, epochs=5,
                 no_train_evals=True, seed=None, dropout=None,
                 context_dropout=None):
        """Train a NE tagger from ``self._train_corpus``.

        :param joint: if True, train joint NE model (default); elsewise, train
                      separate models
        :param rev: if True, train Reversed NE tagger instead of generic
                    straight one
        :param ne: name of the entity to evaluate the tagger; if None
                   (default), then tagger will be evaluated for all entities.
        :param epochs: number of training iterations. If epochs < 0, then the
                       best model will be searched based on evaluation of test
                       corpus. The search will be stopped when the result of
                       next |epochs| iterations will be worse than the best
                       one. It's allowed to specify epochs as tuple of both
                       variants (positive and negative)
        :type epochs: int|tuple(int, int)
        :param no_train_evals: don't make interim and final evaluations on the
                               training set (save time)
        :param seed: init value for the random number generator
        :type seed: int
        :param dropout: a fraction of weiths to be randomly set to 0 at each
                        predict to prevent overfitting
        :type dropout: float
        :param context_dropout: a fraction of NE tags to be randomly replaced
                                after predict to random NE tags to prevent
                                overfitting
        """
        return (self._train_ne_joint if joint else
                self._train_ne_separate)(rev=rev, ne=ne, epochs=epochs,
                                         no_train_evals=no_train_evals,
                                         seed=seed)

    def train_ne2(self, joint=True, ne=None, epochs=5,
                  test_max_repeats=0, no_train_evals=True, seed=None,
                  dropout=None, context_dropout=None):
        """Train a NE-2 tagger from ``self._train_corpus``.

        :param joint: if True, train joint NE-2 model (default); elsewise,
                      train separate models
        :param epochs: number of training iterations. If epochs < 0, then the
                       best model will be searched based on evaluation of test
                       corpus. The search will be stopped when the result of
                       next |epochs| iterations will be worse than the best
                       one. It's allowed to specify epochs as tuple of both
                       variants (positive and negative)
        :type epochs: int|tuple(int, int)
        :param test_max_repeats: parameter for ``evaluate_ne2()``
        :type test_max_repeats: int
        :param no_train_evals: don't make interim and final evaluations on the
                               training set (save time)
        :param seed: init value for the random number generator
        :type seed: int
        :param dropout: a fraction of weiths to be randomly set to 0 at each
                        predict to prevent overfitting
        :type dropout: float
        :param context_dropout: a fraction of NE tags to be randomly replaced
                                after predict to random NE tags to prevent
                                overfitting
        """
        return (
            self._train_ne2_joint if joint else
            self._train_ne2_separate
        )(
            ne=ne, epochs=epochs,
            no_train_evals=no_train_evals, test_max_repeats=test_max_repeats,
            seed=seed
        )

    def _train_ne_joint(self, rev=False, ne=None, epochs=5,
                        no_train_evals=True, seed=None,
                        dropout=None, context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                                                           epochs, epochs_ = \
                        self._train_init(epochs, seed, allow_empty_cdict=True)
        assert not ne, 'ERROR: ne must be None with joint=True'

        default_ne = '_'

        model = _AveragedPerceptron(default_class=default_ne)
        if rev:
            self._ne_rev_model = model
        else:
            self._ne_model = model

        self._ne_freq = vote(x['MISC'].get('NE')
                                 for x in self._train_corpus for x in x
                                     if x['FORM'] and x['LEMMA'] and x['UPOS']
                                    and '-' not in x['ID']
                                    and x['MISC'].get('NE'))
        ne_classes = sorted(x[0] for x in self._ne_freq)

        header = '{}NE'.format('Reversed ' if rev else '')
        last_class_idx = len(ne_classes)
        print(ne_classes, file=LOG_FILE)
        ne_classes.append(default_ne)
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

                tokens = [(x['FORM'], x['LEMMA'],
                           x['UPOS'], x['FEATS'], x['MISC'])
                              for x in sentence
                                  if x['FORM'] and x['LEMMA'] and x['UPOS']
                                 and '-' not in x['ID']]
                tokens_ = [[*x[:4], None] for x in tokens]
                if rev:
                    tokens = tokens[::-1]
                context, lemma_context, pos_context, feats_context = \
                    [list(x) for x in zip(*[t[:4] for t in tokens])] \
                        if tokens else \
                    [[]] * 4
                prev, prev2 = self.features.START

                max_i = len(tokens) - 1
                for i, (wform, pos, lemma, feats, misc) in enumerate(tokens):
                    i_ = max_i - i if rev else i
                    gold_ne = misc.get('NE', default_ne)
                    guess, coef = None, None
                    if self._guess_ne:
                        guess, coef = self._guess_ne(None, None, i_,
                                                     tokens_, cdict)
                    if coef is not None:
                        if guess is None:
                            guess = default_ne
                        if guess == gold_ne:
                            td2 += 1
                        else:
                            fd2 += 1
                    if coef == 1.:
                        if guess == gold_ne:
                            td += 1
                        else:
                            fd += 1
                    else:
                        features = self.features.get_ne_features(
                            i, context, lemma_context, pos_context,
                            feats_context, prev, prev2
                        )
                        guess = model.predict(
                            features, suggest=guess, suggest_coef=coef,
                            dropout=dropout
                        )
                        if guess == gold_ne:
                            tp += 1
                        else:
                            fp += 1
                        model.update(gold_ne, guess, features)
                    if guess != default_ne or gold_ne != default_ne:
                        n += 1
                        c += guess == gold_ne

                    prev2 = prev
                    tokens_[i_][4] = prev = \
                        guess if not context_dropout \
                              or rand() >= context_dropout else \
                        ne_classes[randint(0, last_class_idx)]

            print_progress(sent_no + 1, end_value=corpus_len,
                           step=progress_step)
            epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                self._train_eval(
                    model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    self.evaluate_ne, {'joint': True, 'rev': rev}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals,
            self.evaluate_ne, {'joint': True, 'rev': rev}
        )

    def _train_ne_separate(self, rev=False, ne=None, epochs=5,
                           no_train_evals=True, seed=None, dropout=None,
                           context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                                                           epochs, epochs_ = \
                        self._train_init(epochs, seed, allow_empty_cdict=True)

        models = self._ne_rev_models if rev else self._ne_models

        self._ne_freq = vote(x['MISC'].get('NE')
                                 for x in self._train_corpus for x in x
                                     if x['FORM'] and x['LEMMA'] and x['UPOS']
                                    and '-' not in x['ID']
                                    and x['MISC'].get('NE'))
        ne_classes = [ne] if ne else [x[0] for x in self._ne_freq]

        for ne in [ne] if ne else sorted(ne_classes):
            model = models[ne] = _AveragedPerceptron()
            header = '{}NE<<{}>>'.format('Reversed ' if rev else '', ne)
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

                    tokens = [(x['FORM'], x['LEMMA'],
                               x['UPOS'], x['FEATS'], x['MISC'])
                                  for x in sentence
                                      if x['FORM'] and x['LEMMA'] and x['UPOS']
                                     and '-' not in x['ID']]
                    tokens_ = [[*t[:4], None] for t in tokens]
                    if rev:
                        tokens = tokens[::-1]
                    context, lemma_context, pos_context, feats_context = \
                        [list(x) for x in zip(*[t[:4] for t in tokens])] \
                            if tokens else \
                        [[]] * 4
                    prev, prev2 = self.features.START

                    max_i = len(tokens) - 1
                    for i, (wform, pos, lemma, feats, misc) in enumerate(tokens):
                        i_ = max_i - i if rev else i
                        gold_ne = misc.get('NE') == ne
                        guess, coef = None, None
                        if self._guess_ne:
                            guess, coef = self._guess_ne(None, None, i_,
                                                         tokens_, cdict)
                        if coef is not None:
                            guess = guess == ne
                            if guess == gold_ne:
                                td2 += 1
                            else:
                                fd2 += 1
                        if coef == 1.:
                            if guess == gold_ne:
                                td += 1
                            else:
                                fd += 1
                        else:
                            features = self.features.get_ne_features(
                                i, context, lemma_context, pos_context,
                                feats_context, prev, prev2
                            )
                            guess = model.predict(
                                features, suggest=guess, suggest_coef=coef,
                                dropout=dropout
                            )
                            if guess == gold_ne:
                                tp += 1
                            else:
                                fp += 1
                            model.update(gold_ne, guess, features)
                        if guess or gold_ne:
                            n += 1
                            c += guess == gold_ne

                        prev2 = prev
                        tokens_[i_][4] = prev = \
                            guess if not context_dropout \
                                  or rand() >= context_dropout else \
                            bool(randint(0, 1))

                print_progress(sent_no + 1, end_value=corpus_len,
                               step=progress_step)
                epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                    self._train_eval(
                        model, epoch, epochs, epochs_,
                        best_epoch, best_score, best_weights,
                        eqs, bads, score,
                        td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                        self.evaluate_ne,
                        {'joint': False, 'rev': rev, 'ne': ne}
                    )
                if eqs == -1:
                    break

            res = self._train_done(
                header, model, eqs, no_train_evals, self.evaluate_ne,
                {'joint': False, 'rev': rev, 'ne': ne}
            )

        return res if ne else \
               f_evaluate(joint=False, rev=rev, ne=ne, silent=True)

    def _train_ne2_joint(self, ne=None, epochs=5,
                         no_train_evals=True, test_max_repeats=0, seed=None,
                         dropout=None, context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                                                           epochs, epochs_ = \
                        self._train_init(epochs, seed, allow_empty_cdict=True)
        assert not ne, 'ERROR: ne must be None with joint=True'

        default_ne = '_'

        model = self._ne2_model = _AveragedPerceptron(default_class=default_ne)

        ne_classes = sorted(x[0] for x in self._ne_freq)

        header = 'NE-2'
        last_class_idx = len(ne_classes)
        print(ne_classes, file=LOG_FILE)
        ne_classes.append(default_ne)
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

                tokens = [(x['FORM'], x['LEMMA'],
                           x['UPOS'], x['FEATS'], x['MISC'])
                              for x in sentence
                                  if x['FORM'] and x['LEMMA'] and x['UPOS']
                                 and '-' not in x['ID']]
                context, lemma_context, pos_context, feats_context, \
                                                                ne_context = \
                    [list(x) for x in zip(*[[*t[:4], t[4].get('NE',
                                                              default_ne)]
                                                for t in tokens])] \
                        if tokens else \
                    [[]] * 5
                tokens_ = [[*t[:4], None] for t in tokens]

                for i, (wform, pos, lemma, feats, misc) in enumerate(tokens):
                    gold_ne = misc.get('NE', default_ne)
                    guess, coef = None, None
                    if self._guess_ne:
                        guess, coef = self._guess_ne(None, None, i,
                                                     tokens_, cdict)
                    if coef is not None:
                        if guess is None:
                            guess = default_ne
                        if guess == gold_ne:
                            td2 += 1
                        else:
                            fd2 += 1
                    if coef == 1.:
                        if guess == gold_ne:
                            td += 1
                        else:
                            fd += 1
                    else:
                        features = self.features.get_ne2_features(
                            i, context, lemma_context, pos_context,
                            feats_context, ne_context
                        )
                        guess = model.predict(
                            features, suggest=guess, suggest_coef=coef,
                            dropout=dropout
                        )
                        if guess == gold_ne:
                            tp += 1
                        else:
                            fp += 1
                        model.update(gold_ne, guess, features)
                    if guess != default_ne or gold_ne != default_ne:
                        n += 1
                        c += guess == gold_ne

                    tokens_[i][4] = ne_context[i] = \
                        guess if not context_dropout \
                              or rand() >= context_dropout else \
                        ne_classes[randint(0, last_class_idx)]

            print_progress(sent_no + 1, end_value=corpus_len,
                           step=progress_step)
            epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                self._train_eval(
                    model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    self.evaluate_ne2, {'joint': True, 'with_backoff': False,
                                        'max_repeats': test_max_repeats}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals,
            self.evaluate_ne2, {'joint': True, 'with_backoff': False,
                                'max_repeats': test_max_repeats}
        )

    def _train_ne2_separate(self, ne=None, epochs=5,
                            no_train_evals=True, test_max_repeats=0,
                            seed=None, dropout=None, context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                                                           epochs, epochs_ = \
                        self._train_init(epochs, seed, allow_empty_cdict=True)

        default_ne = '_'

        models = self._ne2_models

        ne_classes = [ne] if ne else [x[0] for x in self._ne_freq]

        for ne in [ne] if ne else sorted(ne_classes):
            model = models[ne] = _AveragedPerceptron()
            header = 'NE-2<<{}>>'.format(ne)
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

                    tokens = [(x['FORM'], x['LEMMA'],
                               x['UPOS'], x['FEATS'], x['MISC'])
                                  for x in sentence
                                      if x['FORM'] and x['LEMMA'] and x['UPOS']
                                     and '-' not in x['ID']]
                    context, lemma_context, pos_context, feats_context, \
                                                                ne_context = \
                        [list(x) for x in zip(*[[*t[:4], t[4].get('NE',
                                                                  default_ne)]
                                                    for t in tokens])] \
                            if tokens else \
                        [[]] * 5
                    tokens_ = [[*t[:4], None] for t in tokens]

                    for i, (wform, pos, lemma, feats, misc) in enumerate(tokens):
                        gold_ne = misc.get('NE') == ne
                        guess, coef = None, None
                        if self._guess_ne:
                            guess, coef = self._guess_ne(None, None, i,
                                                         tokens_, cdict)
                        if coef is not None:
                            guess = guess == ne
                            if guess == gold_ne:
                                td2 += 1
                            else:
                                fd2 += 1
                        if coef == 1.:
                            if guess == gold_ne:
                                td += 1
                            else:
                                fd += 1
                        else:
                            features = self.features.get_ne2_features(
                                i, context, lemma_context, pos_context,
                                feats_context, ne_context
                            )
                            guess = model.predict(
                                features, suggest=guess, suggest_coef=coef,
                                dropout=dropout
                            )
                            if guess == gold_ne:
                                tp += 1
                            else:
                                fp += 1
                            model.update(gold_ne, guess, features)
                        if guess or gold_ne:
                            n += 1
                            c += guess == gold_ne

                        tokens_[i][4] = ne_context[i] = \
                            guess if not context_dropout \
                                  or rand() >= context_dropout else \
                            bool(randint(0, 1))

                print_progress(sent_no + 1, end_value=corpus_len,
                               step=progress_step)
                epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                    self._train_eval(
                        model, epoch, epochs, epochs_,
                        best_epoch, best_score, best_weights,
                        eqs, bads, score,
                        td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                        self.evaluate_ne2,
                        {'joint': False, 'with_backoff': False,
                         'max_repeats': test_max_repeats, 'ne': ne}
                    )
                if eqs == -1:
                    break

            res = self._train_done(
                header, model, eqs, no_train_evals, self.evaluate_ne2,
                {'joint': False, 'with_backoff': False,
                 'max_repeats': test_max_repeats, 'ne': ne}
            )

        return res if ne else \
               f_evaluate(joint=False, ne=ne, silent=True)
