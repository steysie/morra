# -*- coding: utf-8 -*-
# Morra project: Morphological parser
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Single-pass forward and backward morphological parsers for POS and FEATS
tagging. Also, wrapper for LEMMA generator of Corpuscula project is added.
"""
from collections import OrderedDict
from copy import deepcopy
import pickle
import random
from random import randint, random as rand
import sys

from corpuscula.corpus_utils import _AbstractCorpus
from corpuscula.utils import LOG_FILE, print_progress
from morra.base_parser import _AveragedPerceptron, BaseParser
from morra.features import Features


class MorphParser(BaseParser):
    """Single-pass forward and backward morphological parsers. Based on
    PerceptronTagger from NLTK extended by CorpusDict methods and generalized
    to features other than POS"""

    def __init__(self, features='RU',
                 guess_pos=None, guess_lemma=None, guess_feat=None):
        super().__init__()
        self.features = Features(lang=features) \
            if isinstance(features, str) else features

        self._guess_pos    = guess_pos
        self._guess_lemma  = guess_lemma
        self._guess_feat   = guess_feat

        self._pos_model        = None
        self._pos_rev_model    = None
        self._lemma_model      = None
        self._feats_model      = None
        self._feats_rev_model  = None
        self._feats_models     = {}
        self._feats_rev_models = {}

    def backup(self):
        """Get current state"""
        o = super().backup()
        o.update({'pos_model_weights'       : self._pos_model.weights
                                                  if self._pos_model       else
                                              None,
                  'pos_rev_model_weights'   : self._pos_rev_model.weights
                                                  if self._pos_rev_model   else
                                              None,
                  'self._lemma_model'       : self._lemma_model,
                  'feats_model_weights'     : self._feats_model.weights
                                                  if self._feats_model     else
                                              None,
                  'feats_rev_model_weights' : self._feats_rev_model.weights
                                                  if self._feats_rev_model else
                                              None,
                  'feats_models_weights'    : {
                      x: y.weights for x, y in self._feats_models.items()
                  },
                  'feats_rev_models_weights': {
                      x: y.weights for x, y in self._feats_rev_models.items()
                  }})
        return o

    def restore(self, o):
        """Restore current state from backup object"""
        super().restore(o)
        (pos_model_weights       ,
         pos_rev_model_weights   ,
         self._lemma_model       ,
         feats_model_weights     ,
         feats_rev_model_weights ,
         feats_models_weights    ,
         feats_rev_models_weights) = \
             [o.get(x) for x in ['pos_model_weights'       ,
                                 'pos_rev_model_weights'   ,
                                 'self._lemma_model'       ,
                                 'feats_model_weights'     ,
                                 'feats_rev_model_weights' ,
                                 'feats_models_weights'    ,
                                 'feats_rev_models_weights']]
        if pos_model_weights:
            self._pos_model = _AveragedPerceptron()
            #self._pos_model.default_class = self._cdict.most_common_tag()
            self._pos_model.weights = pos_model_weights
        else:
            self._pos_model = None
        if pos_rev_model_weights:
            self._pos_rev_model = _AveragedPerceptron()
            self._pos_rev_model.weights = pos_rev_model_weights
        else:
            self._pos_rev_model = None
        if feats_model_weights:
            self._feats_model = _AveragedPerceptron()
            self._feats_model.weights = feats_model_weights
        else:
            self._feats_model = None
        if feats_rev_model_weights:
            self._feats_rev_model = _AveragedPerceptron()
            self._feats_rev_model.weights = feats_rev_model_weights
        else:
            self._feats_rev_model = None
        models = self._feats_models = {}
        if feats_models_weights:
            for feat, weights in feats_models_weights.items():
                model = models[feat] = _AveragedPerceptron()
                #model.default_class = '_'
                model.weights = weights
        models = self._feats_rev_models = {}
        if feats_rev_models_weights:
            for feat, weights in feats_rev_models_weights.items():
                model = models[feat] = _AveragedPerceptron()
                model.weights = weights

    def _save_pos_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._pos_model.weights if self._pos_model else
                        None, f, 2)

    def _load_pos_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._pos_model = _AveragedPerceptron()
                self._pos_model.weights = weights
            else:
                self._pos_model = None

    def _save_pos_rev_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._pos_rev_model.weights if self._pos_rev_model else
                        None, f, 2)

    def _load_pos_rev_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._pos_rev_model = _AveragedPerceptron()
                self._pos_rev_model.weights = weights
            else:
                self._pos_rev_model = None

    def _save_lemma_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._lemma_model, f, 2)

    def _load_lemma_model(self, file_path):
        with open(file_path, 'rb') as f:
            self._lemma_model = pickle.load(f)

    def _save_feats_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._feats_model.weights if self._feats_model else
                        None, f, 2)

    def _load_feats_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._feats_model = _AveragedPerceptron()
                self._feats_model.weights = weights
            else:
                self._feats_model = None

    def _save_feats_rev_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._feats_rev_model.weights
                            if self._feats_rev_model else
                        None, f, 2)

    def _load_feats_rev_model(self, file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
            if weights:
                self._feats_rev_model = _AveragedPerceptron()
                self._feats_rev_model.weights = weights
            else:
                self._feats_rev_model = None

    def _save_feats_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump(
                (feat, self._feats_models[feat].weights) if feat else
                {x: y.weights for x, y in self._feats_models.items()},
                f, 2
            )

    def _load_feats_models(self, file_path):
        with open(file_path, 'rb') as f:
            o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._feats_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._feats_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def _save_feats_rev_models(self, file_path, feat=None):
        with open(file_path, 'wb') as f:
            pickle.dump(
                (feat, self._feats_rev_models[feat].weights) if feat else
                {x: y.weights for x, y in self._feats_rev_models.items()},
                f, 2
            )

    def _load_feats_rev_models(self, file_path):
        with open(file_path, 'rb') as f:
            o = pickle.load(f)
            if isinstance(o, tuple):
                feat, weights = o
                model = self._feats_rev_models[feat] = _AveragedPerceptron()
                model.weights = weights
            else:
                models = self._feats_rev_models = {}
                for feat, weights in o.items():
                    model = models[feat] = _AveragedPerceptron()
                    model.weights = weights

    def predict_pos(self, sentence, rev=False, inplace=True):
        """Tag the *sentence* with the POS tagger.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param rev: if True, use Reversed POS tagger instead of generic
                    straight one
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        cdict = self._cdict
        model = self._pos_rev_model if rev else self._pos_model
        assert model, \
               'ERROR: Use train_pos({}) prior to prepare {}POS tagger' \
                   .format(*(('rev=True', 'Reversed') if rev else ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        tokens = [[x['FORM'], None]
                      for x in sent
                          if x['FORM'] and '-' not in x['ID']]
        context = [t[0] for t in (reversed(tokens) if rev else tokens)]
        prev, prev2 = self.features.START
        i = 0
        max_i = len(context) - 1
        for token in reversed(sent) if rev else sent:
            wform = token['FORM']
            if wform and '-' not in token['ID']:
                i_ = max_i - i if rev else i
                guess, coef = cdict.predict_tag(wform, isfirst=i_ == 0)
                if self._guess_pos:
                    guess, coef = self._guess_pos(guess, coef, i_,
                                                  tokens, cdict)
                if guess is None or coef < 1.:
                    features = self.features.get_pos_features(
                        i, context, prev, prev2
                    )
                    guess = model.predict(
                        features, suggest=guess, suggest_coef=coef
                    )
                prev2 = prev
                token['UPOS'] = tokens[i_][1] = prev = guess
                i += 1
            else:
                token['UPOS'] = None
        return sentence

    def predict_lemma(self, sentence, inplace=True):
        """Generate lemmata for wforms of the *sentence*.

        :param sentence: sentence in Parsed CONLL-U format; UPOS field must be 
                         already filled
        :type sentence: list(dict)
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, the new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        cdict = self._cdict
        assert self._lemma_model, \
               'ERROR: Use train_lemma() prior to prepare Lemma generator'
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        tokens = [[x['FORM'], x['UPOS'], None]
                      for x in sent
                          if x['FORM'] and x['UPOS'] and '-' not in x['ID']]
        i = 0
        for token in sent:
            wform, pos = token['FORM'], token['UPOS']
            if wform and pos and '-' not in token['ID']:
                guess, coef = cdict.predict_lemma(wform, pos, isfirst=i == 0)
                if self._guess_lemma:
                    guess, coef = self._guess_lemma(guess, coef, i,
                                                    tokens, cdict)
                token['LEMMA'] = tokens[i][2] = guess
                i += 1
            else:
                token['LEMMA'] = None
        return sentence

    def predict_feats(self, sentence, joint=False, rev=False, feat=None,
                      inplace=True):
        """Tag the *sentence* with the FEATS tagger.

        :param sentence: sentence in Parsed CONLL-U format; UPOS and LEMMA
                         fields must be already filled
        :type sentence: list(dict)
        :param joint: if True, use joint FEATS model; elsewise, use separate
                      models (default)
        :param rev: if True, use Reversed FEATS tagger instead of generic
                    straight one
        :param feat: name of the feat to tag; if None, then all possible feats
                     will be tagged
        :type feat: str
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        return (
            self._predict_feats_joint if joint else
            self._predict_feats_separate
        )(
            sentence, rev=rev, feat=feat, inplace=inplace
        )

    def _predict_feats_separate(self, sentence, rev=False, feat=None,
                                inplace=True):
        cdict = self._cdict
        models = self._feats_rev_models if rev else self._feats_models
        assert models, \
               'ERROR: Use train_feats(joint=False{}) prior to prepare ' \
               '{}FEATS tagger' \
                   .format(*((', rev=True', 'Reversed') if rev else ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        if not feat:
            for token in sent:
                token['FEATS'] = OrderedDict()
            for feat in cdict.get_feats():
                self._predict_feats_separate(sent, rev=rev, feat=feat,
                                             inplace=True)
        else:
            default_val = '_'
            model = models[feat]
            val_cnt = len(cdict.get_feats()[feat]) - 1
            tokens = [[x['FORM'], x['LEMMA'], x['UPOS'], None]
                          for x in sent
                              if x['FORM'] and x['LEMMA'] and x['UPOS']
                             and '-' not in x['ID']]
            context, lemma_context, pos_context = [list(x) for x in zip(
                *[t[:3] for t in (reversed(tokens) if rev else tokens)]
            )] if tokens else [[]] * 3
            prev, prev2 = self.features.START
            i = 0
            max_i = len(tokens) - 1
            for token in reversed(sent) if rev else sent:
                wform, lemma, pos, feats = token['FORM'], token['LEMMA'], \
                                           token['UPOS'], token['FEATS']
                if wform and lemma and pos and '-' not in token['ID']:
                    i_ = max_i - i if rev else i
                    guess, coef = cdict.predict_feat(feat, wform, lemma, pos)
                    if self._guess_feat:
                        guess, coef = self._guess_feat(guess, coef, i_,
                                                       feat, tokens, cdict)
                    if coef is not None and guess is None:
                        guess = default_val
                    if coef != 1.:
                        features = self.features.get_feat_features(
                            i, context, lemma_context, pos_context,
                            False, val_cnt, prev, prev2
                        )
                        guess = model.predict(
                            features, suggest=guess, suggest_coef=coef
                        )
                    prev2 = prev
                    tokens[i_][3] = prev = guess
                    if guess != default_val:
                        feats[feat] = guess
                    else:
                        feats.pop(feat, None)
                    i += 1
                else:
                    feats.clear()
        return sentence

    def _predict_feats_joint(self, sentence, rev=False, feat=None,
                             inplace=True):
        assert not feat, 'ERROR: feat must be None with joint=True'
        cdict = self._cdict
        model = self._feats_rev_model if rev else self._feats_model
        assert model, \
               'ERROR: Use train_feats(joint=True{}) prior to prepare ' \
               '{}FEATS tagger' \
                   .format(*((', rev=True', 'Reversed') if rev else ('', '')))
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        tokens = [(x['FORM'], x['LEMMA'], x['UPOS'])
                      for x in (reversed(sentence) if rev else sentence)
                          if x['FORM'] and x['LEMMA'] and x['UPOS']
                         and '-' not in x['ID']]
        context, lemma_context, pos_context = \
            [list(x) for x in zip(*[t for t in tokens])] if tokens else \
            [[]] * 3
        prev, prev2 = self.features.START
        i = 0
        for token in reversed(sent) if rev else sent:
            wform, lemma, pos, feats = token['FORM'], token['LEMMA'], \
                                       token['UPOS'], token['FEATS']
            feats.clear()
            if wform and lemma and pos and '-' not in token['ID']:
                features = self.features.get_feat_features(
                    i, context, lemma_context, pos_context,
                    True, 0, prev, prev2
                )
                guess = model.predict(features)
                prev2 = prev
                prev = guess
                feats.clear()
                if guess:
                    for feat, val in [t.split('=') for t in guess.split('|')]:
                        feats[feat] = val
                i += 1
        return sentence

    def predict(self, sentence, pos_rev=False, feats_joint=False,
                feats_rev=False, inplace=True):
        """Tag the *sentence* with the all available taggers.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :param pos_rev: if True, use Reversed POS tagger instead of generic
                        straight one
        :param feats_joint: if True, use joint FEATS model; elsewise, use
                            separate models (default)
        :param feats_rev: if True, use Reversed FEATS tagger instead of generic
                          straight one
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        return \
            self.predict_feats(
                self.predict_lemma(
                    self.predict_pos(
                        sentence, rev=pos_rev, inplace=inplace
                    ),
                    inplace=inplace
                ),
                joint=feats_joint, rev=feats_rev, inplace=inplace
            )

    def predict_pos_sents(self, sentences=None, rev=False, inplace=True,
                          save_to=None):
        """Apply ``self.predict_pos()`` to each element of *sentences*.

        :param sentences: a name of a file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param rev: if True, use Reversed POS tagger instead of generic
                    straight one
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
                (self.predict_pos(s, rev=rev, inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def predict_lemma_sents(self, sentences=None, inplace=True, save_to=None):
        """Apply ``self.predict_lemma()`` to each element of *sentences*.

        :param sentences: a name of a file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
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
                (self.predict_lemma(s, inplace=inplace) for s in sentences),
            save_to=save_to
        )

    def predict_feats_sents(self, sentences=None, joint=False, rev=False,
                            feat=None, inplace=True, save_to=None):
        """Apply ``self.predict_feats()`` to each element of *sentences*.

        :param sentences: a name of a file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param joint: if True, use joint FEATS model; elsewise, use separate
                      models (default)
        :param rev: if True, use Reversed FEATS tagger instead of generic
                    straight one
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
                (self.predict_feats(s, joint=joint, rev=rev, feat=feat,
                                    inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def predict_sents(self, sentences=None, pos_rev=False, feats_joint=False,
                      feats_rev=False, inplace=True, save_to=None):
        """Apply ``self.predict()`` to each element of *sentences*.

        :param sentences: a name of the file in CONLL-U format or list/iterator
                          of sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param pos_rev: if True, use Reversed POS tagger instead of generic
                        straight one
        :param feats_joint: if True, use joint FEATS model; elsewise, use
                            separate models (default)
        :param feats_rev: if True, use Reversed FEATS tagger instead of generic
                          straight one
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
                (self.predict(s, pos_rev=pos_rev, feats_joint=feats_joint,
                              feats_rev=feats_rev, inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def evaluate_pos(self, gold=None, test=None, rev=False, pos=None,
                     unknown_only=False, silent=False):
        """Score the accuracy of the POS tagger against the *gold* standard.
        Remove POS tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param rev: if True, use Reversed POS tagger instead of generic
                    straight one
        :param pos: name of the tag to evaluate the tagger; if None, then
                    tagger will be evaluated for all tags
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        n = c = nt = ct = w1 = w2 = 0
        if gold is None:
            gold = self._test_corpus
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
        or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = self._get_corpus(gold, silent=True)
        if test:
            test = self._get_corpus(test, silent=True)
        header = '{}POS{}'.format('Reversed ' if rev else '',
                                  '<<{}>>'.format(pos) if pos else '')
        if not silent:
            print('Evaluate ' + header, file=LOG_FILE)
        progress_step = max(int(corpus_len / 60), 1000) \
                            if corpus_len else 1000
        progress_check_step = min(int(corpus_len / 100), 1000) \
                                  if corpus_len else 100
        cdict = self._cdict
        i = -1
        for i, gold_sent in enumerate(gold):
            if not silent and not i % progress_check_step:
                print_progress(i, end_value=corpus_len, step=progress_step,
                               file=LOG_FILE)
            test_sent = next(test) if test else \
                        self.predict_pos(gold_sent, rev=rev, inplace=False)
            for j, gold_token in enumerate(gold_sent):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_pos = gold_token['UPOS']
                    if not (
                        unknown_only and cdict.wform_isknown(wform,
                                                             tag=gold_pos)
                    ):
                        test_pos = test_sent[j]['UPOS']
                        n += 1
                        c += gold_pos == test_pos
                        if pos and (gold_pos == pos or test_pos == pos):
                            nt += 1
                            if gold_pos == test_pos:
                                ct += 1
                            elif test_pos == pos:
                                w1 += 1
                            else:
                                w2 += 1
        if not silent:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                print_progress(i + 1,
                               end_value=corpus_len if corpus_len else 0,
                               step=progress_step, file=LOG_FILE)
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {}'.format(nt if pos else n),
                      file=LOG_FILE)
                print(sp   + ' correct: {}'.format(ct if pos else c),
                      file=LOG_FILE)
                print(sp   + '   wrong: {}{}'
                                 .format(nt - ct if pos else n - c,
                                         ' [{} excess / {} absent]'
                                             .format(w1, w2) if pos else
                                         ''),
                      file=LOG_FILE)
                if pos:
                    print(sp   + 'Accuracy: {}'
                                     .format(ct / nt if nt > 0 else 1.),
                          file=LOG_FILE)
                print('Total accuracy: {}'.format(c / n if n > 0 else 1.),
                      file=LOG_FILE)
        return c / n if n > 0 else 1.

    def evaluate_lemma(self, gold=None, test=None, unknown_only=False,
                       silent=False):
        """Score the accuracy of the Lemma generator against the *gold*
        standard. Remove lemmata from the *gold* standard text, generate new
        lemmata using the tagger, then compute the accuracy score. If *test* is
        not None, compute the accuracy of the *test* corpus with respect to the
        *gold*.

        :param gold: a corpus of tagged sentences to score the generator on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
        :param silent: suppress log
        :return: accuracy score of the generator against the gold
        :rtype: float
        """
        n = c = 0
        if gold is None:
            gold = self._test_corpus
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
        or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = self._get_corpus(gold, silent=True)
        if test:
            test = self._get_corpus(test, silent=True)
        header = 'LEMMA'
        if not silent:
            print('Evaluate ' + header, file=LOG_FILE)
        progress_step = max(int(corpus_len / 60), 1000) \
                            if corpus_len else 1000
        progress_check_step = min(int(corpus_len / 100), 1000) \
                                  if corpus_len else 100
        cdict = self._cdict
        i = -1
        for i, gold_sent in enumerate(gold):
            if not silent and not i % progress_check_step:
                print_progress(i, end_value=corpus_len, step=progress_step,
                               file=LOG_FILE)
            test_sent = next(test) if test else \
                        self.predict_lemma(gold_sent, inplace=False)
            for j, gold_token in enumerate(gold_sent):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_pos = gold_token['UPOS']
                    if not (
                        unknown_only and cdict.wform_isknown(wform,
                                                             tag=gold_pos)
                    ):
                        n += 1
                        c += gold_token['LEMMA'] == test_sent[j]['LEMMA']
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
                print(sp   + '   wrong: {}'.format(n - c), file=LOG_FILE)
                print(sp   + 'Accuracy: {}'.format(c / n if n > 0 else 1.),
                      file=LOG_FILE)
        return c / n if n > 0 else 1.

    def evaluate_feats(self, gold=None, test=None, joint=False, rev=False,
                       feat=None, unknown_only=False, silent=False):
        """Score the accuracy of the FEATS tagger against the *gold* standard.
        Remove feats (or only one specified feat) from the *gold* standard
        text, generate new feats using the tagger, then compute the accuracy
        score. If *test* is not None, compute the accuracy of the *test* corpus
        with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param joint: if True, use joint FEATS model; elsewise, use separate
                      models (default)
        :param rev: if True, use Reversed FEATS tagger instead of generic
                    straight one
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
        n = c = nt = ct = 0
        if gold is None:
            gold = self._test_corpus
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
        or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = self._get_corpus(gold, silent=True)
        if test:
            test = self._get_corpus(test, silent=True)
        header = '{}FEAT{}{}'.format('Reversed ' if rev else '',
                                     '' if feat else 'S',
                                     '-j' if joint else '',
                                     '<<{}>>'.format(feat) if feat else '')
        if not silent:
            print('Evaluate ' + header, file=LOG_FILE)
        progress_step = max(int(corpus_len / 60), 1000) \
                            if corpus_len else 1000
        progress_check_step = min(int(corpus_len / 100), 1000) \
                                  if corpus_len else 100
        cdict = self._cdict
        feat_vals = cdict.get_feats()
        i = -1
        for i, gold_sent in enumerate(gold):
            if not silent and not i % progress_check_step:
                print_progress(i, end_value=corpus_len, step=progress_step,
                               file=LOG_FILE)
            #gold_sent = self.predict_pos(gold_sent, rev=rev, inplace=False)
            test_sent = next(test) if test else \
                        self.predict_feats(gold_sent, joint=joint, rev=rev,
                                           feat=None if joint else feat,
                                           inplace=False)
            for j, gold_token in enumerate(gold_sent):
                wform, gold_pos = gold_token['FORM'], gold_token['UPOS']
                if wform and '-' not in gold_token['ID']:
                    if not (
                        unknown_only and cdict.wform_isknown(wform,
                                                             tag=gold_pos)
                    ):
                        nf = cf = 0
                        gold_feats = gold_token['FEATS']
                        test_feats = test_sent[j]['FEATS']
                        for feat_ in [feat] if feat else feat_vals:
                            gold_feat = gold_feats.get(feat_)
                            test_feat = test_feats.get(feat_)
                            if gold_feat or test_feat:
                                nf += 1
                                if gold_feat == test_feat:
                                    cf += 1
                        if nf > 0:
                            n += 1
                            c += nf == cf
                            nt += nf
                            ct += cf
        if not silent:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                print_progress(i + 1,
                               end_value=corpus_len if corpus_len else 0,
                               step=progress_step, file=LOG_FILE)
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {} tokens, {} tags'.format(n, nt),
                      file=LOG_FILE)
                print(sp   + ' correct: {} tokens, {} tags'.format(c, ct),
                      file=LOG_FILE)
                print(sp   + '   wrong: {} tokens, {} tags'
                                 .format(n - c, nt - ct),
                      file=LOG_FILE)
                print(sp   + 'Accuracy: {} / {}'
                                 .format( c / n  if n  > 0 else 1.,
                                         ct / nt if nt > 0 else 1.),
                          file=LOG_FILE)
        return c / n if n > 0 else 1., ct / nt if nt > 0 else 1.

    def evaluate(self, gold=None, test=None, pos_rev=False,
                 feats_joint=False, feats_rev=False, feat=None,
                 unknown_only=False, silent=False):
        """Score a joint accuracy of the all available taggers against the
        *gold* standard. Extract wforms from the *gold* standard text, retag it
        using all the taggers, then compute a joint accuracy score. If *test*
        is not None, compute the accuracy of the *test* corpus with respect to
        the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param pos_rev: if True, use Reversed POS tagger instead of generic
                        straight one
        :param feats_joint: if True, use joint FEATS model; elsewise, use
                            separate models (default)
        :param feats_rev: if True, use Reversed FEATS tagger instead of generic
                          straight one
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
        n = c = nt = ct = 0
        if gold is None:
            gold = self._test_corpus
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
        or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = self._get_corpus(gold, silent=True)
        if test:
            test = self._get_corpus(test, silent=True)
        feat_vals = self._cdict.get_feats()
        if not silent:
            print('Evaluate', file=LOG_FILE)
        progress_step = max(int(corpus_len / 60), 1000) \
                            if corpus_len else 1000
        progress_check_step = min(int(corpus_len / 100), 1000) \
                                  if corpus_len else 100
        cdict = self._cdict
        i = -1
        for i, gold_sent in enumerate(gold):
            if not silent and not i % progress_check_step:
                print_progress(i, end_value=corpus_len, step=progress_step,
                               file=LOG_FILE)
            test_sent = next(test) if test else \
                        self.predict(gold_sent, pos_rev=pos_rev,
                                     feats_joint=feats_joint,
                                     feats_rev=feats_rev, inplace=False)
            for j, gold_token in enumerate(gold_sent):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_pos = gold_token['UPOS']
                    if not (
                        unknown_only and cdict.wform_isknown(wform,
                                                             tag=gold_pos)
                    ):
                        test_token = test_sent[j]
                        nf = cf = 0
                        # LEMMA
                        nf += 1
                        cf += gold_token['LEMMA'] == test_token['LEMMA']
                        # UPOS
                        nf += 1
                        cf += gold_pos == test_token['UPOS']
                        # FEATS
                        gold_feats = gold_token['FEATS']
                        test_feats = test_token['FEATS']
                        for feat_ in [feat] if feat else feat_vals:
                            gold_feat = gold_feats.get(feat_)
                            test_feat = test_feats.get(feat_)
                            if gold_feat or test_feat:
                                nf += 1
                                if gold_feat == test_feat:
                                    cf += 1
                        if nf > 0:
                            n += 1
                            c += nf == cf
                            nt += nf
                            ct += cf
        res = c / n if n > 0 else 1., ct / nt if nt > 0 else 1.
        if not silent:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                print_progress(i + 1,
                               end_value=corpus_len if corpus_len else 0,
                               step=progress_step, file=LOG_FILE)
                print('   total: {} tokens, {} tags'.format(n, nt),
                      file=LOG_FILE)
                print(' correct: {} tokens, {} tags'.format(c, ct),
                      file=LOG_FILE)
                print('   wrong: {} tokens, {} tags'.format(n - c, nt - ct),
                      file=LOG_FILE)
                print('Accuracy: {} / {}'.format(res[0], res[1]),
                      file=LOG_FILE)
        return res

    def train_pos(self, rev=False, epochs=5, no_train_evals=True, seed=None,
                  dropout=None, context_dropout=None):
        """Train a POS tagger from ``self._train_corpus``.

        :param rev: if True, train Reversed POS tagger instead of generic
                    straight one
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
        :param context_dropout: a fraction of POS tags to be randomly replaced
                                after predict to random POS tags to prevent
                                overfitting
        :type context_dropout: float
        """
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)

        model = _AveragedPerceptron(default_class=cdict.most_common_tag())
        if rev:
            self._pos_rev_model = model
        else:
            self._pos_model = model

        header = '{}POS'.format('Reversed ' if rev else '')
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
                tokens_ = [[t[0], None] for t in tokens]
                if rev:
                    tokens = tokens[::-1]
                context = [t[0] for t in tokens]
                prev, prev2 = self.features.START

                max_i = len(tokens) - 1
                for i, (wform, pos) in enumerate(tokens):
                    i_ = max_i - i if rev else i
                    guess, coef = cdict.predict_tag(wform, isfirst=i_ == 0)
                    if self._guess_pos:
                        guess, coef = self._guess_pos(guess, coef, i_,
                                                      tokens_, cdict)
                    if guess is not None:
                        if guess == pos:
                            td2 += 1
                        else:
                            fd2 += 1
                    if guess is None or coef < 1.:
                        features = self.features.get_pos_features(
                            i, context, prev, prev2
                        )
                        guess = model.predict(
                            features, suggest=guess, suggest_coef=coef,
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

                    prev2 = prev
                    tokens_[i][1] = prev = \
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
                    self.evaluate_pos, {'rev': rev}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals, self.evaluate_pos,
            {'rev': rev}
        )

    def train_lemma(self, epochs=5, no_train_evals=True, seed=None):
        """Train a lemma tagger from ``self._train_corpus``.

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
        """
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)

        self._lemma_model = True
        # Yes, we do nothing :)

    def train_feats(self, joint=False, rev=False, feat=None, epochs=5,
                    no_train_evals=True, seed=None, dropout=None,
                    context_dropout=None):
        """Train FEATS taggers from ``self._train_corpus``.

        :param joint: if True, train joint FEATS model; elsewise, train
                      separate models (default)
        :param rev: if True, train Reversed FEATS tagger instead of generic
                    straight one
        :param feat: name of the feat to evaluate the tagger; if None
                     (default), then tagger will be evaluated for all feats
        :type feat: str
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
        :param context_dropout: a fraction of FEATS tags to be randomly replaced
                                after predict to random FEATS tags to prevent
                                overfitting
        :type context_dropout: float
        """
        return (
            self._train_feats_joint if joint else
            self._train_feats_separate
        )(
            rev=rev, feat=feat, epochs=epochs, no_train_evals=no_train_evals,
            seed=seed, dropout=dropout, context_dropout=context_dropout
        )

    def _train_feats_separate(self, rev=False, feat=None, epochs=5,
                              no_train_evals=True, seed=None, dropout=None,
                              context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)

        if feat:
            models = self._feats_rev_models if rev else self._feats_models
        else:
            models = {}
            if rev:
                self._feats_rev_models = models
            else:
                self._feats_models = models

        default_val = '_'
        feat_vals = cdict.get_feats()
        if feat:
            feat_vals = {feat: feat_vals[feat]}
        for feat in sorted(feat_vals):
            header = '{}FEAT<<{}>>'.format('Reversed ' if rev else '', feat)
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

                    #sentence = self.predict_pos(sentence, rev=rev, inplace=False)
                    tokens = [(x['FORM'], x['LEMMA'], x['UPOS'], x['FEATS'])
                                  for x in sentence
                                      if x['FORM'] and x['LEMMA'] and x['UPOS']
                                     and '-' not in x['ID']]
                    tokens_ = [[*t[:3], None] for t in tokens]
                    if rev:
                        tokens = tokens[::-1]
                    context, lemma_context, pos_context = \
                        [list(x) for x in zip(*[t[:3] for t in tokens])] \
                            if tokens else \
                        [[]] * 3
                    prev, prev2 = self.features.START

                    max_i = len(tokens) - 1
                    for i, (wform, lemma, pos, feats) in enumerate(tokens):
                        i_ = max_i - i if rev else i
                        gold_val = feats.get(feat, default_val)
                        guess, coef = cdict.predict_feat(feat,
                                                         wform, lemma, pos)
                        if self._guess_feat:
                            guess, coef = self._guess_feat(guess, coef, i_,
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
                            features = self.features.get_feat_features(
                                i, context, lemma_context, pos_context,
                                False, last_val_idx, prev, prev2
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

                        prev2 = prev
                        tokens_[i_][3] = prev = \
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
                        lambda **kwargs: self.evaluate_feats(**kwargs)[1],
                        {'joint': False, 'rev': rev, 'feat': feat}
                    )
                if eqs == -1:
                    break

            res = self._train_done(
                header, model, eqs, no_train_evals,
                lambda **kwargs: self.evaluate_feats(**kwargs)[1],
                {'joint': False, 'rev': rev, 'feat': feat}
            )

        return res if feat else \
               f_evaluate(joint=False, rev=rev, feat=feat, silent=True)

    def _train_feats_joint(self, rev=False, feat=None, epochs=5,
                           no_train_evals=True, seed=None, dropout=None,
                           context_dropout=None):
        cdict, corpus_len, progress_step, progress_check_step, \
                              epochs, epochs_ = self._train_init(epochs, seed)
        assert not feat, 'ERROR: feat must be None with joint=True'

        model = _AveragedPerceptron()
        if rev:
            self._feats_rev_model = model
        else:
            self._feats_model = model

        header = '{}FEATS'.format('Reversed ' if rev else '')
        vals, vals_len, vals_sorted = set(), 0, []
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
                              for x in (reversed(sentence) if rev else
                                        sentence)
                                  if x['FORM'] and x['LEMMA'] and x['UPOS']
                                 and '-' not in x['ID']]
                context, lemma_context, pos_context = \
                    [list(x) for x in zip(*[t[:3] for t in tokens])] \
                        if tokens else \
                    [[]] * 3
                prev, prev2 = self.features.START

                max_i = len(tokens) - 1
                for i, (_, _, _, feats) in enumerate(tokens):
                    gold = '|'.join('='.join((x, feats[x]))
                                        for x in sorted(feats))
                    features = self.features.get_feat_features(
                        i, context, lemma_context, pos_context,
                        True, 0, prev, prev2
                    )
                    guess = model.predict(features, dropout=dropout)
                    model.update(gold, guess, features)
                    n += 1
                    c += guess == gold

                    vals.add(gold)
                    if len(vals) > vals_len:
                        vals_sorted = sorted(vals)
                        vals_len = len(vals)
                    prev2 = prev
                    prev = \
                        guess if not context_dropout \
                              or rand() >= context_dropout else \
                        vals_sorted[randint(0, vals_len - 1)]

            print_progress(sent_no + 1, end_value=corpus_len,
                           step=progress_step)
            epoch, epochs, best_epoch, best_score, best_weights, \
                                                          eqs, bads, score = \
                self._train_eval(
                    model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    lambda **kwargs: self.evaluate_feats(**kwargs)[1],
                    {'joint': True, 'rev': rev}
                )
            if eqs == -1:
                break

        return self._train_done(
            header, model, eqs, no_train_evals,
            lambda **kwargs: self.evaluate_feats(**kwargs)[1],
            {'joint': True, 'rev': rev}
        )
