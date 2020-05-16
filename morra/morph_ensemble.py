# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
A tool that allows use several MorphParsers together in the ensemble.
"""
from copy import deepcopy

from corpuscula.corpus_utils import _AbstractCorpus
from corpuscula.utils import LOG_FILE, print_progress
from morra.base_parser import BaseParser

class MorphEnsemble:
    """A simple tool that allow to conjoin several MorphParsers to the
    ensemble"""

    def __init__(self, cdict):
        """
        :param cdict: corpuscula CorpusDict of the current training corpus
        """
        self._cdict = cdict
        self._predict_methods = []

    def add(self, predict_method, **kwargs):
        """Add parser to the ensemble.

        :param predict_method: a method of MorphParser that we prefer to use
                               for prediction
        :param kwargs: keyword arguments for *predict_method*
        :return: index of added method
        """
        index = len(self._predict_methods)
        self._predict_methods.append((predict_method, kwargs))
        return index

    def pop(self, index):
        """Remove parser from the ensemble.

        :param index: an index returned by the `.add()` method
        :return: the method removed
        """
        return self.models.pop(index)

    def predict(self, fields_to_predict, sentence, inplace=True):
        """Predict necessary fields of *sentences*.

        :param fields_to_predict: the list of CONLL-U fields names you want to
                                  get a prediction for. E.g.: ['UPOS',
                                  'LEMMA', 'FEATS]. Note, that these fields
                                  must be between the fields that the methods
                                  added with the `.add()` can predict
        :param sentence: the sentence in *Parsed CONLL-U* format
        :param inplace: if True (default), method changes and returns the given
                        *sentence** itself. Elsewise, the new sentence will be
                        created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        if isinstance(fields_to_predict, str):
            fields_to_predict = [fields_to_predict]
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        results = []
        for predict, kwargs in self._predict_methods:
            results.append((x for x in predict(deepcopy(sent), **kwargs)))
        results_len = len(results)

        for token in sent:
            for fld in fields_to_predict:
                res = {}
                if isinstance(token[fld], dict):
                    best_val, best_scores = {}, {}
                    for s in results:
                        t = next(s)
                        feats = t[fld]
                        for feat, val in feats.items():
                            score = res[feat][val] = \
                                res.setdefault(feat, {None: results_len}) \
                                   .get(val, 0) + 1
                            res[feat][None] -= 1
                            if score > best_scores.get(feat, (None, -1))[1]:
                                best_scores[feat] = (val, score)
                    for feat, scores in best_scores.items():
                        val, score = scores
                        if score >= res[feat][None]:
                            best_val[feat] = val
                else:
                    best_val, best_score = None, -1
                    for s in results:
                        t = next(s)
                        tag = t[fld]
                        score = res[tag] = res.setdefault(tag, 0) + 1
                        if score > best_score:
                            best_val, best_score = tag, score
                token[fld] = best_val
        return sentence

    def predict_sents(self, fields_to_predict, sentences, inplace=True,
                      save_to=None):
        """Apply ``self.predict()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :param inplace: if True, method changes and returns the given
                        sentences themselves; elsewise, new list of sentences
                        will be created
        :param save_to: if not None then the result will be saved to the file
                        with a specified name
        :type save_to: str
        """
        if isinstance(sentences, type) and issubclass(sentences,
                                                      _AbstractCorpus):
            sentences = sentences.test()
        assert sentences, 'ERROR: Sentences must not be empty'
        sentences = BaseParser._get_corpus(sentences, asis=True)
        sentences = self.predict(fields_to_predict, sentences, inplace=inplace)
        if save_to:
            self.save_conllu(sentences, save_to, silent=True)
            sentences = BaseParser._get_corpus(save_to, asis=True)
        return sentences

    def evaluate(self, fields_to_evaluate, gold=None, test=None,
                 feat=None, unknown_only=False, silent=False):
        """Score an accuracy of the ensemble tagging against the *gold*
        standard. Extract wforms from the *gold* standard text, retag it using
        the ensemble, then compute a joint accuracy score. If *test* is not
        None, compute the accuracy of the *test* corpus with respect to the
        gold.

        :param fields_to_evaluate: the list of CONLL-U fields names you want to
                                   evaluate a prediction for. E.g.: ['UPOS',
                                   'LEMMA', 'FEATS]. Note, that these fields
                                   must be between the fields that the methods
                                   added with the `.add()` can predict
        :param gold: a corpus of tagged sentences to score the tagger on.
        :param test: a corpus of tagged sentences to compare with *gold*
        :param feat: name of the feat to evaluate the ensemble; if None, then
                     the ensemble will be evaluated for all feats
        :type feat: str
        :param unknown_only: calculate accuracy score only for words that not
                             present in train corpus
        :param silent: suppress log
        :return: joint accuracy scores of the taggers against the gold:
                 1. by tokens: the tagging of the whole token may be either
                    correct or not
                 2. by tags: sum of correctly detected tags to sum of all tags
                    that are non-empty in either gold or retagged sentences
        :rtype: tuple(float, float)
        """
        n = c = nt = ct = 0
        if (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
        or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        assert gold, 'ERROR: Gold must not be empty'
        corpus_len = len(gold) if isinstance(gold, list) else None
        gold = BaseParser._get_corpus(gold, silent=True)
        if test:
            test = BaseParser._get_corpus(test, silent=True)
        if not silent:
            print('Evaluate', file=LOG_FILE)
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
            test_sent = next(test) if test else \
                        self.predict(fields_to_evaluate, gold_sent,
                                     inplace=False)
            for j, gold_token in enumerate(gold_sent):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_pos = gold_token['UPOS']
                    if not unknown_only or cdict.wform_isknown(wform,
                                                               tag=gold_pos):
                        test_token = test_sent[j]
                        nf = cf = 0
                        # LEMMA
                        if 'LEMMA' in fields_to_evaluate:
                            nf += 1
                            cf += gold_token['LEMMA'] == test_token['LEMMA']
                        # UPOS
                        if 'UPOS' in fields_to_evaluate:
                            nf += 1
                            cf += gold_pos == test_token['UPOS']
                        # FEATS
                        if 'FEATS' in fields_to_evaluate:
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
