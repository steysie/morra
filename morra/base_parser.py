# -*- coding: utf-8 -*-
# Morra project: Base parser
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Base classes for the project.
"""
from collections import OrderedDict, defaultdict
from copy import deepcopy
from math import isclose
import pickle
import random
from random import random as rand
from scipy.special import softmax
import sys

from corpuscula import Conllu, CorpusDict
from corpuscula.corpus_utils import _AbstractCorpus
from corpuscula.utils import LOG_FILE, print_progress


class _AveragedPerceptron:
    """This class is a port of the Textblob Averaged Perceptron Tagger
    Copyright 2013 Matthew Honnibal
    URL: <https://github.com/sloria/textblob-aptagger>

    See more implementation details here:
        https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

    Small changes and extra comments where added"""

    def __init__(self, default_class=None):
        """
        :param default_class: the most common class of the train set;
                              uses only on the first train iterations
        :type default_class: str
        """
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0
        # If None than you must be ready to handle the None class in the
        # beginning of training, when model has no classes at all (e.g.,
        # you can ignore None class or replace it with smth meaningful)
        self.default_class = default_class

    def predict(self, features, suggest=None, suggest_coef=None,
                with_score=False, dropout=0):
        """Dot-product the features and current weights and return the best
        label.

        :type suggest: str
        :type suggest_coef: float
        Suggestion and relevance coef of that suggestion given from some other
        tagger. May be omitted

        :param with_score: return not only most probable class but also its score
        :param dropout: a fraction of weiths to be randomly set to 0 at each
                        predict. Use it during training time to prevent
                        overfitting
        :type dropout: float
        :rtype: str | if with_score: tuple(str, float)
        """
        scores = defaultdict(float)
        for feat, value in features.items() if isinstance(features,
                                                          OrderedDict) else \
                           sorted(features.items()):
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                if not dropout or rand() >= dropout:
                    scores[label] += value * weight# / (1 - dropout)
        classes = scores.keys()
        if classes:
            # Increase scores of the suggested class by the relevance coef
            # of the suggestion
            if suggest:
                scores[suggest] *= 1 + suggest_coef
            # Round all scores off to avoid inaccuracy of floating point
            # operations
            scores = defaultdict(float, zip(
                classes, map(lambda x: round(x, 3), scores.values())
            ))
            # Do a secondary alphabetic sort, for stability
            res = max(classes, key=lambda label: (scores[label], label))
        else:
            res = self.default_class
        if with_score:
            scores = list(zip(*scores.items()))
            res = res, dict(zip(scores[0], softmax(scores[1])))[res] \
                           if scores else \
                       None
        return res

    def update(self, truth, guess, features):
        """Update the feature weights"""
        # self.weights: {feature: {class: weight}}
        def upd_feat(c, f, w, v):
            # param = (feature, class)
            param = (f, c)
            # для каждой пары "фича-класс" сохраняем:
            #     сумму весов на всех предыдущих итерациях
            self._totals[param] += (self.i - self._tstamps[param]) * w
            #     номер итерации, на которой вес менялся последний раз
            self._tstamps[param] = self.i
            # изменяем вес
            self.weights[f][c] = w + v

        # увеличиваем номер итерации
        self.i += 1
        if truth == guess:
            return
        # обновляем веса, только если угадали неправильно
        for f in features:
            # если весов ещё нет, но делаем нулевые веса для каждой фичи,
            # и после сразу их меняем
            weights = self.weights.setdefault(f, OrderedDict())
            # для верного класса добавляем 1
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            # для ошибочного вычитаем 1
            if guess is not None:
                upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        """Average weights from all iterations"""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                # добавляем к сумме весов актуальный вес с учётом кол-ва
                # итерация, которые он продержался
                total += (self.i - self._tstamps[param]) * weight
                # усредняем вес по всем итерациям
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def get_classes(self):
        """Classes found during train process"""
        return set(
            [x for x in self.weights.values() for x in x.keys()]
                if self.weights else
            []
        )


class BaseParser:
    """Base class for all parsers of the project"""

    def __init__(self):
        self._cdict = CorpusDict(log_file=LOG_FILE)

        self._train_corpus = []
        self._test_corpus  = []

    def backup(self):
        """Get current state"""
        return {'cdict_backup': self._cdict.backup()}

    def restore(self, o):
        """Restore current state from backup object"""
        cdict_backup = o.get('cdict_backup')
        if cdict_backup:
            self._cdict.restore(cdict_backup)

    def save(self, file_path):
        print('Saving model...', end=' ', file=LOG_FILE)
        LOG_FILE.flush()
        with open(file_path, 'wb') as f:
            pickle.dump(self.backup(), f, 2)
            print('done.', file=LOG_FILE)

    def load(self, file_path):
        print('Loading model...', end=' ', file=LOG_FILE)
        LOG_FILE.flush()
        with open(file_path, 'rb') as f:
            o = pickle.load(f)
            print('done.', file=LOG_FILE)
            self.restore(o)

    def _save_cdict(self, file_path):
        self._cdict.backup_to(file_path)

    def _load_cdict(self, file_path):
        self._cdict.restore_from(file_path)

    @staticmethod
    def load_conllu(*args, **kwargs):
        """Wrapper for ``Conllu.load()``"""
        silent = kwargs.pop('silent', None)
        return Conllu.load(*args, **kwargs,
                           log_file=None if silent else LOG_FILE)

    @staticmethod
    def save_conllu(*args, **kwargs):
        """Wrapper for ``Conllu.save()``"""
        silent = kwargs.pop('silent', None)
        return Conllu.save(*args, **kwargs,
                           log_file=None if silent else LOG_FILE)

    @staticmethod
    def split_corpus(corpus, split=[.8, .1, .1], save_split_to=None,
                     seed=None, silent=False):
        """Split a *corpus* in the given proportion.

        :param corpus: a name of file in CONLL-U format or list/iterator of
                       sentences in Parsed CONLL-U
        :param split: list of sizes of the necessary *corpus* parts. If values
                      are of int type, they are interpreted as lengths of new
                      corpuses in sentences; if values are float, they are
                      proportions of a given *corpus*. The types of the
                      *split* values can't be mixed: they are either all int,
                      or all float. The sum of float values must be less or
                      equals to 1; the sum of int values can't be greater than
                      the lentgh of the *corpus*
        :param save_split_to: list of file names to save the result of the
                              *corpus* splitting. Can be `None` (default;
                              don't save parts to files) or its length must be
                              equal to the length of *split*
        :param silent: if True, suppress output
        :return: a list of new corpuses
        """
        assert save_split_to is None or len(save_split_to) == len(split), \
               'ERROR: lengths of split and save_split_to must be equal'
        isfloat = len([x for x in split if isinstance(x, float)]) > 0
        if isfloat:
            assert sum(split) <= 1, \
                   "ERROR: sum of split can't be greater that 1"
        corpus = list(Conllu.load(corpus,
                                  log_file=None if silent else LOG_FILE))
        corpus_len = len(corpus)
        if isfloat:
            split = list(map(lambda x: round(corpus_len * x), split))
            diff = corpus_len - sum(split)
            if abs(diff) == 1:
                split[-1] += diff
        assert sum(split) <= corpus_len, \
               "ERROR: sum of split can't be greater that corpus length"
        random.seed(seed)
        random.shuffle(corpus)
        res = []
        pos_b = 0
        for i, sp in enumerate(split):
            pos_e = pos_b + sp
            corpus_ = corpus[pos_b:pos_e]
            pos_b = pos_e
            if save_split_to:
                Conllu.save(corpus_, save_split_to[i])
            res.append(corpus_)
        return res

    @classmethod
    def _get_corpus(cls, corpus, asis=False, silent=False):
        if isinstance(corpus, str):
            corpus = cls.load_conllu(corpus, silent=silent)
        return (s[0] if not asis and isinstance(s, tuple) else s
                   for s in corpus)

    def _predict_sents(self, sentences, predict_method, save_to=None):
        silent_save = False
        if sentences is None:
            sentences = deepcopy(self._test_corpus)
            slient_save = True
        elif isinstance(sentences, type) and issubclass(sentences,
                                                        _AbstractCorpus):
            sentences = sentences.test()
        assert sentences, 'ERROR: Sentences must not be empty'
        sentences = self._get_corpus(sentences, asis=True)
        sentences = predict_method(sentences)
        if save_to:
            self.save_conllu(sentences, save_to, silent=silent_save)
            sentences = self._get_corpus(save_to, asis=True)
        return sentences

    def parse_train_corpus(self, cnt_thresh=None, ambiguity_thresh=None):
        """Create a CorpusDict for train corpus(es) loaded. For one instance it
        may be used only once. Use ``load_train_corpus()`` with append=True
        to append one more corpus to the CorpusDict after it's created.

        :type cnt_thresh: int
        :type ambiguity_thresh: float
        Params for ``CorpusDict.fit()`` method. If omitted then default values
        will be used
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        print('Train:', end=' ', file=LOG_FILE)
        kwargs = {}
        if cnt_thresh is not None:
            kwargs['cnt_thresh'] = cnt_thresh
        if ambiguity_thresh is not None:
            kwargs['ambiguity_thresh'] = ambiguity_thresh
        self._cdict = CorpusDict(corpus=self._train_corpus,
                                 format='conllu_parsed',
                                 **kwargs, log_file=LOG_FILE)

    def load_train_corpus(self, corpus, append=False, parse=False, test=None,
                          seed=None):
        """Load train corpus and (possibly) create a CorpusDict for it

        :param corpus: a name of file in CONLL-U format or list/iterator of
                       sentences in Parsed CONLL-U
        :param append: add corpus to already loaded one(s)
        :param parse: extract corpus statistics to CorpusDict right after
                      loading
        :param test: if not None, then train corpus will be shuffled and
                     specified part of it stored as test corpus
        :type test: float
        :param seed: init value for the random number generator. Used only 
                     if test is not None
        :type seed: int
        """
        assert append or not self._train_corpus, \
               'ERROR: Train corpus is already loaded. Use append=True to ' \
               'append one more corpus'

        print('Train:', end=' ', file=LOG_FILE)
        if (isinstance(corpus, type) and issubclass(corpus, _AbstractCorpus)) \
        or isinstance(corpus, _AbstractCorpus):
            corpus = corpus.train()
        corpus = self._get_corpus(corpus)
        if test is not None:
            corpus = list(corpus)
            assert test >= 0 and test <= 1, \
                   'ERROR: A value of "test" parameter must be between 0 and 1'
            test_len = round(len(corpus) * test)
            if test_len == 0:
                print('WARNING: All the corpus given will be saved as train ' \
                      'corpus (parameter "test" is too small)', file=LOG_FILE)
            else:
                if test_len == len(corpus):
                    print('WARNING: All the corpus given will be saved as ' \
                          'test corpus (parameter "test" is too large)',
                          file=LOG_FILE)
                    self.load_test_corpus(corpus, append=append)
                    corpus = None
                else:
                    random.seed(seed)
                    random.shuffle(corpus)
                    self.load_test_corpus(corpus[:test_len], append=append)
                    corpus = corpus[test_len:]
                print('stored.', file=LOG_FILE)
        if corpus:
            if append == False:
                self._train_corpus = []
            self._train_corpus.extend(corpus)
            if parse:
                self._cdict = CorpusDict(corpus=self._train_corpus,
                                         format='conllu_parsed',
                                         log_file=LOG_FILE)

    def load_test_corpus(self, corpus, append=False):
        """Load development test corpus to validate on during training
        iterations.

        :param corpus: a name of file in CONLL-U format or list/iterator of
                       sentences in Parsed CONLL-U
        :param append: add corpus to already loaded one(s)
        """
        assert append or not self._test_corpus, \
               'ERROR: Test corpus is already loaded. Use append=True to ' \
               'append one more corpus'

        print('Test:', end=' ', file=LOG_FILE)
        if (isinstance(corpus, type) and issubclass(corpus, _AbstractCorpus)) \
        or isinstance(corpus, _AbstractCorpus):
            try:
                corpus = corpus.dev()
            except ValueError:
                try:
                    corpus = corpus.test()
                except ValueError:
                    raise ValueError(('ERROR: {} does not have a test part. '
                                      'Use "test" attribute of '
                                      'load_train_corpus() instead')
                                         .format(corpus.name))
        corpus = self._get_corpus(corpus)
        if append == False:
            self._test_corpus = []
        self._test_corpus.extend(corpus)

    def remove_rare_feats(self, abs_thresh=None, rel_thresh=None,
                          full_rel_thresh=None):
        """Remove feats from train and test corpora, occurence of which
        in the train corpus is less then a threshold.

        :param abs_thresh: remove features if their count in the train corpus
                           is less than this value
        :type abs_thresh: int
        :param rel_thresh: remove features if their frequency with respect to
                           total feats count of the train corpus is less than
                           this value
        :type rel_thresh: float
        :param full_rel_thresh: remove features if their frequency with respect
                                to the full count of the tokens of the train
                                corpus is less than this value
        :type full_rell_thresh: float

        *rell_thresh* and *full_rell_thresh* must be between 0 and 1"""
        corpus_len = len(self._train_corpus)
        progress_step = max(int(corpus_len / 60), 1000)
        progress_check_step = min(int(corpus_len / 100), 1000)

        print('Search rare feats...', file=LOG_FILE)
        feat_freq = {}
        token_cnt = 0
        sent_no = -1
        for sent_no, sentence in enumerate(self._train_corpus):
            if not sent_no % progress_check_step:
                print_progress(sent_no, end_value=corpus_len,
                               step=progress_step)
            for token in sentence:
                token_cnt += 1
                feats = token['FEATS']
                for feat, _ in feats.items():
                    feat_freq[feat] = feat_freq.setdefault(feat, 0) + 1
        print_progress(sent_no + 1, end_value=corpus_len, step=progress_step,
                       file=LOG_FILE)
        total_cnt = sum(feat_freq.values())
        feats_to_remove = sorted(
            feat for feat, cnt in feat_freq.items()
                if (abs_thresh and cnt < abs_thresh)
                or (rel_thresh and cnt / total_cnt < rel_thresh)
                or (full_rel_thresh and cnt / token_cnt < rull_rel_thresh)
        )

        if not feats_to_remove:
            print('Finished: Nothing to do.', file=LOG_FILE)
        else:
            print('Rare feats:', feats_to_remove, file=LOG_FILE)
            train_removes = test_removes = 0
            for i, corpus in enumerate([self._train_corpus,
                                        self._test_corpus]):
                corpus_len = len(corpus)
                progress_step = max(int(corpus_len / 60), 1000)
                progress_check_step = min(int(corpus_len / 100), 1000)

                sent_no = -1
                for sent_no, sentence in enumerate(corpus):
                    if not sent_no % progress_check_step:
                        print_progress(sent_no, end_value=corpus_len,
                                       step=progress_step)
                    for token in sentence:
                        feats = token['FEATS']
                        for feat in feats_to_remove:
                            val = feats.pop(feat, None)
                            if val:
                                if i:
                                    test_removes += 1
                                else:
                                    train_removes += 1
                print_progress(sent_no + 1, end_value=corpus_len,
                               step=progress_step, file=LOG_FILE)
            print(('Finished: {} feats removes from the train corpus, '
                   '{} from the test corpus')
                      .format(train_removes, test_removes),
                  file=LOG_FILE)

    def _train_init(self, epochs, seed, allow_empty_cdict=False):
        cdict = self._cdict
        assert self._train_corpus, 'ERROR: Train corpus is not loaded'
        if not allow_empty_cdict:
            assert not cdict.isempty(), \
                   'ERROR: Train corpus is not yet prepared'
        epochs, epochs_ = epochs if isinstance(epochs, tuple) else \
                          (epochs, 0) if epochs >= 0 else \
                          (1, epochs)
        if epochs == 0:
            epochs = 1
        if epochs_ > 0:
            epochs = -epochs_
        assert epochs_ != 0 or self._test_corpus, \
               'ERROR: epochs < 0 may be used only with a test corpus'
        corpus_len = len(self._train_corpus)
        progress_step = max(int(corpus_len / 60), 1000)
        progress_check_step = min(int(corpus_len / 100), 1000)
        random.seed(seed)
        return cdict, corpus_len, progress_step, progress_check_step, \
                                                               epochs, epochs_

    def _train_eval(self, model, epoch, epochs, epochs_,
                    best_epoch, best_score, best_weights,
                    eqs, bads, prev_score,
                    td, fd, td2, fd2, tp, fp, c, n, no_train_evals,
                    f_evaluate, f_evaluate_args):
        if td is not None:
            print('dict   : correct: {}, wrong: {}, accuracy: {}'
                      .format(td, fd, round(td / (td + fd), 4)
                                          if td + fd > 0 else 1.),
                  file=LOG_FILE)
        if td2 is not None:
            print('dict_s : correct: {}, wrong: {}, accuracy: {}'
                      .format(td2, fd2, round(td2 / (td2 + fd2), 4)
                                            if td2 + fd2 > 0 else 1.),
                  file=LOG_FILE)
        if tp is not None:
            print('predict: correct: {}, wrong: {}, accuracy: {}'
                      .format(tp, fp, round(tp / (tp + fp), 4)
                                          if tp + fp > 0 else 1.),
                  file=LOG_FILE)
        sp1 = ' ' * (len(str(c)) + len(str(n)))
        sp2 = ' ' * (8 + len(str(tp)) + len(str(fp)) - len(sp1))
        print('accuracy: train during train:' + sp2 + '{}/{}={}'
                  .format(c, n, round(c / n, 4) if n > 0 else 1.),
              file=LOG_FILE)
        if not no_train_evals:
            print(' ' * 10 + 'train after train :  ' + sp1 + sp2
                + str(round(f_evaluate(gold=self._train_corpus, silent=True,
                                       **f_evaluate_args), 4)),
                  file=LOG_FILE)
        if self._test_corpus:
            weights = deepcopy(model.weights)
            model.average_weights()
            score = f_evaluate(silent=True, **f_evaluate_args)
            iseq = isclose(score, best_score)
            if score > best_score:
                bads = 0
            elif score <= prev_score:
                bads += 1
            print('Effective test accuracy: {} {}{}'
                      .format(score, '==' if iseq else
                                     '>>' if score > best_score else
                                     '<< >' if score > prev_score else
                                     '<< =' if score == prev_score else
                                     '<< <',
                              ' ({})'.format(bads) if bads else ''),
                  file=LOG_FILE)
            if iseq:
                eqs += 1
            else:
                eqs = 0
                if score > best_score:
                    best_epoch, best_score, best_weights = \
                        epoch, score, model.weights
            if epochs_ and epoch + 1 == epochs:
                epochs = epochs_
            if epoch + 1 == epochs or (epochs < 0
                                   and bads >= -epochs):#best_epoch - epoch <= epochs):
                if eqs != epoch - best_epoch:
                    print('Search finished. Return to Epoch', best_epoch,
                          file=LOG_FILE)
                    model.weights = best_weights
                else:
                    print('Search finished', file=LOG_FILE)
                eqs = -1
            else:
                model.weights = weights
        else:
            score = prev_score
        return epoch + 1, epochs, best_epoch, best_score, best_weights, \
               eqs, bads, score

    def _train_done(self, header, model, eqs, no_train_evals,
                    f_evaluate, f_evaluate_args):
        if eqs >= 0:
            model.average_weights()
        res = None
        if not no_train_evals or self._test_corpus:
            print('Final {} {}{}{} accuracy:'
                      .format(header,
                              'train' if not no_train_evals else '',
                              '/' if not no_train_evals
                                 and self._test_corpus else '',
                              'test' if self._test_corpus else ''),
                  end=' ', file=LOG_FILE)
            if not no_train_evals:
                print(round(f_evaluate(gold=self._train_corpus, silent=True,
                                       **f_evaluate_args), 4),
                      end='', file=LOG_FILE)
            if not no_train_evals and self._test_corpus:
                print(' / ', end='', file=LOG_FILE)
            if self._test_corpus:
                res = f_evaluate(silent=True, **f_evaluate_args)
                print(round(res, 4), end='', file=LOG_FILE)
            print(file=LOG_FILE)
        return res

results = []
def autotrain(train_func, *args, silent=False,
              backup_model_func=None, restore_model_func=None,
              reload_trainset_func=None, fit_params={}, params_in_process=None,
              **kwargs):
    """
    This is a tool for models' hyperparameters selection. It made simple,
    without any multiprocessing, so it's just save time of user from cyclical
    rerunnings the training process with new parameters, but it's not speedup
    the training. Howewer, user can simultaneously run several versions of
    `autotrain()` with different hyperparameters' combinations.

    :param train_func: method to train the model.

    :param args: positional args for *train_func*. Will be passed as is.

    :param silent: if True, suppress the output.

    :param backup_model_func: method to backup the model internal state. If not
                              None, it will be used to save the state of the
                              model with the highest score, for that we don't
                              train it anew

    :param restore_model_func: method to restore the model internal state. If
                               not None, then in the end, this method will be
                               invoked with the only param: internal state 
                               of the best model. There is no use or even may
                               be cause of runtime error, to specify
                               *restore_model_func* without
                               *backup_model_func*

    :param reload_trainset_func: method to load the training corpus anew. It
                                 may become necessary if you want to achieve
                                 complete reproducibility. The training corpus
                                 is randomly shuffled before each training
                                 iteration, so you should reload it before each
                                 new training if you want to have possibility
                                 to repeat the training with the best
                                 parameters and get the model with exactly the
                                 same internal state

    :param fit_params: dict of hyperparameters' keys and lists of their values
                       among which we want to find the best. It is not very
                       flexible. We just check combinations of all the given
                       values of all the given hyperparameters.
                       E.g.:
                           fit_params={
                               'dropout': [None, .005, .01],
                               'context_dropout': [None, .05, .1]
                           }
                       produces next additional keyword args for *train_func*:
                           {'dropout': None, 'context_dropout': None}
                           {'dropout': None, 'context_dropout': .05}
                           {'dropout': None, 'context_dropout': .1}
                           {'dropout': .005, 'context_dropout': None}
                           {'dropout': .005, 'context_dropout': .05}
                           {'dropout': .005, 'context_dropout': .1}
                           {'dropout': .01, 'context_dropout': None}
                           {'dropout': .01, 'context_dropout': .05}
                           {'dropout': .01, 'context_dropout': .1}

    :param params_in_process: for internal use only. Leave it as it is

    :param kwargs: keyword args for *train_func*. Will be passed as is

    When all training have done, the tool returns
    `tuple(<best model score>, <best model params>, <best model internal state>)`.
    If *backup_model_func* is `None`, `<best model internal state>` will be 
    `None`, too

    Also, `<best model internal state>` will be loaded to your model, if
    *reload_trainset_func* is not `None`.
    """
    global results
    if not silent:
        print('AUTOTRAIN model STARTED', file=LOG_FILE)
        print('Fixed params:', kwargs, file=LOG_FILE)
        print('Fitting params:', fit_params, file=LOG_FILE)
        results = []
    _kwargs = deepcopy(kwargs)
    if not params_in_process:
        params_in_process = {}
    if fit_params:
        _fit_params = deepcopy(fit_params)
        kwarg = list(_fit_params.keys())[0]
        vals = _fit_params.pop(kwarg)
        best_res, best_kwargs, best_model = -1, None, None
        if isinstance(vals, list) or isinstance(vals, tuple):
            for val in vals:
                _kwargs[kwarg] = params_in_process[kwarg] = val
                res = autotrain(train_func, *args, silent=True,
                                backup_model_func=backup_model_func,
                                reload_trainset_func=reload_trainset_func,
                                fit_params=_fit_params,
                                params_in_process=params_in_process,
                                **_kwargs)
                if res[0] > best_res:
                    best_res, best_kwargs, best_model = res
                    best_kwargs[kwarg] = val
        elif isinstance(vals, int):
            raise 'ERROR: processing int values of fit_params ' \
                  'is not implemented yet'
        elif isinstance(vals, float):
            raise 'ERROR: processing float values of fit_params ' \
                  'is not implemented yet'
        else:
            raise 'ERROR: fit_params can contain only values of ' \
                  'list, int or float'
        res = best_res, best_kwargs, best_model
    else:
        print(file=LOG_FILE)
        print('Train params:', kwargs, file=LOG_FILE)
        if reload_trainset_func:
            reload_trainset_func()
        res = train_func(*args, **kwargs), {}, backup_model_func() \
                                                   if backup_model_func else \
                                               None
        results.append((res[0], deepcopy(params_in_process)))
    if restore_model_func:
        restore_model_func(res[2])
    if not silent:
        print('AUTOTRAIN model FINISHED', file=LOG_FILE)
        if results:
            print('Results:')
            for res_ in sorted(results, key=lambda x: (1 - x[0], str(x[1]))):
                print(res_)
            print('Optimal fitting params:', res[1], file=LOG_FILE)
        print('Train quality:', res[0], file=LOG_FILE)
        print(file=LOG_FILE)
    return res

def _evaluate(test_corpus, gold_corpus, what=None, feat=None, silent=False):
    """Score a joint accuracy of the all available tagger against the
    gold standard. Extract wforms from the gold standard text, retag them
    using all the taggers, then compute a joint accuracy score.

    :param gold_corpus: a corpus of gold sentences
    :param test_corpus: a corpus of retagged sentences from the gold corpus
    :param what: 'pos'|'lemma'|'feats': what you want to evaluate.
                 if *what* is None (default) then a joint accuracy will be
                 calculated
    :type what: str
    :param feat: name of the exact feat, tagging of that you want to evaluate.
                 Used only if *what* == 'feat'. If None (default) then tagger
                 will be evaluated for all feats
    :param silent: if True then the result will not be visualized
    :return: accuracy scores of the tags in the test corpus against the gold:
             1. by tokens: the tagging of the whole token may be either
                correct or not
             2. by tags: sum of correctly detected tags to sum of all tags that
                is non-empty in either gold or test sentences (visualized only
                if what is None or if what == 'feats' and feat is None
    :rtype: tuple(float, float)
    """
    if what:
        what = what.lower()
    assert what in [None, 'lemma', 'upos', 'feats']
    gold_corpus = list(MorphParser._get_corpus(gold_corpus))
    test_corpus = list(MorphParser._get_corpus(test_corpus))
    feat_vals = {}
    for corpus in [gold_corpus, test_corpus]:
        for sent in corpus:
            if isinstance(sent, tuple):
                sent = sent[0]
            for token in sent:
                for feat, feat_val in token['FEATS'].items():
                    feat_vals.setdefault(feat, set()).add(feat_val)
    if not silent:
        print('Evaluate', file=LOG_FILE)
    n = c = nt = ct = 0
    i = -1
    for i, gold_sent in enumerate(gold_corpus):
        if not silent and not i % 100:
            print_progress(i, end_value=None, step=1000, file=LOG_FILE)
        if isinstance(gold_sent, tuple):
            gold_sent = gold_sent[0]
        test_sent = test_corpus[i]
        if isinstance(test_sent, tuple):
            test_sent = test_sent[0]
        for j, gold_token in enumerate(gold_sent):
            test_token = test_sent[j]
            nf = cf = 0
            # LEMMA
            if what is None or what == 'lemma':
                if gold_token['FORM'] and '-' not in gold_token['ID']:
                    nf += 1
                    cf += gold_token['LEMMA'] == test_token['LEMMA']
            # UPOS
            if what is None or what == 'upos':
                if gold_token['FORM'] and '-' not in gold_token['ID']:
                    nf += 1
                    cf += gold_token['UPOS'] == test_token['UPOS']
            # FEATS
            if what is None or what == 'feats':
                if gold_token['FORM'] and gold_token['LEMMA'] \
               and gold_token['UPOS'] and '-' not in gold_token['ID']:
                    gold_feats = gold_token['FEATS']
                    test_feats = test_token['FEATS']
                    for feat_ in [feat] if what == 'feats' and feat else \
                                 feat_vals:
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
            print_progress(i + 1, end_value=0, step=1000, file=LOG_FILE)
            print('   total: {} tokens, {} tags'.format(n, nt),
                  file=LOG_FILE)
            print(' correct: {} tokens, {} tags'.format(c, ct),
                  file=LOG_FILE)
            print('   wrong: {} tokens, {} tags'.format(n - c, nt - ct),
                  file=LOG_FILE)
            print('Accuracy: {} / {}'.format(res[0], res[1]),
                  file=LOG_FILE)
    return res
