# -*- coding: utf-8 -*-
# Morra project: Morphological parser 3
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Get the results of join and separate FEATS-2 taggers and make adjusted
tagging on the ground of them. POS tagger did not change.
"""
from copy import deepcopy

from morra.morph_parser2 import MorphParser2


class MorphParser3(MorphParser2):

    def predict_feats3(self, sentence,
                       with_s_backoff=True, max_s_repeats=0,
                       with_j_backoff=True, max_j_repeats=0,
                       inplace=True):
        """Tag the *sentence* with the FEATS-3 tagger.

        :param sentence: sentence in Parsed CONLL-U format; UPOS and LEMMA
                         fields must be already filled
        :type sentence: list(dict)
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param with_s_backoff: if result of separate FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_feats2(joint=False)``
        :type max_s_repeats: int
        :param with_j_backoff: if result of joint FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_feats2(joint=True)``
        :type max_j_repeats: int
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        if not inplace:
            sentence = deepcopy(sentence)
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        sent = self.predict_feats2(sent, joint=False,
            with_backoff=with_s_backoff, max_repeats=max_s_repeats,
            inplace=True
        )
        sent_j = iter(self.predict_feats2(sent, joint=True,
            with_backoff=with_j_backoff, max_repeats=max_j_repeats,
            inplace=False
        ))
        for token in sent:
            feats = token['FEATS']
            feats_j = next(sent_j)['FEATS']
            for feat, val in feats_j.items():
                if feat not in feats:
                    feats[feat] = val
            del_feats = []
            for feat, val in feats.items():
                if feat not in feats_j:
                    del_feats.append(feat)
            for feat in del_feats:
                feats.pop(feat, None)
        return sentence

    def predict3(self, sentence, pos_backoff=True, pos_repeats=0,
                 feats_s_backoff=True, feats_s_repeats=0,
                 feats_j_backoff=True, feats_j_repeats=0, inplace=True):
        """Tag the *sentence* with the all available taggers.

        :param sentence: sentence in Parsed CONLL-U format
        :type sentence: list(dict)
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: parameter for ``predict_pos2()``
        :type pos_repeats: int
        :type feats_s_backoff: if result of separate FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_s_repeats: parameter for ``predict_feats3()``
        :type feats_s_repeats: int
        :type feats_j_backoff: if result of joint FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_j_repeats: parameter for ``predict_feats3()``
        :type feats_j_repeats: int
        :param inplace: if True, method changes and returns the given sentence
                        itself; elsewise, new sentence will be created
        :return: tagged *sentence* in Parsed CONLL-U format
        """
        return \
            self.predict_feats3(
                self.predict_lemma(
                    self.predict_pos2(
                        sentence, with_backoff=pos_backoff,
                        max_repeats=pos_repeats, inplace=inplace
                    ),
                    inplace=inplace
                ),
                with_s_backoff=feats_s_backoff, max_s_repeats=feats_s_repeats,
                with_j_backoff=feats_j_backoff, max_j_repeats=feats_j_repeats,
                inplace=inplace
           )

    def predict_feats3_sents(self, sentences=None,
                             with_s_backoff=True, max_s_repeats=0,
                             with_j_backoff=True, max_j_repeats=0,
                             inplace=True, save_to=None):
        """Apply ``self.predict_feats2()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :type with_s_backoff: if result of separate FEATS-2 tagger differs
                              from both its base taggers, get one of the bases
                              on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_feats3()``
        :type max_s_repeats: int
        :type with_j_backoff: if result of joint FEATS-2 tagger differs
                              from both its base taggers, get one of the bases
                              on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_feats3()``
        :type max_j_repeats: int
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
                (self.predict_feats3(s,
                                     with_s_backoff=with_s_backoff,
                                     max_s_repeats=max_s_repeats,
                                     with_j_backoff=with_j_backoff,
                                     max_j_repeats=max_j_repeats,
                                     inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def predict3_sents(self, sentences=None,
                       pos_backoff=True, pos_repeats=0,
                       feats_s_backoff=True, feats_s_repeats=0,
                       feats_j_backoff=True, feats_j_repeats=0,
                       inplace=True, save_to=None):
        """Apply ``self.predict2()`` to each element of *sentences*.

        :param sentences: a name of file in CONLL-U format or list/iterator of
                          sentences in Parsed CONLL-U. If None, then loaded
                          test corpus is used
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: parameter for ``predict3()``
        :type feats_s_backoff: if result of separate FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_s_repeats: parameter for ``predict3()``
        :type feats_s_repeats: int
        :type feats_j_backoff: if result of joint FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_j_repeats: parameter for ``predict3()``
        :type feats_j_repeats: int
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
                (self.predict3(s,
                               pos_backoff=pos_backoff,
                               pos_repeats=pos_repeats,
                               feats_s_backoff=feats_s_backoff,
                               feats_s_repeats=feats_s_repeats,
                               feats_j_backoff=feats_j_backoff,
                               feats_j_repeats=feats_j_repeats,
                               inplace=inplace)
                     for s in sentences),
            save_to=save_to
        )

    def evaluate_feats3(self, gold=None, test=None,
                        with_s_backoff=True, max_s_repeats=0,
                        with_j_backoff=True, max_j_repeats=0,
                        feat=None, unknown_only=False, silent=False):
        """Score the accuracy of the FEATS-2 tagger against the *gold*
        standard. Remove feats (or only one specified feat) from the *gold*
        standard text, generate new feats using the tagger, then compute the
        accuracy score. If *test* is not None, compute the accuracy of the
        *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param with_s_backoff: if result of separate FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_s_repeats: parameter for ``predict_feats3()``
        :type max_s_repeats: int
        :param with_j_backoff: if result of joint FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param max_j_repeats: parameter for ``predict_feats3()``
        :type max_j_repeats: int
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
            lambda sentence, rev=None, joint=None, feat=None, inplace=True: \
                self.predict_feats3(sentence,
                    with_s_backoff=with_s_backoff, max_s_repeats=max_s_repeats,
                    with_j_backoff=with_j_backoff, max_j_repeats=max_j_repeats,
                    inplace=inplace
                )
        res = self.evaluate_feats(gold=gold, test=test, feat=feat,
                                  unknown_only=unknown_only, silent=silent)
        self.predict_feats = f
        return res

    def evaluate3(self, gold=None, test=None,
                  pos_backoff=True, pos_repeats=0,
                  feats_s_backoff=True, feats_s_repeats=0,
                  feats_j_backoff=True, feats_j_repeats=0,
                  feat=None, unknown_only=False, silent=False):
        """Score a joint accuracy of the all available taggers against the
        *gold* standard. Extract wforms from the *gold* standard text, retag it
        using all the taggers, then compute a joint accuracy score.  If *test*
        is not None, compute the accuracy of the *test* corpus with respect to
        the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :type pos_backoff: if result of POS-2 tagger differs from both its
                           base taggers, get one of the bases on the ground
                           of some heuristics
        :param pos_repeats: parameter for ``predict3()``
        :type feats_s_backoff: if result of separate FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_s_repeats: parameter for ``predict3()``
        :type feats_s_repeats: int
        :type feats_j_backoff: if result of joint FEATS-2 tagger differs
                               from both its base taggers, get one of the bases
                               on the ground of some heuristics
        :param feats_j_repeats: parameter for ``predict3()``
        :type feats_j_repeats: int
        :param feat: name of the feat to evaluate the tagger; if None, then
                     tagger will be evaluated for all feats
        :type feat: str
        :param silent: suppress log
        :param unknown_only: calculate accuracy score only for words that are
                             not present in train corpus
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
                   feats_joint=None, feats_rev=None, inplace=False: \
                self.predict3(sentence,
                              pos_backoff=pos_backoff,
                              pos_repeats=pos_repeats,
                              feats_s_backoff=feats_s_backoff,
                              feats_s_repeats=feats_s_repeats,
                              feats_j_backoff=feats_j_backoff,
                              feats_j_repeats=feats_j_repeats,
                              inplace=inplace)
        res = self.evaluate(gold=gold, test=test, feat=feat,
                            unknown_only=unknown_only, silent=silent)
        self.predict = f
        return res
