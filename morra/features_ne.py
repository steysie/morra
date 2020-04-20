# -*- coding: utf-8 -*-
# Morra project: Features for MorphParserNE
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
If you need MorphParserNE to support your language, add feature-functions for
your language here. Then, create parser as:

    ``MorphParserNE(features='<your lang>')``
"""
from morra.base_features import BaseFeatures


class FeaturesNE(BaseFeatures):
    """Features for MorphParserNE"""

    def __init__(self, lang='RU'):
        super().__init__(lang=lang)
        if lang == 'RU':
            self.get_ne_features = self.get_ne_features_RU
            self.get_ne2_features = self.get_ne2_features_RU

    def get_ne_features_RU(self, i, context, lemma_context, pos_context,
                           feats_context, prev, prev2):
        prev  = str(prev)
        prev2 = str(prev2)

        context       = self.START +       context + self.END
        lemma_context = self.START + lemma_context + self.END
        pos_context   = self.START +   pos_context + self.END
        feats_context =    [{}, {}] + feats_context + [{}, {}]
        i += len(self.START)

        wform = context[i]
        wform_i  = self.normalize(wform)
        wform_b1 = self.normalize(context[i - 1])
        wform_b2 = self.normalize(context[i - 2])
        wform_f1 = self.normalize(context[i + 1])
        wform_f2 = self.normalize(context[i + 2])

        lemma_i  = lemma_context[i]
        lemma_b1 = lemma_context[i - 1]
        lemma_b2 = lemma_context[i - 2]
        lemma_f1 = lemma_context[i + 1]
        lemma_f2 = lemma_context[i + 2]

        pos_i  = pos_context[i]
        pos_b1 = pos_context[i - 1]
        pos_b2 = pos_context[i - 2]
        pos_f1 = pos_context[i + 1]
        pos_f2 = pos_context[i + 2]

        feats_i  = feats_context[i]
        feats_b1 = feats_context[i - 1]
        feats_b2 = feats_context[i - 2]
        feats_f1 = feats_context[i + 1]
        feats_f2 = feats_context[i + 2]

        features = self.init_features()

        self.add_feature(features, '_')
        self.add_feature(features, 'w', wform_i)
        self.add_feature(features, 's', self.wform_shape(lemma_i))
        len_ = len(wform)
        for val in range(1, 8):
            self.add_feature(features, 'p' + str(val),
                             '' if val > len_ else wform[:val])
        for val in range(1, 8):
            self.add_feature(features, 's' + str(val),
                             '' if val > len_ else wform[-val:])
        for val in range(3, 7):
            self.add_feature(features, 'S' + str(val),
                             '' if val > len_ else wform[val:])
        for val in range(3, 7):
            _len = len_ - 1 - val
            for start in range(1, _len):
                self.add_feature(features, '<' + str(val) + ':' + str(start),
                                 wform[len_ - start - val:len_ - start])
        self.add_feature(features, 'p', pos_i)
        #for feat, val in feats_i.items():
        #    self.add_feature(features, 'i feat', feat, val)
        #case = feats_i.get('Case')
        #if case:
        #    self.add_feature(features, 'i feat-Case', case)

        self.add_feature(features, '-1n', prev)
        self.add_feature(features, '-2n', prev2)
        self.add_feature(features, '-1n-2n', prev, prev2)
        self.add_feature(features, '-1nl', prev, lemma_i)

        self.add_feature(features, '-1l', lemma_b1)
        self.add_feature(features, '-1s4', wform_b1[-4:])
        self.add_feature(features, '-1p', pos_b1)

        self.add_feature(features, '+1l', lemma_f1)
        self.add_feature(features, '+1s3', wform_f1[-3:])
        self.add_feature(features, '+1p', pos_f1)

        self.add_feature(features, '-2l', lemma_b2)
        self.add_feature(features, '-2p', pos_b2)

        self.add_feature(features, '+2l', lemma_f2)
        self.add_feature(features, '+2p', pos_f2)
        return features

    def get_ne2_features_RU(self, i, context, lemma_context, pos_context,
                            feats_context, ne_context):
        context       = self.START +       context + self.END
        lemma_context = self.START + lemma_context + self.END
        pos_context   = self.START +   pos_context + self.END
        feats_context =    [{}, {}] + feats_context + [{}, {}]
        ne_context    = self.START +    ne_context + self.END
        i += len(self.START)

        wform = context[i]
        wform_i  = self.normalize(wform)
        wform_b1 = self.normalize(context[i - 1])
        wform_b2 = self.normalize(context[i - 2])
        wform_f1 = self.normalize(context[i + 1])
        wform_f2 = self.normalize(context[i + 2])

        lemma_i  = lemma_context[i]
        lemma_b1 = lemma_context[i - 1]
        lemma_b2 = lemma_context[i - 2]
        lemma_f1 = lemma_context[i + 1]
        lemma_f2 = lemma_context[i + 2]

        pos_i  = pos_context[i]
        pos_b1 = pos_context[i - 1]
        pos_b2 = pos_context[i - 2]
        pos_f1 = pos_context[i + 1]
        pos_f2 = pos_context[i + 2]

        feats_i  = feats_context[i]
        feats_b1 = feats_context[i - 1]
        feats_b2 = feats_context[i - 2]
        feats_f1 = feats_context[i + 1]
        feats_f2 = feats_context[i + 2]

        ne_i  = str(ne_context[i])
        ne_b1 = str(ne_context[i - 1])
        ne_b2 = str(ne_context[i - 2])
        ne_f1 = str(ne_context[i + 1])
        ne_f2 = str(ne_context[i + 2])

        features = self.init_features()

        self.add_feature(features, '_')
        self.add_feature(features, 'w', wform)
        self.add_feature(features, 's', self.wform_shape(wform))
        len_ = len(wform_i)
        for val in range(1, 8):
            self.add_feature(features, 'p' + str(val),
                             '' if val > len_ else wform_i[:val])
        for val in range(1, 8):
            self.add_feature(features, 's' + str(val),
                             '' if val > len_ else wform_i[-val:])
        for val in range(3, 7):
            _len = len_ - 1 - val
            for start in range(1, _len):
                self.add_feature(features, '<' + str(val) + ':' + str(start),
                                 wform_i[len_ - start - val:len_ - start]) 
        for val in range(2, 7):
            self.add_feature(features, 'S' + str(val),
                             '' if val > len_ else wform_i[val:])
        self.add_feature(features, 'p', pos_i)

        self.add_feature(features, '-1n', ne_b1)
        self.add_feature(features, '-2n', ne_b2)
        self.add_feature(features, '-1n-2n', ne_b1, ne_b2)
        self.add_feature(features, '-1nl', ne_b1, lemma_i)
        self.add_feature(features, '+1n', ne_f1)
        self.add_feature(features, '+2n', ne_f2)
        self.add_feature(features, '+1n+2n', ne_f1, ne_f2)
        self.add_feature(features, '+1nl', ne_f1, lemma_i)
        self.add_feature(features, '-1n+1n', ne_b1, ne_f1)

        self.add_feature(features, '-1l', lemma_b1)
        self.add_feature(features, '-1s4', wform_b1[-4:])
        self.add_feature(features, '-1p', pos_b1)

        self.add_feature(features, '+1l', lemma_f1)
        self.add_feature(features, '+1s3', wform_f1[-3:])
        self.add_feature(features, '+1p', pos_f1)

        self.add_feature(features, '-2l', lemma_b2)
        self.add_feature(features, '-2pos', pos_b2)

        self.add_feature(features, '+2l', lemma_f2)
        self.add_feature(features, '+2p', pos_f2)
        return features
