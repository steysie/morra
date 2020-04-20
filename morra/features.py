# -*- coding: utf-8 -*-
# Morra project: Features for MorphParser
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
If you need MorphParser to support your language, add feature-functions for
your language here. Then, create parser as:

    ``MorphParser(features='<your lang>')``
"""
from morra.base_features import BaseFeatures


class Features(BaseFeatures):
    """Features for MorphParser"""

    def __init__(self, lang='RU'):
        super().__init__(lang=lang)
        if lang == 'RU':
            self.get_pos_features = self.get_pos_features_RU
            self.get_feat_features = self.get_feat_features_RU

    def get_pos_features_RU(self, i, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained"""
        if prev is None:
            prev = ''
        if prev2 is None:
            prev2 = ''

        context = self.START + context + self.END
        i += len(self.START)

        wform = context[i]
        wform_i  = self.normalize(wform)
        wform_b1 = self.normalize(context[i - 1])
        wform_b2 = self.normalize(context[i - 2])
        wform_f1 = self.normalize(context[i + 1])
        wform_f2 = self.normalize(context[i + 2])

        features = self.init_features()

        # It's useful to have a constant feature, which acts sort of like a prior
        self.add_feature(features, '_')  # bias
        self.add_feature(features, 'w', wform_i)
        self.add_feature(features, 's', self.wform_shape(wform))
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
        #for val in range(1, 8):
        #    self.add_feature(features, 'P' + str(val),
        #                     '' if val > len_ else wform[:-val])

        #for val in range(1, 8):
        #    self.add_feature(features, '>' + str(val), wform_i[val:val + 2])
        #for val in range(1, 8):
        #    self.add_feature(features, '<' + str(val), wform_i[-val - 2:-val])
        #for val in range(1, 8):
        #    _len = len_ - 1 - val
        #    for start in range(1, _len // 2 + 1):
        #        self.add_feature(features, '>' + str(val) + ':' + str(start),
        #                         wform[start:start + val])
        for val in range(3, 7):
            _len = len_ - 1 - val
            for start in range(1, _len):
                self.add_feature(features, '<' + str(val) + ':' + str(start),
                                 wform[len_ -start - val:len_ - start])
        # for val in range(3, 7):
        #     for start in range(1, len_ - 1 - val):
        #         self.add_feature(features, 'i' + str(val) + ':' + str(start),
        #                          wform[start:start + val])
        #for val in range(1, 8):
        #    self.add_feature(features, 'ii' + str(val), wform[val:-val])

        self.add_feature(features, '-1p', prev)
        self.add_feature(features, '-2p', prev2)
        self.add_feature(features, '-1p-2p', prev, prev2)
        self.add_feature(features, '-1pw', prev, wform_i)

        self.add_feature(features, '-1w', wform_b1)
        #self.add_feature(features, '-1s', self.wform_shape(context[i - 1]))
        #self.add_feature(features, '-1s3', wform_b1[-3:])
        self.add_feature(features, '-1s4', wform_b1[-4:])

        self.add_feature(features, '+1w', wform_f1)
        self.add_feature(features, '+1s3', wform_f1[-3:])
        #self.add_feature(features, '+1s4', wform_f1[-4:])

        self.add_feature(features, '-2w', wform_b2)

        self.add_feature(features, '+2w', wform_f2)
        return features

    def get_feat_features_RU(self, i, context, lemma_context, pos_context,
                             joint, val_cnt, prev, prev2):
        if prev is None:
            prev = ''
        if prev2 is None:
            prev2 = ''

        context       = self.START +       context + self.END
        lemma_context = self.START + lemma_context + self.END
        pos_context   = self.START +   pos_context + self.END
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

        if joint or val_cnt < 7:
            self.add_feature(features, '-1f', prev)
            self.add_feature(features, '-1fw', prev, wform_i)
            self.add_feature(features, '-1fl', prev, lemma_i)
            self.add_feature(features, '-1fp', prev, pos_i)
        else:
            self.add_feature(features, '-1f-2f', prev, prev2)
            self.add_feature(features, '-1fw', prev, wform_i)
            self.add_feature(features, '-f', prev if prev != '_' else prev2)
            self.add_feature(features, '-fw',
                             prev if prev != '_' else prev2, wform_i)

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
