# -*- coding: utf-8 -*-
# Morra project: Features for MorphParser2
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
If you need MorphParser2 to support your language, add feature-functions for
your language here. Then, create parser as:

    ``MorphParser2(features='<your lang>')``
"""
from morra.features import Features


class Features2(Features):
    """Features for MorphParser2"""

    def __init__(self, lang='RU'):
        super().__init__(lang=lang)
        if lang == 'RU':
            self.get_pos2_features = self.get_pos2_features_RU
            self.get_feat2_features = self.get_feat2_features_RU

    def get_pos2_features_RU(self, i, context, pos_context):
        context     = self.START + context     + self.END
        pos_context = self.START + pos_context + self.END
        i += len(self.START)

        wform = context[i]
        wform_i  = self.normalize(wform)
        wform_b1 = self.normalize(context[i - 1])
        wform_b2 = self.normalize(context[i - 2])
        wform_f1 = self.normalize(context[i + 1])
        wform_f2 = self.normalize(context[i + 2])

        #pos_i  = pos_context[i]
        pos_b1 = pos_context[i - 1]
        pos_b2 = pos_context[i - 2]
        #pos_b3 = pos_context[i - 3] if i >= 3 else '-START3-'
        pos_f1 = pos_context[i + 1]
        pos_f2 = pos_context[i + 2]
        #pos_f3 = pos_context[i + 3] if i < len(pos_context) - 3 else '-END3-'

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
        #for val in range(1, 8):
        #    self.add_feature(features, 'P' + str(val),
        #                     '' if val > len_ else wform_i[:-val])

        #for val in range(3, 7):
        #    for start in range(1, len_ - 1 - val):
        #        self.add_feature(features, 'i' + str(val) + ':' + str(start),
        #                         wform_i[start:start + val])
        #for val in range(2, 8):
        #    self.add_feature(features, 'I' + str(val), wform[val:-val])
        ##self.add_feature(features, '-3p', pos_b3)
        ##self.add_feature(features, '+3p', pos_f3)

        self.add_feature(features, '-1p', pos_b1)
        self.add_feature(features, '-2p', pos_b2)
        self.add_feature(features, '-1p-2p', pos_b1, pos_b2)
        #self.add_feature(features, '-1p-2p-3p', pos_b1, pos_b2, pos_b3)
        self.add_feature(features, '-1pw', pos_b1, wform_i)
        #self.add_feature(features, '-1p-2pw', pos_b1, pos_b2, wform_i)
        #self.add_feature(features, '-1p-1ww', pos_b1, wform_b1, wform_i)
        self.add_feature(features, '+1p', pos_f1)
        self.add_feature(features, '+2p', pos_f2)
        self.add_feature(features, '+1p+2p', pos_f1, pos_f2)
        #self.add_feature(features, '+1p+2p+3p', pos_f1, pos_f2, pos_f3)
        self.add_feature(features, '+1pw', pos_f1, wform_i)
        self.add_feature(features, '-1p+1p', pos_b1, pos_f1)

        self.add_feature(features, '-1w', wform_b1)
        self.add_feature(features, '-1s4', wform_b1[-4:])

        self.add_feature(features, '+1w', wform_f1)
        self.add_feature(features, '+1s3', wform_f1[-3:])

        self.add_feature(features, '-2w', wform_b2)

        self.add_feature(features, '+2w', wform_f2)
        return features

    def get_feat2_features_RU(self, i, feat, context,
                              lemma_context, pos_context, feats_context,
                              joint, val_cnt):
        if feat == None:
            feat = ''

        context       = self.START +       context + self.END
        lemma_context = self.START + lemma_context + self.END
        pos_context   = self.START +   pos_context + self.END
        feats_context = [{feat: self.START[0]}, {feat: self.START[1]}] \
                      + feats_context \
                      + [{feat: self.END[0]  }, {feat: self.END[1]  }]
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

        #feats_i  = feats_context[i].get(feat, '') if feat else \
        #           '|'.join('='.join((x, feats_context[i][x]))
        #                        for x in sorted(feats_context[i]))
        feats_b1 = feats_context[i - 1].get(feat, '') if feat else \
                   '|'.join('='.join((x, feats_context[i - 1][x]))
                                for x in sorted(feats_context[i - 1]))
        feats_b2 = feats_context[i - 2].get(feat, '') if feat else \
                   '|'.join('='.join((x, feats_context[i - 2][x]))
                                for x in sorted(feats_context[i - 2]))
        feats_f1 = feats_context[i + 1].get(feat, '') if feat else \
                   '|'.join('='.join((x, feats_context[i + 1][x]))
                                for x in sorted(feats_context[i + 1]))
        feats_f2 = feats_context[i + 2].get(feat, '') if feat else \
                   '|'.join('='.join((x, feats_context[i + 2][x]))
                                for x in sorted(feats_context[i + 2]))

        features = self.init_features()

        self.add_feature(features, '_')
        self.add_feature(features, 'w', wform)
        if not joint:
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

        if joint or val_cnt < 7:
            self.add_feature(features, '-1f', feats_b1)
            self.add_feature(features, '-1fw', feats_b1, wform_i)
            self.add_feature(features, '-1fl', feats_b1, lemma_i)
            self.add_feature(features, '-1fp', feats_b1, pos_i)
            self.add_feature(features, '+1f', feats_f1)
            self.add_feature(features, '+1fw', feats_f1, wform_i)
            self.add_feature(features, '+1fl', feats_f1, lemma_i)
            self.add_feature(features, '+1fp', feats_f1, pos_i)
            self.add_feature(features, '-1f+1f', feats_b1, feats_f1)
        else:
            self.add_feature(features, '-1f-2f', feats_b1, feats_b2)
            self.add_feature(features, '-1fw', feats_b1, wform_i)
            self.add_feature(features, '-f',
                             feats_b1 if feats_b1 != '' else feats_b2)
            self.add_feature(features, '-fw',
                             feats_b1 if feats_b1 != '' else feats_b2, wform_i)
            self.add_feature(features, '+1f+2f', feats_f1, feats_f2)
            self.add_feature(features, '+1fw', feats_f1, wform_i)
            self.add_feature(features, '+f',
                             feats_f1 if feats_f1 != '' else feats_f2)
            self.add_feature(features, '+fw',
                             feats_f1 if feats_f1 != '' else feats_f2, wform_i)
            self.add_feature(features, '-f+f',
                             feats_b1 if feats_b1 != '' else feats_b2,
                             feats_f1 if feats_f1 != '' else feats_f2)

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
