# -*- coding: utf-8 -*-
# Morra project: Base features
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
from collections import OrderedDict

class BaseFeatures:
    """Base class for all features of the project"""

    START = ['-START2-', '-START-']
    END = ['-END-', '-END2-']
    alphabet_RU = ''.join(['ё']
                        + [chr(i) for i in range(ord('а'), ord('я') + 1)])

    def __init__(self, lang='RU'):
        if lang == 'RU':
            self.alphabet = self.alphabet_RU
            self.alphabet_upper = self.alphabet.upper()
        else:
            raise ValueError('ERROR: Features for lang "{}" are not defined'
                                 .format(lang))

    @staticmethod
    def init_features():
        return OrderedDict()

    @staticmethod
    def add_feature(features, name, *args):
        f = ' '.join((name,) + tuple(args))
        features[f] = features.get(f, 0) + 1

    def normalize(self, wform):
        '''
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        '''
        #if '-' in wform and wform[0] != '-':
        #    return '!HYPHEN'
        #elif word.isdigit() and len(wform) == 4:
        #    return '!YEAR'
        #el
        if wform is None:
            return wform
        elif wform[0].isdecimal():
            return '!DIGITS'
        else:
            return wform.lower()

    def wform_shape(self, wform):
        shape = ''
        for c in wform:
            if c in self.alphabet:
                c = 'x'
            elif c in self.alphabet_upper:
                c = 'X'
            elif c.isalpha():
                if c.isupper():
                    c = 'A'
                else:
                    c = 'a'
            elif c.isnumeric():
                c = 'd'
            shape += c
        if len(shape) > 5:
            shape = shape[:2] + ''.join(sorted(set(shape[2:-2]))) + shape[-2:]
        return shape
