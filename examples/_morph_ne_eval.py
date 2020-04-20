#!/usr/bin/python
# -*- coding: utf-8 -*-

from morra import MorphParserNE

gold = 'result_ner_test.conllu'
#test = 'result_ner_test_parsed.conllu'
test = 'ner_flair.conllu'
#test = 'ner_stanza.conllu'


mp = MorphParserNE()
mp.evaluate_ne(gold=gold, test=test)
