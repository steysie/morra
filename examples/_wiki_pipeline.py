#!/usr/bin/python
# -*- coding: utf-8 -*-

from corpuscula.conllu import Conllu
from corpuscula.wikipedia_utils import download_wikipedia, Wikipedia
from toxic.wikipedia_utils import TokenizedWikipedia

download_wikipedia(overwrite=False)

'''
with open('wiki.con__u', 'wt', encoding='utf-8') as f:
    for x in Wikipedia().articles():
        print(x[0] + '\t[[ ' + x[1], file=f)
        if x[2]:
            print(file=f)
            print(x[2], file=f)
        print(']]\n\n', file=f)
'''
Conllu.save(TokenizedWikipedia().articles(), 'wiki.conllu', fix=False, log_file=None)
