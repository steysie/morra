#!/usr/bin/python
# -*- coding: utf-8 -*-

from corpuscula.conllu import Conllu
from corpuscula.wikipedia_utils import download_wikipedia, Wikipedia
from morra import MorphParser3
from toxic.wikipedia_utils import TokenizedWikipedia
from local_methods import guess_pos, guess_lemma, guess_feat

download_wikipedia(overwrite=False)
mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma, guess_feat=guess_feat)
mp.load('model_speech.pickle')
mp.predict_lemma_sents(
    mp.predict_pos2_sents(
        Conllu.fix(
            TokenizedWikipedia().articles(), adjust_for_speech=True
        )
    ),
    save_to='wiki_speech_tagged.conllu'
)
