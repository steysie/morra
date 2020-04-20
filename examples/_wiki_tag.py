#!/usr/bin/python
# -*- coding: utf-8 -*-

from morra import MorphParser3
from local_methods import guess_pos, guess_lemma, guess_feat

mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma, guess_feat=guess_feat)
#mp.load('model_speech.pickle')
mp.load('model_.pickle')
mp.predict_lemma_sents(mp.predict_pos2_sents('wiki.conllu'), save_to='wiki_tagged.conllu')
