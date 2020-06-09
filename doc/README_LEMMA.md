<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## Lemmata Detection

For lemmata detection you need to have the training corpus already POS-tagged.
[Part of Speach Tagging](https://github.com/fostroll/morra/blob/master/doc/README_POS.md)

First of all, we need to create the parser object and load training data.
You can find a complete explanation in
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)
chapter.

### Training

For now, ***Morra*** doesn't contain its own LEMMA generator and use the one
from ***Corpuscula*** package. Though, before making predictions you have to run
dummy train method.

**NB:** By this step you should have a parser object `mp` created and training data
loaded.

Just run this method without params:
```python
mp.train_lemma():
```

### Save and Loading trained Models

See
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)

### Predict and Evaluate

When the model is trained, you can use it to generate lemmata for your text data.

Generate lemmata just for one sentence:
```python
sentence = mp.predict_lemma(sentence, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence**, also in *Parsed CONLL-U* format, with LEMMA field
filled.

Generate lemmata for the whole corpus:
```python
sentences = mp.predict_lemmata_sents(sentences=None, inplace=True,
                                     save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or a list/iterator of
sentences in *Parsed CONLL-U*. If None, then the loaded *test corpus* is used.
You can specify ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the results. Default
is `None`: we don't want to save.

Returns an iterator of **sentences** in *Parsed CONLL-U* format with LEMMA field
filled.

Evaluate LEMMA generator:
```python
score = mp.evaluate_lemma(gold=None, test=None, unknown_only=False,
                          silent=False)
```
Calculate accuracy score of the LEMMA generator on the **test** corpus against
the **gold**. Both **gold** and **test** (like any input corpora in any
**Morra** method) may be a name of the file in *CONLL-U* format or a
list/iterator of sentences in *Parsed CONLL-U*.

If **gold** is `None` (default), then loaded *test corpus* is used. If
**gold** is ***Corpuscula***'s corpora wrapper, the `.test()` part will be
used.

If **test** is `None` (default), then the **gold** corpus will be retagged
with unidirectional model, and then the result will be compared with the
original **gold** corpus.

**unknown_only**: calculate accuracy score only for words that are not present
in the *train corpus* (the corpus must be loaded).

**silent**: suppress output.

Returns the accuracy **score**.
