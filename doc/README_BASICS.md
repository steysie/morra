<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## MorphParser Basics

### Initialization

First of all, we need to create the `MorphParser3` object. The simplest way to
do it is:
```python
from morra import MorphParser3
mp = MorphParser3()
```

Usually, it's not enough for maximum scores achieving. The full syntax is:
```python
mp = MorphParser3(features='RU',
                  guess_pos=None, guess_lemma=None, guess_feat=None)
```

All ***Morra*** taggers use hand crafter features. Now it support only `'RU'`
features for Russian language. But you can implement your own version. For
that, you have to make a successor of
[`morra.features.Features2`](https://github.com/fostroll/morra/blob/master/morra/features2.py)
class and implement `get_*_features()` methods you want to improve. Note that
methods you need to change may belong to the parent of the `Feature2` class,
the
[`Features`](https://github.com/fostroll/morra/blob/master/morra/features.py)
class. But for `MorphParser3` your class must deliver all `Feature2` methods,
so your need to implement the successor of the exactly `Feature2` class.

Then, just import your class and pass its name as the value for the
**features** param.

Besides of the **features**, you can pass to constructor the helper methods
for internal POS, LEMMA and FEATS classificators. They serve to improve
internal prediction algorithms and, normally, they should be different for
different training corpora. For example, we have implementation of those
methods for *SynTagRus* in the
[`local_methods.py`](https://github.com/fostroll/morra/blob/master/scripts/local_methods.py)
script.

In the 
[`morph3_pipeline.py`](https://github.com/fostroll/morra/blob/master/examples/morph3_pipeline.py)
script you can find an example of the real tagging pipeline where creates
`MorphParser3` object with usage of those params.

**NB:** For prediction, you should create the parser object with the same
params as for training. We save neither the feature class nor helper methods
with model.

### Loading the Training Data

After create the parser, you need to load traininig data.

```python
mp.load_train_corpus(corpus, append=False, parse=False, test=None, seed=None)
```
Here, **corpus** is a name of file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. Also, it allows one of the
[***Corpuscula***'s corpora wrappers](https://github.com/fostroll/corpuscula/blob/master/doc/README_CORPORA.md)
here. In that case, the `.train()` part will be used.

**append**: add corpus to already loaded one(s).

**parse**: extract corpus statistics right after loading.

**test**: `float` (`0` .. `1`). If set, then **corpus** will be shuffled
and specified part of it stored as development *test corpus*. The rest part
will be stored as *train corpus*. Default is `None`: don't do that.

**seed** (`int`): init value for random number generator. Default is `None`
(don't set init value). Used only if **test** is not `None`.

If you didn't specify the **test** param, you can load development *test
corpus* directly:
```python
mp.load_test_corpus(corpus, append=False)
```
This corpus is used to validate model during training. You can train model
without validation. This will be faster but not informative. And you can't
use autostop feature.

If you'll specify a ***Corpuscula***'s corpora wrapper here, the `.dev()` part
will be used, or, if it doesn't exist, the `.test()` part.

Then, if you didn't specify the **parse** param in `.load_train_corpus()`
method, you should extract corpus statistics now:
```python
mp.parse_train_corpus(self, cnt_thresh=None, ambiguity_thresh=None)
```

### Training

For the training process description refer the corresponding docs:

[Part of Speach Tagging](https://github.com/fostroll/morra/blob/master/doc/README_POS.md)

[Lemmata Detection](https://github.com/fostroll/morra/blob/master/doc/README_LEMMA.md)

[Morphological Feats Tagging](https://github.com/fostroll/morra/blob/master/doc/README_FEATS.md)

### Save and Loading trained Models

Anytime, you can backup and restore internal states of trained models:
```python
o = mp.backup()
mp.restore(o)

mp.save(file_path)
mp.load(file_path)
```

Also, you can backup and restore separate models. `MorphParser3` has
corresponding methods:
```python
mp._save_cdict(file_path)
mp._save_pos_model(file_path)
mp._save_pos_rev_model(file_path)
mp._save_pos2_model(file_path)
mp._save_lemma_model(file_path)
mp._save_feats_model(file_path)       # joint
mp._save_feats_rev_model(file_path)
mp._save_feats2_model(file_path)
mp._save_feats_models(file_path)      # separate
mp._save_feats_rev_models(file_path)
mp._save_feats2_models(file_path)

mp._load_cdict(file_path)
mp._load_pos_model(file_path)
mp._load_pos_rev_model(file_path)
mp._load_pos2_model(file_path)
mp._load_lemma_model(file_path)
mp._load_feats_model(file_path)       # joint
mp._load_feats_rev_model(file_path)
mp._load_feats2_model(file_path)
mp._load_feats_models(file_path)      # separate
mp._load_feats_rev_models(file_path)
mp._load_feats2_models(file_path)
```

### Predict and Evaluate

Apart from the separate methods for each *CONLL-U* field (refer the
corresponding docs above), ***Morra*** have methods for predicting fields and
evaluating models conjointly. Note, that you must already have trained models
of all taggers for using those methods.

#### Unidirectional Models

Predict the fields of just one sentence:
```python
sentence = mp.predict(sentence, pos_rev=False, feats_joint=False,
                      feats_rev=False, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**pos_rev**: if `False` (default), use forward POS tagger; backward elsewise.

**feats_joint**: if `False` (default), use separate FEATS models; elsewise,
use joint model.

**feats_rev**: if `False` (default), use forward FEATS tagger; backward
elsewise.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

Predict the fields of the whole corpus:
```python
sentences = mp.predict_sents(sentences=None, pos_rev=False, feats_joint=False,
                             feats_rev=False, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If `None`, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate conjoint prediction:
```python
score = mp.evaluate(gold=None, test=None, pos_rev=False,
                    feats_joint=False, feats_rev=False, feat=None,
                    unknown_only=False, silent=False)
```
Calculate joint accuracy score of the unidirectional POS, LEMMA and FEATS
predictions on the **test** corpus against the **gold**. Both **gold** and
**test** (like any input corpora in any ***Morra*** method) may be a name of
the file in *CONLL-U* format or list/iterator of sentences in
*Parsed CONLL-U*.

If **gold** is `None` (default), then loaded *test corpus* is used. If
**gold** is a ***Corpuscula***'s corpora wrapper, the `.test()` part will be
used.

If **test** is `None` (default), then the **gold** corpus will be retagged
with unidirectional models and then the result will be compared with the
original **gold** corpus.

**pos_rev**: if `False` (default), use forward POS tagger; backward elsewise.

**feats_joint**: if `False` (default), use separate FEATS models; elsewise,
use joint model.

**feats_rev**: if `False` (default), use forward FEATS tagger; backward
elsewise.

**feat**: calculate FEATS tagger accuracy wrt that only tag.

**unknown_only**: calculate accuracy score only for words that are not present
in the *train corpus* (the corpus must be loaded).

**silent**: suppress output.

Returns the accuracy scores wrt tokens and wrt tags.

#### Bidirectional Models

Predict fields of just one sentence:
```python
sentence = mp.predict2(sentence, pos_backoff=True, pos_repeats=0,
                       feats_joint=False, feats_backoff=True, feats_repeats=0,
                       inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**pos_backoff**: if the result of the bidirectional POS tagger differs from
the results of both base unidirectional taggers, get one of the bases on the
ground of some heuristics.

**pos_repeats**: repeat a prediction step based on the previous one while
changes in prediction are diminishing and **max_repeats** of the POS-2 tagger
is not reached. `0` (default) means one repeat - only for tokens where POS-1
taggers don't concur.

**feats_joint**: if `True`, use joint model; elsewise, use separate models
(default).

**feats_backoff**: if result of bidirectional FEATS tagger differs from the
results of both base unidirectional taggers, get one of the bases on the
ground of some heuristics.

**feats_repeats**: repeat a prediction step based on the previous one while
changes in prediction are diminishing and ***max_repeats*** of the FEATS-2
tagger is not reached. `0` (default) means one repeat - only for tokens where
FEATS-1 taggers don't concur.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

Next method can be used if both joint and separate FEATS-2 models are
available:
```python
sentence = mp.predict3(sentence, pos_backoff=True, pos_repeats=0,
                       feats_s_backoff=True, feats_s_repeats=0,
                       feats_j_backoff=True, feats_j_repeats=0, inplace=True)
```
**feats_s_backoff**: if result of separate FEATS-2 tagger differs from both
its base taggers, get one of the bases on the ground of some heuristics.

**feats_s_repeats**: parameter for `predict_feats3()`

**feats_j_backoff**: if result of joint FEATS-2 tagger differs from both its
base taggers, get one of the bases on the ground of some heuristics.

**feats_j_repeats**: parameter for ``predict_feats3()``

Predict the fields of the whole corpus:
```python
sentences = mp.predict2_sents(sentences=None, pos_backoff=True, pos_repeats=0,
                              feats_joint=False, feats_backoff=True,
                              feats_repeats=0, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If `None`, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

If both joint and separate FEATS-2 models are available:
```python
sentences = mp.predict3_sents(sentences=None, pos_backoff=True, pos_repeats=0,
                              feats_s_backoff=True, feats_s_repeats=0,
                              feats_j_backoff=True, feats_j_repeats=0,
                              inplace=True, save_to=None)
```
All params where explained earlier.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate conjoint prediction:
```python
scores = mp.evaluate2(gold=None, test=None, pos_backoff=True, pos_repeats=0,
                      feats_joint=False, feats_backoff=True, feats_repeats=0,
                      feat=None, unknown_only=False, silent=False)
```
All params where explained earlier.

Returns the accuracy **scores** wrt tokens and wrt tags.

If both joint and separate FEATS-2 models are available:
```python
scores = mp.evaluate3(gold=None, test=None,
                      pos_backoff=True, pos_repeats=0,
                      feats_s_backoff=True, feats_s_repeats=0,
                      feats_j_backoff=True, feats_j_repeats=0,
                      feat=None, unknown_only=False, silent=False)
```
All params where explained earlier.

Returns the accuracy **scores** wrt tokens and wrt tags.
