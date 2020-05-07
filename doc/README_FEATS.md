<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## Morphological Feats Tagging

First of all, we need to create the parser object and load training data.
You can find a complete explanation in
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)
chapter.

### Training

***Morra*** contains total 7 FEATS taggers. There are 2 different model types:
joint, when we train one model for all FEATS tags; and separate, that contains
individual models for every tag. Further, each model type has 2 unidirectional
(forward and backward) and 1 bidirectional taggers. Bidirectional use results
of unidirectional taggers, so we need to train them all.

Also, we have one more top-level tagger that conjoin results of bidirectional
taggers of both types (join and separate). This tagger is just heuristic,
i.e., we don't have to train it.

**NB:** On this step you have a parser object `mp` created and training data
loaded.

Training of unidirectional FEATS tagger:
```python
mp.train_feats(joint=False, rev=False, feat=None, epochs=5,
               no_train_evals=True, seed=None, dropout=None,
               context_dropout=None):
```
**joint**: if `True`, train joint FEATS model; elsewise, train separate models
(default).

**rev**: if `False` (default), train forward tagger; if `True`, train
backward.

**feat**: name of the feat to evaluate the tagger; if `None` (default), then
tagger will be evaluated for all feats.

**epochs**: number of training iterations. If **epochs** greater than `0`,
then the model will be trained for exactly that number of iterations. But you
can specify **epochs** less than `0`, then the best model will be searched
based on evaluation of the *test corpus* (*test corpus* must be loaded it you
want this feature). The search will stop when the result of next `abs(epochs)`
iterations is worse than the best one.

It's allowed to specify epochs as a `tuple` of both variants (positive and
negative). Then, search for the best model will start only when number of
positive epochs will be reached. E.g.: if **epochs**=`(5, -3)`, then the
search will stop after `3` epochs with low score but not early than on the
`5`th epoch.

**no_train_evals**: don't make interim and final evaluations on the *train
corpus*. Stay it as it is (`True`) to save training time.

**seed** (`int`): init value for random number generator. Default is `None`
(don't set init value).

**dropout** `float` (`0` .. `1`): a fraction of weigths to be randomly set to
`0` at each predict to prevent overfitting. Recommended values are about
`0.01` and less. Default is `None` (don't do that).

**context_dropout** `float` (`0` .. `1`): a fraction of POS tags to be
randomly replaced after predict to random POS tags to prevent overfitting.
Recommended values are about `0.1` and less. Default is `None` (don't do
that).

Returns final evaluation score(s) for the trained model (if *test corpus* is
loaded and/or **no_train_evals** is `False`).

After you trained both forward and backward unidirectional taggers of any
type (joint or separate), you can train bidirectional model of that type:
```python
score = mp.train_feats2(joint=False, feat=None, epochs=5,
                        test_max_repeats=0, no_train_evals=True, seed=None,
                        dropout=None, context_dropout=None)
```
All params and return value(s) here have the same meaning as for
`.train_feats()` method except **test_max_repeats**. This param is passing to
the `.predict_feats2()` method (see further) diring evaluation step.

### Save and Loading trained Models

See
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)

### Predict and Evaluate

When models are trained, you can use them to tag your text data. Usually, you
will use only bidirectional model via `.predict_feats2_sents()` or
`.predict_feats3_sents()` methods. But you also have access to unidirectional
models.

#### Unidirectional Models

Tag just one sentence:
```python
sentence = mp.predict_feats(sentence, joint=False, rev=False,
                            feat=None, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**joint**: if `True`, use joint FEATS model; elsewise, use separate models
(default).

**rev**: if `False` (default), use forward tagger; if `True`, backward.

**feat**: name of the feat to tag; if `None` (default), then all possible
feats will be tagged.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

Tag the whole corpus:
```python
sentences = mp.predict_feats_sents(sentences=None, joint=False, rev=False,
                                   feat=None, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If None, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate FEATS tagging:
```python
scores = mp.evaluate_feats(gold=None, test=None, joint=False, rev=False,
                           feat=None, unknown_only=False, silent=False)
```
Calculate the accuracy score of the FEATS tagging of the **test** corpus
against the **gold**. Both **gold** and **test** (like any input corpora in
any ***Morra*** method) may be a name of the file in *CONLL-U* format or
list/iterator of sentences in *Parsed CONLL-U*.

If **gold** is `None` (default), then loaded *test corpus* is used. If
**gold** is a ***Corpuscula***'s corpora wrapper, the `.test()` part will be
used.

If **test** is `None` (default), then the **gold** corpus will be retagged
with unidirectional model and then the result will be compared with the
original **gold** corpus.

**joint**: if `True`, use joint FEATS model; elsewise, use separate models
(default).

**rev**: if `False` (default), use forward tagger; if `True`, backward.

**feat**: name of the feat to evaluate the tagger; if `None` (default), then
the tagger will be evaluated for all feats.

**unknown_only**: calculate accuracy score only for words that are not present
in the *train corpus* (the corpus must be loaded).

**silent**: suppress output.

Returns the accuracy **scores** wrt tokens and wrt tags.

#### Bidirectional Model

Tag just one sentence:
```python
sentence = mp.predict_feats2(sentence, joint=False,
                             with_backoff=True, max_repeats=0,
                             feat=None, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**joint**: if `True`, use joint FEATS-2 model; elsewise, use separate models
(default).

**with_backoff**: if the result of the bidirectional tagger differs from
the results of both base unidirectional taggers, get one of the bases on
the ground of some heuristics.

**max_repeats**: repeat a prediction step based on the previous one while
changes in prediction are diminishing and **max_repeats** is not reached. `0`
(default) means one repeat - only for tokens where unidirectional taggers
don't concur.

**feat**: name of the feat to tag; if `None`, then all possible feats will be
tagged.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

If you have trained bidirectional models of both joint and separate types, you
may use conjoint tagger that use them together:
```python
sentence = mp.predict_feats3(sentence,
                             with_s_backoff=True, max_s_repeats=0,
                             with_j_backoff=True, max_j_repeats=0,
                             inplace=True)
```
Here, **with_s_backoff** and **max_s_repeats** are params **with_backoff** and
**max_repeats** for separate FEATS-2 models; **with_j_backoff** and
**max_j_repeats** are params for joint FEATS-2 model.

Tag the whole corpus:
```python
sentence = mp.predict_feats2_sents(sentences=None, joint=False,
                                   with_backoff=True, max_repeats=0,
                                   feat=None, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If None, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

If both joint and separate FEATS-2 models are available:
```python
sentence = mp.predict_feats3_sents(sentences=None,
                                   with_s_backoff=True, max_s_repeats=0,
                                   with_j_backoff=True, max_j_repeats=0,
                                   inplace=True, save_to=None)
```
All params where explained earlier.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate FEATS tagging:
```python
scores = mp.evaluate_feats2(gold=None, test=None, joint=False, rev=False,
                            with_backoff=True, max_repeats=0,
                            feat=None, unknown_only=False, silent=False)
```
All params where explained earlier.

Returns the accuracy **scores** wrt tokens and wrt tags.

If both joint and separate FEATS-2 models are available:
```python
scores = mp.evaluate_feats3(gold=None, test=None,
                            with_s_backoff=True, max_s_repeats=0,
                            with_j_backoff=True, max_j_repeats=0,
                            feat=None, unknown_only=False, silent=False)
```
All params where explained earlier.

Returns the accuracy **scores** wrt tokens and wrt tags.
