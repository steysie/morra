<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## Named-entity Recognition

### Initialization

For the NER task ***Morra*** has the separate parser.
```python
from morra import MorphParserNE
mp = MorphParserNE()
```

Usually, it's not enough for maximum scores achieving. The full syntax is:
```python
mp = MorphParserNE(features='RU', guess_ne=None)
```

Params of the constructor a similar to params of the `MorphParser` constructor
which description one can find in
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md).
However, the base class of `MorphParserNE` **features** is
[`morra.features.FeaturesNE`](https://github.com/fostroll/morra/blob/master/morra/features_ne.py).

An example of the implementation of the **guess_ne** helper one can find in
[`local_methods.py`](https://github.com/fostroll/morra/blob/master/scripts/local_methods.py).

### Loading the Training Data

`MorphParserNE` works with corpora in *CONLL-U* or *Parsed CONLL-U* formats.
The NE tags are placed to the MISC field as the value of the `NE` variable
(E.g.: `NE=Address`).

So, you have to convert your training corpora *CONLL-U* format and put your
gold NE labels to the MISC field. For conversion, you can use the
[***Corpuscula***](https://github.com/fostroll/corpuscula)
[*CONLL-U* support](https://github.com/fostroll/corpuscula/blob/master/doc/README_CONLLU.md).

Also, you have to accomplish morphological parsing prior the training the NE
tagger. If you make it with one of ***Morra*** taggers, you don't have to
worry about *CONLL-U* conversion, because you'll get this format as output of
the tagger.

When you have your training corpora prepared, you need to load it into parser.
Refer corresponding section of
[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)
to learn how to do it.

### Training

***Morra*** contains total 7 NE taggers. There are 2 different model types:
joint, when we train one model for all NE tags; and separate, that contains
individual models for every tag. Further, each model type has 2 unidirectional
(forward and backward) and 1 bidirectional taggers. Bidirectional use results
of unidirectional taggers, so we need to train them all.

Also, we have one more top-level tagger that conjoin results of bidirectional
taggers of both types (join and separate). This tagger is just heuristic,
i.e., we don't have to train it.

**NB:** On this step you have a parser object `mp` created and training data
loaded.

Training of unidirectional NE tagger:
```python
mp.train_ne(joint=False, rev=False, ne=None, epochs=5,
            no_train_evals=True, seed=None, dropout=None,
            context_dropout=None):
```
**joint**: if `True`, train joint NE model; elsewise, train separate models
(default).

**rev**: if `False` (default), train forward tagger; if `True`, train
backward.

**ne**: name of the entity to evaluate the tagger; if `None` (default), then
tagger will be evaluated for all entities.

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
score = mp.train_ne2(joint=False, ne=None, epochs=5,
                     test_max_repeats=0, no_train_evals=True, seed=None,
                     dropout=None, context_dropout=None)
```
All params and return value(s) here have the same meaning as for `.train_ne()`
method except **test_max_repeats**. This param is passing to the
`.predict_ne2()` method (see further) diring evaluation step.

### Save and Loading trained Models

Anytime, you can backup and restore internal states of trained models:
```python
o = mp.backup()
mp.restore(o)

mp.save(file_path)
mp.load(file_path)
```

Also, you can backup and restore separate models. `MorphParserNE` has
corresponding methods:
```python
mp._save_cdict(file_path)
mp._save_ne_model(file_path)       # joint
mp._save_ne_rev_model(file_path)
mp._save_ne2_model(file_path)
mp._save_ne_models(file_path)      # separate
mp._save_ne_rev_models(file_path)
mp._save_ne2_models(file_path)

mp._load_cdict(file_path)
mp._load_ne_model(file_path)       # joint
mp._load_ne_rev_model(file_path)
mp._load_ne2_model(file_path)
mp._load_ne_models(file_path)      # separate
mp._load_ne_rev_models(file_path)
mp._load_ne2_models(file_path)
```

### Predict and Evaluate

When models are trained, you can use them to tag your text data. Usually, you
will use only bidirectional model via `.predict_ne2_sents()` or
`.predict_ne3_sents()` methods. But you also have access to unidirectional
models.

#### Unidirectional Models

Tag just one sentence:
```python
sentence = mp.predict_ne(sentence, joint=False, rev=False,
                         ne=None, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**joint**: if `True`, use joint NE model; elsewise, use separate models
(default).

**rev**: if `False` (default), use forward tagger; if `True`, backward.

**ne**: a name of the entity to tag; if `None` (default), then all possible
entities will be tagged.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

Tag the whole corpus:
```python
sentences = mp.predict_ne_sents(sentences=None, joint=False, rev=False,
                                ne=None, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If None, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate NE tagging:
```python
score = mp.evaluate_ne(gold=None, test=None, joint=False, rev=False,
                       ne=None, silent=False)
```
Calculate the accuracy score of the NE tagging of the **test** corpus
against the **gold**. Both **gold** and **test** (like any input corpora in
any ***Morra*** method) may be a name of the file in *CONLL-U* format or
list/iterator of sentences in *Parsed CONLL-U*.

If **gold** is `None` (default), then loaded *test corpus* is used. If
**gold** is a ***Corpuscula***'s corpora wrapper, the `.test()` part will be
used.

If **test** is `None` (default), then the **gold** corpus will be retagged
with unidirectional model and then the result will be compared with the
original **gold** corpus.

**joint**: if `True`, use joint NE model; elsewise, use separate models
(default).

**rev**: if `False` (default), use forward tagger; if `True`, backward.

**ne**: a name of the entity to evaluate the tagger; if `None` (default), then
the tagger will be evaluated for all possible entities.

**silent**: suppress output.

Returns the accuracy score.

#### Bidirectional Model

Tag just one sentence:
```python
sentence = mp.predict_ne2(sentence, joint=False,
                          with_backoff=True, max_repeats=0,
                          ne=None, inplace=True)
```
**sentence**: the sentence in *Parsed CONLL-U* format.

**joint**: if `True`, use joint NE-2 model; elsewise, use separate models
(default).

**with_backoff**: if the result of the bidirectional tagger differs from
the results of both base unidirectional taggers, get one of the bases on
the ground of some heuristics.

**max_repeats**: repeat a prediction step based on the previous one while
changes in prediction are diminishing and **max_repeats** is not reached. `0`
(default) means one repeat - only for tokens where unidirectional taggers
don't concur.

**ne**: a name of the entity to tag; if `None` (default), then all possible
entities will be tagged.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

If you have trained bidirectional models of both joint and separate types, you
may use conjoint tagger that use them together:
```python
mp.predict_ne3(sentence,
               with_s_backoff=True, max_s_repeats=0,
               with_j_backoff=True, max_j_repeats=0,
               inplace=True)
```
Here, **with_s_backoff** and **max_s_repeats** are params **with_backoff** and
**max_repeats** for separate NE-2 models; **with_j_backoff** and
**max_j_repeats** are params for joint NE-2 model.

Tag the whole corpus:
```python
sentences = mp.predict_ne2_sents(sentences=None, joint=False,
                                 with_backoff=True, max_repeats=0,
                                 ne=None, inplace=True, save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If None, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

If both joint and separate NE-2 models are available:
```python
mp.predict_ne3_sents(sentences=None,
                     with_s_backoff=True, max_s_repeats=0,
                     with_j_backoff=True, max_j_repeats=0,
                     inplace=True, save_to=None)
```
All params where explained earlier.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate NE tagging:
```python
score = mp.evaluate_ne2(gold=None, test=None, joint=False, rev=False,
                        with_backoff=True, max_repeats=0,
                        ne=None, silent=False)
```
All params where explained earlier.

Returns the accuracy score.

If both joint and separate NE-2 models are available:
```python
score = mp.evaluate_ne3(gold=None, test=None,
                        with_s_backoff=True, max_s_repeats=0,
                        with_j_backoff=True, max_j_repeats=0,
                        ne=None, silent=False)
```
All params where explained earlier.

Returns the accuracy score.
