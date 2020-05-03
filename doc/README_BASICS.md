<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## MorphParser Basics

## Initialization

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
mp.load_test_corpus (corpus, append=False)
```
This corpus is used to validate model during training. You can train model
without validation. This will be faster but not informative. And you can't
use autostop feature.

If you'll specify a ***Corpuscula***'s corpora wrapper here, the `.dev()` part
will be used, or, if it doesn't exist, the `.test()` part.

Then, if you didn't specify the **parse** param in `.load_train_corpus()`
method, you should extract corpus statistics now:
```python
mp.parse_train_corpus (self, cnt_thresh=None, ambiguity_thresh=None)
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
\
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
\
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
