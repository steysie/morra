<div align="right"><strong>RuMor: Russian Morphology project</strong></div>
<h2 align="center">Morra: morphological parser (POS, lemmata, NER etc.)</h2>

## Supplements

***Morra*** package contain few additional utility methods that can simplify
processing of corpuses. Also, it has a tools to simplify hyperparameters 
selection and for using several trained morphological parsers conjointly in
ensemble.

### Corpus processing

Corpus processing utilities are contained in `BaseParser` class. All its
successors also have them, so one can use `MorphParser` that was created for
the main task of prediction:
```python
from morra import MorphParser3
mp = MorphParser3()
```

Wrappers for [***Corpuscula***](https://github.com/fostroll/corpuscula)
`Conllu.load()` and `Conllu.save()` methods:
```python
MorphParser3.load_conllu(*args, **kwargs)
MorphParser3.save_conllu(*args, **kwargs)
```
**args** and **kwargs** are arguments that are passed to corresponding methods
of the ***Corpuscula***'s `Conllu` class.

Split a **corpus** in a given proportion:
```python
MorphParser3.split_corpus(corpus, split=[.8, .1, .1], save_split_to=None,
                          seed=None, silent=False)
```
Here, **corpus** is a name of file in
[*CONLL-U*](https://universaldependencies.org/format.html) format or
list/iterator of sentences in
[*Parsed CONLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)

**split**: `list` of sizes of the necessary **corpus** parts. If values are of
`int` type, they are interpreted as lengths of new corpuses in sentences; if
values are `float`, they are proportions of a given **corpus**. The types of
the **split** values can't be mixed: they are either all `int`, or all
`float`.

**NB:** The sum of `float` values must be less or equals to 1; the sum of `int`
values can't be greater than the lentgh of the **corpus**.

**save_split_to**: `list` of file names to save the result of the **corpus**
splitting. Can be `None` (default; don't save parts to files) or its length
must be equal to the length of **split**.

**silent**: if `True`, suppress output.

Returns a list of new corpuses.

Sometimes, in train corpus, number of tokens with morphological feats of some
type is not enough for robust model training. Feats of such tupes better
remove from train process:
```python
mp.remove_rare_feats(abs_thresh=None, rel_thresh=None, full_rel_thresh=None)
```
That method removes from train and test corpora those feats, occurence of
which in the train corpus is less then a threshold:

**abs_thresh** (of `int` type): remove features if their count in the train
corpus is less than this value.

**rel_thresh** (or `float` type): remove features if their frequency wrt total
feats count of the train corpus is less than this value.

**full_rel_thresh** (or `float` type): remove features if their frequency wrt
the full count of the tokens of the train corpus is less than this value.

**NB:**: **rell_thresh** and **full_rell_thresh** must be between 0 and 1.

### Autotrain

This is a tool for ***Morra*** models' hyperparameters selection. It made
simple, without any multiprocessing, so it's just save time of user from 
cyclical rerunnings the training process with new parameters, but it's not 
speedup the training. Howewer, user can simultaneously run several versions of
`autotrain()` with different hyperparameters' combinations.

```python
from morra import autotrain
autotrain(train_func, *args, silent=False,
          backup_model_func=None, restore_model_func=None,
          reload_trainset_func=None, fit_params={}, params_in_process={},
          **kwargs):
```
Params:

**train_func**: method to train the model.

**args**: positional args for **train_func**. Will be passed as is.

**silent**: if `True`, suppress the output.

**backup_model_func**: method to backup the model internal state. If not
`None`, it will be used to save the state of the model with the highest
score, for that we don't train it anew.

**restore_model_func**: method to restore the model internal state. If not
`None`, then in the end, this method will be invoked with the only param:
internal state of the best model. There is no use or even may be cause of
runtime error, to specify **restore_model_func** without
**backup_model_func**.

**reload_trainset_func**: method to load the training corpus anew. It may
become necessary if you want to achieve complete reproducibility. The training
corpus is randomly shuffled before each training iteration, so you should
reload it before each new training if you want to have possibility to repeat
the training with the best parameters and get the model with exactly the same
internal state.

**fit_params**: `dict` of hyperparameters' keys and lists of their values
among which we want to find the best. It is not very flexible. We just check
combinations of all the given values of all the given hyperparameters. E.g.:
```python
fit_params={
    'dropout': [None, .005, .01],
    'context_dropout': [None, .05, .1]
}
```
produces next additional keyword args for **train_func**:
```python
{'dropout': None, 'context_dropout': None}
{'dropout': None, 'context_dropout': .05}
{'dropout': None, 'context_dropout': .1}
{'dropout': .005, 'context_dropout': None}
{'dropout': .005, 'context_dropout': .05}
{'dropout': .005, 'context_dropout': .1}
{'dropout': .01, 'context_dropout': None}
{'dropout': .01, 'context_dropout': .05}
{'dropout': .01, 'context_dropout': .1}
```

**params_in_process**: for internal use only. Leave it as it is.

**kwargs**: keyword args for **train_func**. Will be passed as is.

When all training have done, the tool returns
`tuple(<best model score>, <best model params>, <best model internal state>)`.
If **backup_model_func** is `None`, `<best model internal state>` will be 
`None`, too.

Also, `<best model internal state>` will be loaded to your model, if
**reload_trainset_func** is not `None`.

### Ensemble

