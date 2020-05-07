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

### Creating an Ensemble

***Morra*** has a simple tool that allow to conjoin several `MorphParsers` to
the ensemble. That can increase stability of prediction and, sometimes,
increase the overall prediction quality.

To create an ensemble, run:
```python
from morra import MorphEnsemble
me = MorphEnsemble(cdict)
```
Here, **cdict** is a
[*Corpus Dictionary*](https://github.com/fostroll/corpuscula/blob/master/doc/README_CDICT.md)
from ***Corpuscula*** package. You don't have to create it directly. You can
use *Corpus Dictionary* of any `MorphParser` that you want to conjoin in the
ensemble. **NB:** We presume that you train all the parsers on the same
corpus; otherwise, there may be nuances.

So, you've created the empty `MorphEnsemble` `me`. After that, you can add 
your parsers to it:
```python
index = me.add(predict_method, **kwargs)
```
Here, **predict_method** is a method of `MorphParser` that we prefer to use
for prediction; **kwargs** - keyword arguments for **predict_method**.

`.add()` returns the **index** of the method added. You can use it to remove
the method from the ensemble:
```python
method = me.pop(index)
```
Although, someone hardly will use this method often.

Supposing, we have parsers **mp1**, **mp2** and **mp3**.
Then, for example, we can add them to the ensemble with:
```python
me.add(mp1.predict3, pos_backoff=False, pos_repeats=2)
me.add(mp2.predict3, pos_backoff=False, pos_repeats=2)
me.add(mp3.predict3, pos_backoff=False, pos_repeats=2)
```

Note, that the addition's order matters. We count a prediction as the best
if maximum number of parsers vote for it. But sometimes, we have several
groups of equally voted parsers with maximum number of members. Then, the best
model will be taken from the group that contains a `MorphParser` which was
added earlier.

If you want to predict only one exact field of *CONLL-U*, you can pass to
`.add()` the more specific methods. E.g., for the only POS field you may want
to use the method `.predict_pos2()`:
```python
me.add(mp1.predict_pos2, with_backoff=False, max_repeats=2)
me.add(mp2.predict_pos2, with_backoff=False, max_repeats=2)
me.add(mp3.predict_pos2, with_backoff=False, max_repeats=2)
```

To use the ensemble for the prediction of the necessary fields of just one
sentence you can with:
```python
sentence = predict(fields_to_predict, sentence, inplace=True)
```
Here, **fields_to_predict** is a list of *CONLL-U* fields names you want to
get a prediction for. E.g.: `fields_to_predict=['UPOS', 'LEMMA', 'FEATS]`.
Note, that these fields must be between the fields that the methods added with
the `.add()` can predict.

**sentence**: the sentence in *Parsed CONLL-U* format.

**inplace**: if `True` (default), method changes and returns the given
**sentence** itself. Elsewise, the new sentence will be created.

Returns the **sentence** tagged, also in *Parsed CONLL-U* format.

Predict the fields of the whole corpus:
```python
sentences = predict_sents(fields_to_predict, sentences, inplace=True,
                          save_to=None)
```
**sentences**: a name of the file in *CONLL-U* format or list/iterator of
sentences in *Parsed CONLL-U*. If `None`, then loaded *test corpus* is used.
You can specify a ***Corpuscula***'s corpora wrapper here. In that case, the
`.test()` part will be used.

**save_to**: the name of the file where you want to save the result. Default
is `None`: we don't want to save.

Returns iterator of tagged **sentences** in *Parsed CONLL-U* format.

Evaluate the ensemble quality:
```python
scores = evaluate(fields_to_evaluate, gold=None, test=None,
                  feat=None, unknown_only=False, silent=False)
```
**fields_to_predict**: a list of *CONLL-U* fields names you want to
evaluate a prediction for. E.g.: `fields_to_evaluate=['UPOS', 'LEMMA',
'FEATS]`. Note, that these fields must be between the fields that the methods
added with the `.add()` can predict.

**gold**: a corpus of tagged sentences to score the tagger on.

**test**: a corpus of tagged sentences to compare with **gold**.

**feat** (of `str` type): name of the feat to evaluate the ensemble; if
`None`, then the ensemble will be evaluated for all feats.

**unknown_only**: calculate accuracy score only for words that not present in
the train corpus (detected by **cdict** passed to constructor).

**silent**: if `True`, suppress output.

Return a `tuple` of accuracy **scores** of the ensemble against the
**gold**:<br/>
1. wrt tokens: the tagging of the whole token may be either correct or
not.<br/>
2. wrt tags: sum of correctly detected tags to sum of all tags that are
non-empty in either **gold** or retagged sentences.
