# vowpal_porpoise

Lightweight python wrapper for [vowpal_wabbit](https://github.com/JohnLangford/vowpal_wabbit/).

Why: Scalable, blazingly fast machine learning.

## Install

1. Install [vowpal_wabbit](https://github.com/JohnLangford/vowpal_wabbit/). Clone and run ``make``
2. Install [cython](http://www.cython.org/). ```pip install cython```
3. Clone [vowpal_porpoise](https://github.com/josephreisinger/vowpal_porpoise)
4. Run: ```python setup.py install``` to install.

Now can you do: ```import vowpal_porpoise``` from python.

## Examples

### Standard Interface

Linear regression with l1 penalty:
```python
from vowpal_porpoise import VW

# Initialize the model
vw = VW(moniker='test',    # a name for the model
        passes=10,         # vw arg: passes
        loss='quadratic',  # vw arg: loss
        learning_rate=10,  # vw arg: learning_rate
        l1=0.01)           # vw arg: l1

# Inside the with training() block a vw process will be 
# open to communication
with vw.training():
    for instance in ['1 |big red square',\
                      '0 |small blue circle']:
        vw.push_instance(instance)

    # here stdin will close
# here the vw process will have finished

# Inside the with predicting() block we can stream instances and 
# acquire their labels
with vw.predicting():
    for instance in ['1 |large burnt sienna rhombus',\
                      '0 |little teal oval']:
        vw.push_instance(instance)

# Read the predictions like this:
predictions = list(vw.read_predictions_())
```

L-BFGS with a rank-5 approximation:
```python
from vowpal_porpoise import VW

# Initialize the model
vw = VW(moniker='test_lbfgs', # a name for the model
        passes=10,            # vw arg: passes
        lbfgs=True,           # turn on lbfgs
        mem=5)                # lbfgs rank
```

Latent Dirichlet Allocation with 100 topics:
```python
from vowpal_porpoise import VW

# Initialize the model
vw = VW(moniker='test_lda',  # a name for the model
        passes=10,           # vw arg: passes
        lda=100,             # turn on lda
        minibatch=100)       # set the minibatch size
```


### Scikit-learn Interface

vowpal_porpoise also ships with an interface into scikit-learn, which allows awesome experiment-level stuff like cross-validation:

```python
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from vowpal_porpoise.sklearn import VW_Classifier

GridSearchCV(
        VW_Classifier(loss='logistic', moniker='example_sklearn',
                      passes=10, silent=True, learning_rate=10),
        param_grid=parameters,
        score_func=f1_score,
        cv=StratifiedKFold(y_train),
).fit(X_train, y_train)
```

Check out [example_sklearn.py](https://github.com/josephreisinger/vowpal_porpoise/blob/master/examples/example_sklearn.py) for more details


### Library Interace (DISABLED as of 2013-08-12)

Via the ```VW``` interface:
```python
with vw.predicting_library():
    for instance in ['1 |large burnt sienna rhombus', \
                      '1 |little teal oval']:
        prediction = vw.push_instance(instance)
```
Now the predictions are returned directly to the parent process, rather than having to read from disk.
See ```examples/example1.py``` for more details.

Alternatively you can use the raw library interface:
```python
import vw_c
vw = vw_c.VW("--loss=quadratic --l1=0.01 -f model")
vw.learn("1 |this is a positive example")
vw.learn("0 |this is a negative example")
vw.finish()
```
Currently does not support passes due to some limitations in the underlying vw C code.

### Need more examples?

* [example1.py](https://github.com/josephreisinger/vowpal_porpoise/blob/master/examples/example1.py): SimpleModel class wrapper around VP (both standard and library flavors)
* [example_library.py](https://github.com/josephreisinger/vowpal_porpoise/blob/master/examples/example_library.py): Demonstrates the low-level vw library wrapper, classifying lines of **alice in wonderland** vs **through the looking glass**.

## Why

vowpal\_wabbit is **insanely**
fast and scalable. vowpal_porpoise is slower, but **only** during the
initial training pass. Once the data has been properly cached it will idle while vowpal\_wabbit does all the heavy lifting.
Furthermore, vowpal\_porpoise was designed to be lightweight and not to get in the way
of vowpal\_wabbit's scalability, e.g. it allows distributed learning via
```--nodes``` and does not require data to be batched in memory. In our
research work we use vowpal\_porpoise on an 80-node cluster running over multiple
terabytes of data.

The main benefit of vowpal\_porpoise is allowing **rapid prototyping** of new
models and feature extractors. We found that we had been doing this in an
ad-hoc way using python scripts to shuffle around massive gzipped text files,
so we just closed the loop and made vowpal\_wabbit a python library.

## How it works

Wraps the vw binary in a subprocess and uses stdin to push data, temporary
files to pull predictions. Why not use the prediction labels vw provides on stdout? It
turns out that the python GIL basically makes streamining in and out of a
process (even asynchronously) painfully difficult. If you know of a clever way
to get around this, please email me. In other languages (e.g. in a forthcoming
scala wrapper) this is not an issue.

Alternatively, you can use a pure api call (```vw_c```, wrapping libvw) for prediction.


## Contact

Joseph Reisinger [@josephreisinger](http://twitter.com/josephreisinger)

## Contributors

* Austin Waters (austin.waters@gmail.com)
* Joseph Reisinger (joeraii@gmail.com)
* Daniel Duckworth (duckworthd@gmail.com)

## License

Apache 2.0
