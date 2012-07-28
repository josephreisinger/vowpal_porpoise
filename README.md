# vowpal_porpoise

Lightweight python wrapper for vowpalwabbit

## Install

1. First install vw in your path. Git clone it here: https://github.com/JohnLangford/vowpal_wabbit/
2. Also make sure that cython is installed: ```pip install cython```
3. To install vowpal_porpoise, run: ```python setup.py install```
4. Now can you do: ```import vowpal_porpoise``` from python.

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

# Inside the with training() block a vw process will be open to communication
with vw.training():
    for instance in ['1 |this is a positive example', '0 |this is a negative example']:
        vw.push_instance(instance)

    # here stdin will close
# here the vw process will have finished

# Inside with predicting() we can stream instances and acquire their labels
with vw.predicting():
    for instance in ['1 |this is another positive example', '0 |this is another negative example']:
        vw.push_instance(instance)

# Read the predictions like this:
predictions = list(vw.read_predictions_())
```

L-BFGS with a rank-5 approximation:
```python
from vowpal_porpoise import VW

# Initialize the model
vw = VW(moniker='test_lda',  # a name for the model
        passes=10,           # vw arg: passes
        lbfgs=True,          # turn on lbfgs
        mem=5)               # lbfgs rank
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



### Library Interace (TESTING)

Raw library interface. Currently does not support passes due to some limitations in the underlying vw C code.
```python
import vw_c
vw = vw_c.VW("--loss=quadratic --l1=0.01 -f model")
vw.learn("1 |this is a positive example")
vw.learn("0 |this is a negative example")
vw.finish()
```

You can also access this code via the ```VW``` interface for prediction like this:
```python
with vw.predicting_library():
    for instance in ['1 |this is another positive example', '0 |this is another negative example']:
        prediction = vw.push_instance(instance)
```
see ```examples/example1.py``` for more details.

### Need more examples?

* ```example1.py```: SimpleModel class wrapper around VP (both standard and library flavors)
* ```example_library.py```: Demonstrates the low-level vw library wrapper, classifying lines of ``alice in wonderland'' vs ``through the looking glass''.

## How it works

Wraps the vw binary in a subprocess and uses stdin to push data, temporary files to pull predictions. Alterantively, you can use a pure api call (wrapping libvw) for prediction.


## Contact

Joseph Reisinger (joeraii@gmail.com)

## Contributors

* Austin Waters (austin.waters@gmail.com)
* Joseph Reisinger (joeraii@gmail.com)

## License

Apache 2.0
