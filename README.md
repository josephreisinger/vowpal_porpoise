# vowpal_porpoise v0.2

Lightweight wrapper for vowpal wabbit:

## Install

1. First install vw in your path. Git clone it here: https://github.com/JohnLangford/vowpal_wabbit/
2. Also make sure that cython is installed
```bash
pip install cython
```
3. To install vowpal_porpoise, run:
```bash
python setup.py install
```
4. Now can you do:
```python
import vowpal_porpoise
```
from python.

## Examples

example1.py: SimpleModel class wrapper around VP
example2.py: Demonstrates the vw library wrapper, classifying alice in wonderland.

## How it works

Wraps the vw binary in a subprocess and uses stdin to push data, temporary files to pull predictions. Alterantively, you can use a pure api call (wrapping libvw) for prediction.


## Contributors

Austin Waters (austin.waters@gmail.com)
Joseph Reisinger (joeraii@gmail.com)

## License

Apache 2.0
