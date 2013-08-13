from __future__ import absolute_import
import sklearn.base
import numpy as np

import vowpal_porpoise

# if you try to say "import sklearn", it thinks sklearn is this file :(


class _VW(sklearn.base.BaseEstimator):
  """scikit-learn interface for Vowpal Wabbit

  Only works for regression and binary classification.
  """

  def __init__(self,
        logger              = None,
        vw                  = 'vw',
        moniker             = None,
        name                = None,
        bits                = None,
        loss                = None,
        passes              = None,
        log_stderr_to_file  = False,
        silent              = False,
        l1                  = None,
        l2                  = None,
        learning_rate       = None,
        quadratic           = None,
        audit               = None,
        power_t             = None,
        adaptive            = False,
        working_dir         = None,
        decay_learning_rate = None,
        initial_t           = None,
        minibatch           = None,
        total               = None,
        node                = None,
        unique_id           = None,
        span_server         = None,
        bfgs                = None,
        oaa                 = None,
        old_model           = None,
        incremental         = False,
        mem                 = None
      ):
    self.vw = vowpal_porpoise.VW(
        logger              = logger,
        vw                  = vw,
        moniker             = moniker,
        name                = name,
        bits                = bits,
        loss                = loss,
        passes              = passes,
        log_stderr_to_file  = log_stderr_to_file,
        silent              = silent,
        l1                  = l1,
        l2                  = l2,
        learning_rate       = learning_rate,
        quadratic           = quadratic,
        audit               = audit,
        power_t             = power_t,
        adaptive            = adaptive,
        working_dir         = working_dir,
        decay_learning_rate = decay_learning_rate,
        initial_t           = initial_t,
        minibatch           = minibatch,
        total               = total,
        node                = node,
        unique_id           = unique_id,
        span_server         = span_server,
        bfgs                = bfgs,
        oaa                 = oaa,
        old_model           = old_model,
        incremental         = incremental,
        mem                 = mem,
    )


  def fit(self, X, y):
    """Fit Vowpal Wabbit

    Parameters
    ----------
    X: [{<feature name>: <feature value>}]
        input features
    y: [int or float]
        output labels
    """
    examples = _as_vw_strings(X, y)

    # clear out old model
    # XXX

    # add examples to model
    with self.vw.training():
      for instance in examples:
        self.vw.push_instance(instance)

    # learning done after "with" statement
    return self

  def predict(self, X):
    """Fit Vowpal Wabbit

    Parameters
    ----------
    X: [{<feature name>: <feature value>}]
        input features
    """
    examples = _as_vw_strings(X)

    # add test examples to model
    with self.vw.predicting():
      for instance in examples:
        self.vw.push_instance(instance)

    # read out predictions
    predictions = np.asarray(list(self.vw.read_predictions_()))

    return predictions


class VW_Regressor(sklearn.base.RegressorMixin, _VW):
  pass


class VW_Classifier(sklearn.base.ClassifierMixin, _VW):

  def predict(self, X):
    result = super(VW_Classifier, self).predict(X)
    result = 2 * (result >= 0) - 1
    return result


def _as_vw_string(x, y=None):
  """Convert {feature: value} to something _VW understands

  Parameters
  ----------
  x : {<feature>: <value>}
  y : int or float
  """
  result = str(y)
  x = " ".join(["%s:%f" % (key, value) for (key, value) in x.items()])
  return result + " | " + x


def _as_vw_strings(X, y=None):
  n_samples = np.shape(X)[0]
  if y is None:
    y = np.ones(n_samples)
  return [_as_vw_string(X[i], y[i]) for i in range(n_samples)]
