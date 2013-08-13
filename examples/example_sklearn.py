"""
This example showcases how to uses scikit-learn's cross-validation tools with
VW_Classifier. We load the MNIST digits dataset, convert it to VW_Classifier's
expected input format, then classify the digit as < 5 or >= 5.
"""
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.datasets import load_digits
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from vowpal_porpoise.sklearn import VW_Classifier


class Array2Dict(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_features = np.shape(X)
        result = []
        for i in range(n_samples):
            result.append({
                          str(j): X[i, j]
                          for j in range(n_features)
                          if X[i, j] != 0
                          })
        return result


def main():
    # parameters to cross-validate over
    parameters = {
        'l2': np.logspace(-5, 0, num=6),
    }

    # load iris data in, make a binary decision problem out of it
    data = load_digits()

    X = Array2Dict().fit_transform(data.data)
    y = 2 * (data.target >= 5) - 1

    i = int(0.8 * len(X))
    X_train, X_test = X[:i], X[i:]
    y_train, y_test = y[:i], y[i:]

    # do the actual learning
    gs = GridSearchCV(
        VW_Classifier(loss='logistic', moniker='example_sklearn',
                      passes=10, silent=True, learning_rate=10),
        param_grid=parameters,
        score_func=f1_score,
        cv=StratifiedKFold(y_train),
    ).fit(X_train, y_train)

    # print out results from cross-validation
    estimator = gs.best_estimator_
    score = gs.best_score_
    print 'Achieved a F1 score of %f using l2 == %f during cross-validation' % (score, estimator.l2)

    # print confusion matrix on test data
    y_est = estimator.fit(X_train, y_train).predict(X_test)
    print 'Confusion Matrix:'
    print confusion_matrix(y_test, y_est)


if __name__ == '__main__':
    main()
