import numpy
import random
from sklearn.base import BaseEstimator, ClassifierMixin

class BinomialClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, p):
        self.p = p

    def fit(self, X, y=None):
        return self

    def _binomial(self):
        return 1 if random.random() < self.p else 2

    def predict(self, X, y=None):
        random.seed(2)
        X = numpy.asarray(X)
        return [self._binomial() for _ in range(X.shape[0])]
