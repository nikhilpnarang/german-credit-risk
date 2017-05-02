import numpy
from defines import Types
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class NumericScaler(BaseEstimator, TransformerMixin):

    def __init__(self, n_numeric, **params):
        self.n_numeric = n_numeric
        self.scaler = StandardScaler(**params)

    def fit(self, X, y=None):
        X = numpy.asarray(X)
        self.scaler.fit(X[:, :self.n_numeric], y)
        return self

    def transform(self, X):
        X = numpy.asarray(X)
        X_head = self.scaler.transform(X[:, :self.n_numeric])
        return numpy.concatenate([X_head, X[:, self.n_numeric:]], axis=1)
