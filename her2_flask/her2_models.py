"""Model wrappers shared by train_models.py and app.py (needed to load the saved models)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline


def mi_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


class ColumnModel(BaseEstimator, ClassifierMixin):
    """Applies a model to a fixed subset of the 86 SISH-aware features (optionally with MI feature selection)."""

    def __init__(self, model=None, feat_cols=None, k_select=None):
        self.model, self.feat_cols, self.k_select = model, feat_cols, k_select

    def _make(self):
        m = clone(self.model)
        if self.k_select:
            m = Pipeline([("select", SelectKBest(mi_score, k=self.k_select)), ("model", m)])
        return m

    def fit(self, X, y):
        self.m_ = self._make().fit(np.asarray(X)[:, self.feat_cols], y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X)[:, self.feat_cols])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
