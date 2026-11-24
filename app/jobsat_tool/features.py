import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAC = ["FACADV","FACBEN","FACCHAL","FACIND","FACLOC","FACRESP","FACSAL","FACSEC","FACSOC"]
PERSON = ["IMP_" + c[3:] for c in FAC]            # 1 = not important ... 4 = very important
JOB = ["ONET_WV_achievement","ONET_WV_working_conditions","ONET_WV_recognition",
       "ONET_WV_relationships","ONET_WV_support","ONET_WV_independence"]
DIM_MAP = {"achievement": ["IMP_CHAL"], "independence": ["IMP_IND", "IMP_RESP"],
           "recognition": ["IMP_ADV"], "working_conditions": ["IMP_SAL", "IMP_SEC", "IMP_BEN"],
           "relationships": ["IMP_SOC"]}
DIMS = list(DIM_MAP)

class FitFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, kinds=("abs", "prod")):
        self.kinds = kinds
    def _pj(self, X):
        P = np.column_stack([X[DIM_MAP[d]].mean(axis=1) for d in DIMS])
        J = X[["ONET_WV_" + d for d in DIMS]].values.astype(float)
        return P, J
    def fit(self, X, y=None):
        P, J = self._pj(X)
        self.pm_, self.ps_ = P.mean(0), P.std(0) + 1e-9
        self.jm_, self.js_ = J.mean(0), J.std(0) + 1e-9
        return self
    def transform(self, X):
        P, J = self._pj(X)
        P, J = (P - self.pm_) / self.ps_, (J - self.jm_) / self.js_
        out = []
        for k in self.kinds:
            if k == "abs":    out.append(np.abs(P - J))
            elif k == "prod": out.append(P * J)
        return np.hstack(out)

def make_product_pipe():
    pre = ColumnTransformer([
        ("person", StandardScaler(), PERSON),
        ("job", StandardScaler(), JOB),
        ("fit", Pipeline([("f", FitFeatures()), ("s", StandardScaler())]), PERSON + JOB)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])