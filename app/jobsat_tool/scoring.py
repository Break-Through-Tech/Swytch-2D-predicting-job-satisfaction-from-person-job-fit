from pathlib import Path
import joblib, numpy as np, pandas as pd
from .features import PERSON, JOB, DIM_MAP, DIMS

ART = Path(__file__).resolve().parent.parent / "artifacts"
model = joblib.load(ART / "model.joblib")
occ = pd.read_csv(ART / "occupations.csv", index_col=0)
_ff = model.named_steps["pre"].named_transformers_["fit"].named_steps["f"]   # training z-score stats

def _frame(values: dict) -> pd.DataFrame:
    X = occ[JOB].copy()
    for k in PERSON:
        X[k] = values[k]
    return X

def score_all(values: dict) -> pd.Series:
    """Higher = more likely satisfied (log-odds). Used for ranking, not shown as a probability."""
    return pd.Series(model.decision_function(_frame(values)), index=occ.index)

def label(values: dict, pid) -> str:
    pct = score_all(values).rank(pct=True)[pid]
    return "Strong fit" if pct >= 0.67 else "Moderate fit" if pct >= 0.33 else "Weaker fit"

def breakdown(values: dict, pid) -> list[dict]:
    P = np.array([np.mean([values[k] for k in DIM_MAP[d]]) for d in DIMS])
    J = occ.loc[pid, ["ONET_WV_" + d for d in DIMS]].values.astype(float)
    P, J = (P - _ff.pm_) / _ff.ps_, (J - _ff.jm_) / _ff.js_
    return [{"dimension": d, "you": float(p), "job": float(j), "gap": float(p - j)} for d, p, j in zip(DIMS, P, J)]

def explain(bd: list[dict], top: int = 2) -> list[str]:
    msgs = []
    for r in sorted(bd, key=lambda r: abs(r["gap"]), reverse=True)[:top]:
        d = r["dimension"].replace("_", " ")
        if r["gap"] > 0.5:    msgs.append(f"You rate {d} higher than this occupation typically offers.")
        elif r["gap"] < -0.5: msgs.append(f"This occupation offers more {d} than you prioritize.")
        else:                 msgs.append(f"{d.capitalize()} looks like a reasonable match.")
    return msgs

def recommend(values: dict, k: int = 10) -> list[dict]:
    s = score_all(values).sort_values(ascending=False).head(k)
    out = []
    for pid in s.index:
        bd = breakdown(values, pid)
        out.append({"profile_id": pid, "occupation": occ.loc[pid, "title"], "label": label(values, pid),
                    "breakdown": bd, "why": explain(bd)})
    return out