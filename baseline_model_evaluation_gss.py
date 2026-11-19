"""Baseline model evaluation on the General Social Survey (GSS) 2024.

This is a copy of baseline_model_evaluation.py (NSCG 2023) adapted to the GSS,
using the variables selected in notebooks/GSS_Exploratory_Data_Analysis.ipynb.
The models, metrics and threshold rule are unchanged, so the two baselines can
be compared side by side.

What carries over from the GSS EDA, and what is done differently:
- Same variable shortlist (satjob, wrkstat, hrs1, realrinc, prestg10, sei10,
  union, age, sex, race, degree, marital, region, class, ...).
- realrinc is used for income; conrinc is the same measure in different dollars,
  and income/income16 are household brackets, so they are left out.
- Missing values are filled with the median (numeric) or mode (categorical), as
  in the EDA, but inside the model pipeline so they are learned from the
  training rows only. Numeric features also get a "was missing" indicator,
  because realrinc is missing for about 16% of workers.
- Rows with no satjob answer are dropped, not imputed. The EDA filled them with
  "moderately satisfied", which would add ~1,000 fake labels.
- The target is not winsorized: satjob is a 4-point scale (very satisfied,
  moderately satisfied, a little dissatisfied, very dissatisfied).
- The sample is limited to people working full or part time, so it matches the
  NSCG sample of employed graduates. satjob is also asked of people who are
  unemployed or keeping house, about their past or household work.

The workflow is intentionally simple and reproducible:
1. Create a baseline feature set using only background factors, plus a second
   set that adds the GSS job-context variables from the EDA.
2. Split the data with stratification to respect class imbalance.
3. Compare several baseline models: dummy, logistic regression, and tree-based
   learners.
4. Report metrics that do not reward a trivial all-satisfied classifier.
5. Summarize qualitative properties that cover accuracy, satisfaction signal,
   and group sensitivity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
N_BOOT = 1000
ROOT = Path(__file__).resolve().parent
EXTRACT_PATH = ROOT / "data" / "gss" / "gss2024_extract.csv"
# Cumulative GSS 1972-2024 file (Release 3a), SAS format, from gss.norc.org.
DEFAULT_SAS_PATH = Path.home() / "Downloads" / "GSS_sas" / "gss7224_r3a.sas7bdat"

EXTRACT_COLUMNS = [
    "YEAR", "ID", "WTSSPS", "SATJOB", "WRKSTAT", "HRS1", "REALRINC", "AGE", "SEX",
    "RACE", "HISPANIC", "DEGREE", "MARITAL", "WRKSLF", "WRKGOVT1", "WRKGOVT2",
    "WKSUP", "PRESTG10", "SEI10", "REGION", "CLASS", "UNION",
]

# GSS codebook values (GSS 2024 Codebook R3a).
SATJOB_LABELS = {1: "Very satisfied", 2: "Moderately satisfied", 3: "A little dissatisfied", 4: "Very dissatisfied"}
SEX_LABELS = {1: "Male", 2: "Female"}
RACE_LABELS = {1: "White", 2: "Black", 3: "Other"}
DEGREE_LABELS = {0: "Less than high school", 1: "High school", 2: "Associate", 3: "Bachelor", 4: "Graduate"}
REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
CLASS_LABELS = {1: "Lower", 2: "Working", 3: "Middle", 4: "Upper"}
WORKING = [1, 2]  # WRKSTAT: working full time, working part time

# Background factors, matched to the NSCG baseline where the GSS has an equivalent.
# NSCG employer size has no GSS 2024 equivalent.
NUMERIC_FEATURES = [
    "log_REALRINC",  # log1p(REALRINC); NSCG: log_SALARY
    "HRS1",  # hours worked last week; NSCG: HRSWK
    "AGE",
]
CATEGORICAL_FEATURES = [
    "sex",
    "race_ethnicity",  # RACE + HISPANIC; NSCG: RACETHM
    "degree",  # NSCG: DGRDG
    "sector",  # WRKSLF + WRKGOVT1/2; NSCG: EMSECSM
    "supervises",  # WKSUP; NSCG: SUPWK
    "married",  # NSCG: MARIND
]

# Job-context variables from the GSS EDA that NSCG does not have.
# satfin (financial satisfaction) and richwork (keep working if rich) are left
# out: both are attitudes close to the target rather than background factors.
CONTEXT_NUMERIC = ["PRESTG10", "SEI10"]  # occupational prestige, socioeconomic index
CONTEXT_CATEGORICAL = ["region", "class", "union"]

FEATURE_SETS: Dict[str, Tuple[List[str], List[str]]] = {
    "baseline": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "baseline_plus_gss_context": (NUMERIC_FEATURES + CONTEXT_NUMERIC, CATEGORICAL_FEATURES + CONTEXT_CATEGORICAL),
}

QUALITATIVE_PROPERTIES = [
    "Predictive accuracy: does the model separate satisfied from unsatisfied people?",
    "Satisfaction signal: does it capture meaningful variation without overpredicting the majority class?",
    "Group sensitivity: does performance remain consistent across demographic or work-status groups?",
]


def extract_gss_2024(sas_path: Path, out_path: Path = EXTRACT_PATH) -> None:
    """Pull the 2024 respondents and the columns used here out of the cumulative GSS file."""
    import pyreadstat

    if not sas_path.exists():
        raise FileNotFoundError(
            f"{sas_path} not found. Download the cumulative GSS SAS file from gss.norc.org "
            "or pass --sas-path."
        )
    raw, _ = pyreadstat.read_sas7bdat(str(sas_path), usecols=EXTRACT_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw[raw["YEAR"] == 2024].to_csv(out_path, index=False)


def _label(series: pd.Series, labels: Dict[int, str]) -> pd.Series:
    return series.map(labels)


def _yes_no(series: pd.Series) -> pd.Series:
    return series.map({1: "Yes", 2: "No"})


def load_gss(path: Path = EXTRACT_PATH) -> pd.DataFrame:
    """Load the GSS 2024 extract, keep workers who answered satjob, and build the features."""
    raw = pd.read_csv(path)
    df = raw[raw["SATJOB"].notna() & raw["WRKSTAT"].isin(WORKING)].copy()

    df["log_REALRINC"] = np.log1p(df["REALRINC"])
    df["sex"] = _label(df["SEX"], SEX_LABELS)
    df["race_ethnicity"] = np.where(df["HISPANIC"] > 1, "Hispanic", _label(df["RACE"], RACE_LABELS))
    df.loc[df["RACE"].isna() & df["HISPANIC"].isna(), "race_ethnicity"] = np.nan
    df["degree"] = _label(df["DEGREE"], DEGREE_LABELS)
    df["sector"] = np.select(
        [df["WRKSLF"] == 1, df["WRKGOVT1"] == 1, df["WRKGOVT2"] == 1],
        ["Self-employed", "Government", "Private"],
        default="Nonprofit/other",
    )
    df.loc[df["WRKSLF"].isna() & df["WRKGOVT1"].isna() & df["WRKGOVT2"].isna(), "sector"] = np.nan
    df["supervises"] = _yes_no(df["WKSUP"])
    df["married"] = np.where(df["MARITAL"] == 1, "Yes", "No")
    df.loc[df["MARITAL"].isna(), "married"] = np.nan
    df["region"] = _label(df["REGION"], REGION_LABELS)
    df["class"] = _label(df["CLASS"], CLASS_LABELS)
    # UNION is only asked on some ballots, so a missing answer is its own category.
    df["union"] = df["UNION"].map({1: "Member", 3: "Member", 2: "Not member", 4: "Not member"}).fillna("Not asked")

    df["job_satisfaction"] = _label(df["SATJOB"], SATJOB_LABELS)
    # The minority class (a little or very dissatisfied, ~13%) is the one we want to find.
    df["not_satisfied"] = df["SATJOB"].isin([3, 4]).astype(int)
    return df.reset_index(drop=True)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_models() -> Dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "hist_gradient_boost": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Cut-off that maximizes TPR - FPR (balanced accuracy) on training predictions."""
    if len(np.unique(y_score)) < 2:
        return 1.0
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # roc_curve's first threshold is inf; cap it so "flag nobody" reads as 1.0.
    return float(min(thr[np.argmax(tpr - fpr)], 1.0))


def compute_metric_summary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOT) -> Tuple[float, float]:
    """95% CI for test ROC AUC; each resample keeps the class ratio. The GSS test set is small."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_group_sensitivity(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    group_col: str = "sex",
) -> pd.DataFrame:
    """Compare observed dissatisfaction, flag rate, recall and within-group AUC by group."""
    df = pd.DataFrame(
        {"group": X_test[group_col].to_numpy(), "y_true": y_true, "y_score": y_score, "y_pred": y_pred}
    )
    rows = []
    for level, g in df.groupby("group"):
        has_both = g["y_true"].nunique() == 2
        rows.append(
            {
                group_col: level,
                "n": len(g),
                "observed_not_satisfied_rate": g["y_true"].mean(),
                "flagged_rate": g["y_pred"].mean(),
                "recall": g.loc[g["y_true"] == 1, "y_pred"].mean() if g["y_true"].any() else np.nan,
                "roc_auc": roc_auc_score(g["y_true"], g["y_score"]) if has_both else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_satisfaction_levels(levels: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    """Mean risk score for each of the 4 original satisfaction answers."""
    order = list(SATJOB_LABELS.values())
    return (
        pd.DataFrame({"job_satisfaction": levels.to_numpy(), "risk": y_score})
        .groupby("job_satisfaction")["risk"]
        .agg(n="size", mean_risk="mean")
        .reindex(order)
        .reset_index()
    )


def evaluate_baseline_models(
    df: pd.DataFrame,
    target_col: str = "not_satisfied",
    group_col: str = "sex",
    test_size: float = 0.2,
    feature_sets: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
) -> Dict[str, object]:
    """Run a reproducible baseline evaluation for each feature set on the same split."""
    feature_sets = feature_sets or FEATURE_SETS
    y = df[target_col].astype(int)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=y,
        random_state=SEED,
    )
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    result = {}
    for set_name, (numeric_cols, categorical_cols) in feature_sets.items():
        X_train = df.loc[train_idx, numeric_cols + categorical_cols]
        X_test = df.loc[test_idx, numeric_cols + categorical_cols]
        for name, model in build_models().items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                    ("model", model),
                ]
            )
            # Out-of-fold training predictions give the CV AUC and the decision threshold,
            # so the test set is never used to tune anything.
            oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            threshold = youden_threshold(y_train.to_numpy(), oof)

            pipeline.fit(X_train, y_train)
            y_score = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= threshold).astype(int)
            result[(set_name, name)] = {
                "cv_roc_auc": roc_auc_score(y_train, oof),
                "threshold": threshold,
                "metrics": compute_metric_summary(y_test.to_numpy(), y_pred, y_score),
                "roc_auc_ci": bootstrap_auc_ci(y_test.to_numpy(), y_score) if name != "dummy_majority" else (0.5, 0.5),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "group_summary": summarize_group_sensitivity(
                    df.loc[test_idx], y_test.to_numpy(), y_score, y_pred, group_col=group_col
                ),
                "satisfaction_levels": summarize_satisfaction_levels(df.loc[test_idx, "job_satisfaction"], y_score),
            }

    return {
        "qualitative_properties": QUALITATIVE_PROPERTIES,
        "target_share": float(y.mean()),
        "train_test_size": {"train": len(train_idx), "test": len(test_idx), "test_not_satisfied": int(y_test.sum())},
        "results": result,
    }


def results_table(report: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for (set_name, name), payload in report["results"].items():
        lo, hi = payload["roc_auc_ci"]
        rows.append(
            {
                "features": set_name,
                "model": name,
                "cv_roc_auc": payload["cv_roc_auc"],
                **payload["metrics"],
                "roc_auc_95ci": f"[{lo:.3f}, {hi:.3f}]",
                "threshold": payload["threshold"],
            }
        )
    return pd.DataFrame(rows)


def print_report(report: Dict[str, object]) -> None:
    print("Baseline model evaluation (GSS 2024)")
    print("=" * 36)
    print("Qualitative properties:")
    for idx, prop in enumerate(report["qualitative_properties"], start=1):
        print(f"  {idx}. {prop}")
    print(f"\nTarget share (not satisfied): {report['target_share']:.1%}")
    print(f"Train/test split: {report['train_test_size']}")

    print("\nModel comparison (test set; positive class = not satisfied)")
    print("-" * 36)
    table = results_table(report)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))

    best = ("baseline", "hist_gradient_boost")
    print("\nRisk score by actual satisfaction answer (baseline, gradient boosting):")
    print(report["results"][best]["satisfaction_levels"].round(3).to_string(index=False))
    print("\nGroup sensitivity check (baseline, gradient boosting, by sex):")
    print(report["results"][best]["group_summary"].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sas-path", type=Path, default=DEFAULT_SAS_PATH,
                        help="cumulative GSS .sas7bdat file, used once to build the 2024 extract")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "baseline_results_gss.csv")
    args = parser.parse_args()

    if not EXTRACT_PATH.exists():
        print(f"Building {EXTRACT_PATH.relative_to(ROOT)} from {args.sas_path} ...")
        extract_gss_2024(args.sas_path)

    df = load_gss()
    report = evaluate_baseline_models(df)
    print_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_table(report).to_csv(args.out, index=False)
    print(f"\nSaved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
