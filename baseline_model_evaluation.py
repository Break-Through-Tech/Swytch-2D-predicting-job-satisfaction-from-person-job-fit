"""Baseline model evaluation for predicting job satisfaction from background features.

This script implements the October milestone evaluation setup for the Swytch
project. It keeps the crosswalk and occupation-linking work separate from the
baseline model, and it focuses on a fair comparison when the satisfied and
not-satisfied groups are imbalanced.

It runs on the cleaned NSCG 2023 file (data/college_grads_2023/nscg2023_clean.csv,
79,800 employed college graduates) and mirrors the definitions in
notebooks/Baseline_Model_Evaluation.ipynb, so its numbers match
results/baseline_results.csv.

The workflow is intentionally simple and reproducible:
1. Create a baseline feature set using only background factors.
2. Split the data with stratification to respect class imbalance.
3. Compare several baseline models: dummy, logistic regression, and tree-based
   learners.
4. Report metrics that do not reward a trivial all-satisfied classifier.
5. Summarize qualitative properties that cover accuracy, satisfaction signal,
   and group sensitivity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
DATA_PATH = Path(__file__).resolve().parent / "data" / "college_grads_2023" / "nscg2023_clean.csv"

# NSCG 2023 column names. OCEDRLP (job related to degree) is left out on purpose:
# it is a crude person-job fit measure and belongs with the fit features.
NUMERIC_FEATURES = [
    "log_SALARY",  # log1p(SALARY); salary is very right-skewed
    "HRSWK",  # hours worked per week
    "AGE",
]
CATEGORICAL_FEATURES = [
    "SEX_2023",
    "RACETHM",  # race/ethnicity
    "DGRDG",  # highest degree
    "EMSECSM",  # employer sector
    "EMSIZE",  # employer size
    "SUPWK",  # supervises others
    "MARIND",  # married
]
BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Codebook labels (Ppcg23.sas), used only for the readable group table.
SEX_LABELS = {"F": "Female", "M": "Male"}

QUALITATIVE_PROPERTIES = [
    "Predictive accuracy: does the model separate satisfied from unsatisfied people?",
    "Satisfaction signal: does it capture meaningful variation without overpredicting the majority class?",
    "Group sensitivity: does performance remain consistent across demographic or work-status groups?",
]


def load_nscg(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned NSCG file and add the model target and derived features.

    NSCG imputes item non-response before release, so there are no missing
    values to handle.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/NSCG_2023_Exploratory_Data_Analysis.ipynb to create it."
        )
    df = pd.read_csv(path, low_memory=False)
    df["log_SALARY"] = np.log1p(df["SALARY"])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    # The minority class (somewhat or very dissatisfied, ~9.9%) is the one we want to find.
    df["not_satisfied"] = 1 - df["satisfied"]
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
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
    return float(thr[np.argmax(tpr - fpr)])


def compute_metric_summary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def summarize_group_sensitivity(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    group_col: str = "SEX_2023",
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
                group_col: SEX_LABELS.get(level, level) if group_col == "SEX_2023" else level,
                "n": len(g),
                "observed_not_satisfied_rate": g["y_true"].mean(),
                "flagged_rate": g["y_pred"].mean(),
                "recall": g.loc[g["y_true"] == 1, "y_pred"].mean() if g["y_true"].any() else np.nan,
                "roc_auc": roc_auc_score(g["y_true"], g["y_score"]) if has_both else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_baseline_models(
    df: pd.DataFrame,
    target_col: str = "not_satisfied",
    group_col: str = "SEX_2023",
    test_size: float = 0.2,
) -> Dict[str, object]:
    """Run a reproducible baseline evaluation on a background-only feature set."""
    X = df[BASELINE_FEATURES]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    result = {}
    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
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
        result[name] = {
            "cv_roc_auc": roc_auc_score(y_train, oof),
            "threshold": threshold,
            "metrics": compute_metric_summary(y_test.to_numpy(), y_pred, y_score),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "group_summary": summarize_group_sensitivity(
                X_test, y_test.to_numpy(), y_score, y_pred, group_col=group_col
            ),
        }

    return {
        "qualitative_properties": QUALITATIVE_PROPERTIES,
        "target_share": float(y.mean()),
        "train_test_shape": {"train": X_train.shape, "test": X_test.shape},
        "results": result,
    }


def print_report(report: Dict[str, object]) -> None:
    print("Baseline model evaluation (NSCG 2023)")
    print("=" * 37)
    print("Qualitative properties:")
    for idx, prop in enumerate(report["qualitative_properties"], start=1):
        print(f"  {idx}. {prop}")
    print(f"\nTarget share (not satisfied): {report['target_share']:.1%}")
    print(f"Train/test split: {report['train_test_shape']}")

    print("\nModel comparison (test set; positive class = not satisfied)")
    print("-" * 37)
    for name, payload in report["results"].items():
        print(f"{name}:")
        print(f"  cv_roc_auc: {payload['cv_roc_auc']:.3f}")
        for metric_name, value in payload["metrics"].items():
            print(f"  {metric_name}: {value:.3f}")
        print(f"  threshold: {payload['threshold']:.3f}")
        print()

    print("Group sensitivity check (gradient boosting, by sex):")
    print(report["results"]["hist_gradient_boost"]["group_summary"].round(3).to_string(index=False))


def main() -> None:
    df = load_nscg()
    report = evaluate_baseline_models(df)
    print_report(report)


if __name__ == "__main__":
    main()
