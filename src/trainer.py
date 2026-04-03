"""
Baseline router comparison: heuristic Always-RAG vs default sklearn / XGBoost models.

Loads `data/router_training_matrix.pkl` (from `prepare_dataset.py`), uses all `feat_*`
columns to predict `target_label` (1 = model wrong → prefer RAG).

Metrics: Train F1, 5-fold CV F1 (stratified, tqdm), Test F1, Test ROC-AUC.

XGBoost additionally runs GridSearchCV over depth, learning rate, n_estimators, subsample,
and L1/L2 regularization (reg_alpha / reg_lambda), saves `outputs/feature_importance.png`,
and prints unoptimized vs tuned metrics.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

warnings.filterwarnings("ignore", category=UserWarning)


def load_training_matrix(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return pd.DataFrame(obj)
    raise RuntimeError(f"Unsupported PKL content from {path}: {type(obj)}")


def feat_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("feat_")]
    if not cols:
        raise RuntimeError("No columns starting with 'feat_' found.")
    return sorted(cols)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC for binary labels; returns nan if undefined (e.g. single class in y_true)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def cross_val_f1_stratified(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: StratifiedKFold,
    desc: str,
) -> float:
    """5-fold stratified CV mean F1 with tqdm over folds."""
    scores: list[float] = []
    splits = list(cv.split(X, y))
    for train_idx, val_idx in tqdm(splits, total=len(splits), desc=desc):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        est = clone(estimator)
        est.fit(X_tr, y_tr)
        pred = est.predict(X_va)
        scores.append(f1_score(y_va, pred, zero_division=0))
    return float(np.mean(scores))


def xgb_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost imbalance weight: (#negative) / (#positive) for label 1 = positive."""
    y_arr = np.asarray(y).astype(int).ravel()
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def plot_xgb_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    *,
    top_k: int = 25,
) -> None:
    """Save a horizontal bar chart of XGBoost feature importances."""
    if plt is None:
        raise RuntimeError("matplotlib is required for feature importance plots: pip install matplotlib")

    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
    ax.barh(range(len(names)), vals[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance (XGBoost default)")
    ax.set_title("XGBoost (tuned) — feature importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'Strategy/Model':<34} {'Train F1':>10} {'CV F1':>10} {'Test F1':>10} {'Test AUC':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        name = str(r["name"])[:33]
        def fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'N/A':>10}"
            if isinstance(v, (float, np.floating)):
                return f"{float(v):10.4f}"
            return f"{v!s:>10}"

        print(
            f"{name:<34} {fmt(r['train_f1'])} {fmt(r['cv_f1'])} {fmt(r['test_f1'])} {fmt(r['test_auc'])}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline router training / comparison")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/router_training_matrix.pkl"),
        help="PKL from prepare_dataset.py (DataFrame)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Split + CV seed")
    parser.add_argument(
        "--feature-importance-out",
        type=Path,
        default=Path("outputs/feature_importance.png"),
        help="Path for tuned XGBoost feature importance plot",
    )
    args = parser.parse_args()

    df = load_training_matrix(args.data)
    if "target_label" not in df.columns:
        raise RuntimeError("Column 'target_label' missing. Run prepare_dataset.py first.")

    feat_cols = feat_columns(df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["target_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Loaded: {args.data} | samples={len(df)} | features={len(feat_cols)}")
    print(
        f"Train={len(X_train)} Test={len(X_test)} | "
        f"pos_rate train={y_train.mean():.4f} test={y_test.mean():.4f}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # Numeric pipeline: impute then model (trees/LR); LR also scales continuous features.
    imputer = SimpleImputer(strategy="median")

    lr_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    rf_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise RuntimeError("Install xgboost: pip install xgboost") from e

    spw = xgb_scale_pos_weight(y_train)
    xgb_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "clf",
                XGBClassifier(
                    random_state=args.random_state,
                    scale_pos_weight=spw,
                    objective="binary:logistic",
                    n_estimators=100,
                ),
            ),
        ]
    )

    rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Heuristic: Always-RAG (always predict positive class 1)
    # -------------------------------------------------------------------------
    train_f1_always = f1_score(y_train, np.ones(len(y_train), dtype=int), zero_division=0)
    cv_f1_always: list[float] = []
    for _, val_idx in tqdm(
        list(cv.split(X_train, y_train)),
        total=cv.get_n_splits(),
        desc="5-fold CV | Always-RAG",
    ):
        y_va = y_train.iloc[val_idx]
        cv_f1_always.append(f1_score(y_va, np.ones(len(y_va), dtype=int), zero_division=0))
    test_f1_always = f1_score(y_test, np.ones(len(y_test), dtype=int), zero_division=0)
    test_auc_always = safe_roc_auc(y_test.values, np.ones(len(y_test), dtype=float))

    rows.append(
        {
            "name": "Always-RAG (predict 1)",
            "train_f1": train_f1_always,
            "cv_f1": float(np.mean(cv_f1_always)),
            "test_f1": test_f1_always,
            "test_auc": test_auc_always,
        }
    )

    # -------------------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------------------
    lr_pipe.fit(X_train, y_train)
    train_f1_lr = f1_score(y_train, lr_pipe.predict(X_train), zero_division=0)
    cv_f1_lr = cross_val_f1_stratified(
        lr_pipe,
        X_train,
        y_train,
        cv=cv,
        desc="5-fold CV | Logistic Regression",
    )
    test_f1_lr = f1_score(y_test, lr_pipe.predict(X_test), zero_division=0)
    test_auc_lr = safe_roc_auc(y_test.values, lr_pipe.predict_proba(X_test)[:, 1])

    rows.append(
        {
            "name": "Logistic Regression (balanced)",
            "train_f1": train_f1_lr,
            "cv_f1": cv_f1_lr,
            "test_f1": test_f1_lr,
            "test_auc": test_auc_lr,
        }
    )

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------
    rf_pipe.fit(X_train, y_train)
    train_f1_rf = f1_score(y_train, rf_pipe.predict(X_train), zero_division=0)
    cv_f1_rf = cross_val_f1_stratified(
        rf_pipe,
        X_train,
        y_train,
        cv=cv,
        desc="5-fold CV | Random Forest",
    )
    test_f1_rf = f1_score(y_test, rf_pipe.predict(X_test), zero_division=0)
    test_auc_rf = safe_roc_auc(y_test.values, rf_pipe.predict_proba(X_test)[:, 1])

    rows.append(
        {
            "name": "Random Forest (balanced)",
            "train_f1": train_f1_rf,
            "cv_f1": cv_f1_rf,
            "test_f1": test_f1_rf,
            "test_auc": test_auc_rf,
        }
    )

    # -------------------------------------------------------------------------
    # XGBoost (unoptimized baseline — often overfits on tabular router features)
    # -------------------------------------------------------------------------
    xgb_pipe.fit(X_train, y_train)
    train_f1_xgb = f1_score(y_train, xgb_pipe.predict(X_train), zero_division=0)
    cv_f1_xgb = cross_val_f1_stratified(
        xgb_pipe,
        X_train,
        y_train,
        cv=cv,
        desc="5-fold CV | XGBoost (unoptimized)",
    )
    test_f1_xgb = f1_score(y_test, xgb_pipe.predict(X_test), zero_division=0)
    test_auc_xgb = safe_roc_auc(y_test.values, xgb_pipe.predict_proba(X_test)[:, 1])

    rows.append(
        {
            "name": f"XGBoost unopt (spw={spw:.3f})",
            "train_f1": train_f1_xgb,
            "cv_f1": cv_f1_xgb,
            "test_f1": test_f1_xgb,
            "test_auc": test_auc_xgb,
        }
    )

    print("\n=== Baseline comparison (Always-RAG, LR, RF, XGB unoptimized) ===")
    print_results_table(rows)

    # -------------------------------------------------------------------------
    # XGBoost hyperparameter search (GridSearchCV) + feature importance plot
    # -------------------------------------------------------------------------
    xgb_search_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    random_state=args.random_state,
                    scale_pos_weight=spw,
                    objective="binary:logistic",
                ),
            ),
        ]
    )

    param_grid: dict[str, list[Any]] = {
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.01, 0.1],
        "clf__n_estimators": [100, 200],
        "clf__subsample": [0.8],
        "clf__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0],
    }

    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    print(
        f"\nXGBoost GridSearchCV: {n_combos} param combos × 5-fold CV "
        f"({n_combos * 5} fits) — scoring=f1 …"
    )
    grid = GridSearchCV(
        estimator=xgb_search_pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    train_f1_xgb_t = f1_score(y_train, best.predict(X_train), zero_division=0)
    cv_f1_xgb_t = float(grid.best_score_)
    test_f1_xgb_t = f1_score(y_test, best.predict(X_test), zero_division=0)
    test_auc_xgb_t = safe_roc_auc(y_test.values, best.predict_proba(X_test)[:, 1])

    clf_best = best.named_steps["clf"]
    importances = np.asarray(clf_best.feature_importances_, dtype=float)
    plot_xgb_feature_importance(
        list(X_train.columns),
        importances,
        args.feature_importance_out,
        top_k=min(25, len(feat_cols)),
    )

    print(f"\nSaved feature importance plot: {args.feature_importance_out.resolve()}")

    print("\n" + "=" * 76)
    print("XGBoost: Unoptimized vs GridSearchCV-tuned (same train/test split)")
    print("=" * 76)
    cmp_rows = [
        {
            "name": "XGBoost unoptimized",
            "train_f1": train_f1_xgb,
            "cv_f1": cv_f1_xgb,
            "test_f1": test_f1_xgb,
            "test_auc": test_auc_xgb,
        },
        {
            "name": "XGBoost tuned (best CV F1)",
            "train_f1": train_f1_xgb_t,
            "cv_f1": cv_f1_xgb_t,
            "test_f1": test_f1_xgb_t,
            "test_auc": test_auc_xgb_t,
        },
    ]
    print_results_table(cmp_rows)
    print("Best params:", grid.best_params_)

    rows.append(
        {
            "name": "XGBoost tuned (GridSearchCV)",
            "train_f1": train_f1_xgb_t,
            "cv_f1": cv_f1_xgb_t,
            "test_f1": test_f1_xgb_t,
            "test_auc": test_auc_xgb_t,
        }
    )

    print("\n=== Full comparison (baselines + tuned XGBoost) ===")
    print_results_table(rows)


if __name__ == "__main__":
    main()
