"""
Baseline router comparison for the R2R Router (mid-term report).

Loads ``data/router_training_matrix.pkl`` (from ``prepare_dataset.py``), uses all ``feat_*``
columns to predict ``target_label`` (1 = model answer wrong → prefer RAG).

Reports Train F1, 5-fold stratified CV F1 (with tqdm), Test F1, and Test ROC-AUC for:
  - Always-RAG (always predict 1)
  - Logistic Regression (class_weight='balanced')
  - Random Forest (class_weight='balanced')
  - XGBoost tuned via GridSearchCV on the full feature matrix (scale_pos_weight from y_train)

The best XGBoost pipeline from GridSearchCV is used for ``outputs/feature_importance.png``
(all features, ranked; default top 18 bars when 18 features exist).

A calibration section plots reliability diagrams (``sklearn.calibration.calibration_curve``) for a
vanilla Qwen score (``1 - chosen_token_prob`` as P(wrong)) and the tuned XGBoost router, fits
Platt scaling via ``CalibratedClassifierCV(..., method='sigmoid')``, and reports test F1 and ECE;
``outputs/calibration_plot.png`` stores the combined figure.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
    top_k: int | None = 18,
    title: str = "XGBoost (GridSearchCV-tuned) — feature importance",
) -> None:
    """Save a horizontal bar chart of the highest XGBoost feature importances."""
    if plt is None:
        raise RuntimeError("matplotlib is required for feature importance plots: pip install matplotlib")

    n = len(feature_names)
    k = n if top_k is None else min(top_k, n)
    order = np.argsort(importances)[::-1][:k]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
    ax.barh(range(len(names)), vals[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance (XGBoost default)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error (ECE): weighted mean |accuracy − confidence| over probability bins.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        w = float(mask.mean())
        if w == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += w * abs(acc - conf)
    return float(ece)


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Binary F1 from continuous scores at a decision threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, pred, zero_division=0))


def plot_calibration_reliability(
    y_true: np.ndarray,
    series: dict[str, np.ndarray],
    out_path: Path,
    *,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram comparing predicted vs. empirical positive rate per bin."""
    if plt is None:
        raise RuntimeError("matplotlib is required for calibration plots: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k:", linewidth=1.5, label="Perfect calibration")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, (name, y_prob) in enumerate(series.items()):
        probs = np.asarray(y_prob, dtype=float).ravel()
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        prob_true, prob_pred = calibration_curve(
            y_true,
            probs,
            n_bins=n_bins,
            strategy="uniform",
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1.5,
            label=name,
            color=colors[i % len(colors)],
        )
    ax.set_xlabel("Mean predicted probability (positive class = model wrong)")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Reliability diagram — router calibration (test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results_table(rows: list[dict[str, Any]]) -> None:
    w = 42
    header = f"{'Strategy/Model':<{w}} {'Train F1':>10} {'CV F1':>10} {'Test F1':>10} {'Test AUC':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        name = str(r["name"])[: w - 1]

        def fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'N/A':>10}"
            if isinstance(v, (float, np.floating)):
                return f"{float(v):10.4f}"
            return f"{v!s:>10}"

        print(
            f"{name:<{w}} {fmt(r['train_f1'])} {fmt(r['cv_f1'])} {fmt(r['test_f1'])} {fmt(r['test_auc'])}"
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
    parser.add_argument(
        "--importance-top-k",
        type=int,
        default=18,
        help="Number of features to show in the importance plot (default: 18)",
    )
    parser.add_argument(
        "--print-importance-ranking",
        action="store_true",
        help="Print the full XGBoost feature-importance ranking (all features, descending)",
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=Path("outputs/calibration_plot.png"),
        help="Reliability diagram output path",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for calibration_curve and ECE",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=3,
        help="Folds inside CalibratedClassifierCV (Platt / sigmoid)",
    )
    args = parser.parse_args()

    df = load_training_matrix(args.data)
    if "target_label" not in df.columns:
        raise RuntimeError("Column 'target_label' missing. Run prepare_dataset.py first.")

    feat_cols = feat_columns(df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["target_label"].astype(int)

    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]

    print(f"Loaded: {args.data} | samples={len(df)} | features={len(feat_cols)}")
    print(
        f"Train={len(X_train)} Test={len(X_test)} | "
        f"pos_rate train={y_train.mean():.4f} test={y_test.mean():.4f}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

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

    rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Heuristic baseline: Always-RAG (always predict positive class 1)
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
    # Logistic Regression (class_weight balanced)
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
    # Random Forest (class_weight balanced)
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
    # XGBoost: GridSearchCV on the full feature set (scale_pos_weight from y_train)
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
        f"({n_combos * 5} fits), scoring=f1 …"
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

    rows.append(
        {
            "name": "XGBoost tuned (GridSearchCV)",
            "train_f1": train_f1_xgb_t,
            "cv_f1": cv_f1_xgb_t,
            "test_f1": test_f1_xgb_t,
            "test_auc": test_auc_xgb_t,
        }
    )

    # -------------------------------------------------------------------------
    # Calibration analysis: reliability diagram, ECE, Platt scaling (sigmoid)
    # -------------------------------------------------------------------------
    y_test_arr = y_test.values.astype(int)
    prob_xgb = best.predict_proba(X_test)[:, 1]

    calibration_series: dict[str, np.ndarray] = {}

    if "chosen_token_prob" in df.columns:
        cp_test = df.iloc[idx_test]["chosen_token_prob"].astype(float).values
        prob_vanilla = np.clip(1.0 - cp_test, 1e-7, 1.0 - 1e-7)
        calibration_series["Vanilla Qwen (1 − p_first token)"] = prob_vanilla
    else:
        print(
            "\n[Calibration] Column 'chosen_token_prob' not found in training matrix; "
            "skipping vanilla Qwen curve (re-run prepare_dataset with inference columns)."
        )

    calibration_series["XGBoost tuned (GridSearchCV)"] = prob_xgb

    xgb_platt = CalibratedClassifierCV(
        estimator=clone(best),
        method="sigmoid",
        cv=args.calibration_cv,
    )
    xgb_platt.fit(X_train, y_train)
    prob_platt = xgb_platt.predict_proba(X_test)[:, 1]
    calibration_series["XGBoost + Platt (CalibratedClassifierCV)"] = prob_platt

    print("\n=== Calibration analysis (test set, positive class = model wrong) ===")
    for name, p in calibration_series.items():
        f1_t = f1_at_threshold(y_test_arr, p)
        ece = expected_calibration_error(y_test_arr, p, n_bins=args.calibration_bins)
        print(f"  {name}:  F1 = {f1_t:.4f}  |  ECE = {ece:.4f}")

    plot_calibration_reliability(
        y_test_arr,
        calibration_series,
        args.calibration_out,
        n_bins=args.calibration_bins,
    )
    print(f"\nSaved calibration plot: {args.calibration_out.resolve()}")

    clf_best = best.named_steps["clf"]
    importances = np.asarray(clf_best.feature_importances_, dtype=float)
    n_feat = len(feat_cols)
    k_plot = None if args.importance_top_k <= 0 else min(args.importance_top_k, n_feat)

    # Optionally print the full importance ranking (useful when n_feat > plot top_k).
    if args.print_importance_ranking:
        order_all = np.argsort(importances)[::-1]
        print("\n=== XGBoost feature importance ranking (descending) ===")
        for rank, idx in enumerate(order_all, start=1):
            print(f"{rank:>2}. {X_train.columns[idx]}: {importances[idx]:.6f}")

    plot_xgb_feature_importance(
        list(X_train.columns),
        importances,
        args.feature_importance_out,
        top_k=k_plot,
        title=(
            f"XGBoost (GridSearchCV-tuned) — all {n_feat} features by importance"
            if k_plot is None or k_plot == n_feat
            else f"XGBoost (GridSearchCV-tuned) — top {k_plot} of {n_feat} features"
        ),
    )

    print("\n=== Router baseline comparison (mid-term table) ===")
    print_results_table(rows)
    print("XGBoost best params (GridSearchCV):", grid.best_params_)
    print(f"\nSaved feature importance plot: {args.feature_importance_out.resolve()}")


if __name__ == "__main__":
    main()
