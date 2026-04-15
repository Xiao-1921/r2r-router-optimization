"""
Generate mid-term report figures from router training matrices.

Reads ``data/router_training_matrix.pkl`` (22 features) and optionally
``data/router_training_matrix_semantic.pkl`` (38 features), reproduces the same
train/test split and models as ``trainer.py``, and writes assets under ``outputs/report/``.

Requires: matplotlib, seaborn, scikit-learn, xgboost, pandas, numpy.

Benchmark F1 uses the same **binary** F1 as ``trainer.py`` (positive class = model wrong).
If you see ~0.59 for Logistic Regression elsewhere, it is often **macro** F1 — see script output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import seaborn as sns

from trainer import (
    expected_calibration_error,
    feat_columns,
    load_training_matrix,
    safe_roc_auc,
    xgb_scale_pos_weight,
)

try:
    from xgboost import XGBClassifier
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pip install xgboost") from e

# Human-readable names for report figures (22-feature set + common extras).
READABLE_FEAT_NAMES: dict[str, str] = {
    "feat_top2_margin": "Internal uncertainty (top-2 margin)",
    "feat_prob_margin": "Probability margin (1 − first-token prob.)",
    "feat_chosen_token_prob": "First-token probability",
    "feat_first_token_entropy": "First-token entropy",
    "feat_avg_entropy": "Average token entropy",
    "feat_perplexity": "Sequence perplexity",
    "feat_top5_entropy": "Top-5 first-token entropy",
    "feat_dist_std": "Top-5 logprob spread (std. dev.)",
    "feat_dist_kurtosis": "Top-5 logprob kurtosis",
    "feat_word_count": "Word count",
    "feat_char_count": "Character count",
    "feat_avg_word_len": "Average word length",
    "feat_punct_count": "Punctuation count",
    "feat_entity_count": "Named entity count",
    "feat_noun_ratio": "Noun ratio",
    "feat_verb_ratio": "Verb ratio",
    "feat_has_negation": "Has negation (flag)",
    "feat_negation_count": "Negation count",
    "feat_has_quant": "Has quantifier (flag)",
    "feat_quant_count": "Quantifier count",
    "feat_has_comparison": "Has comparison (flag)",
    "feat_comparison_count": "Comparison count",
}


def readable_feat(name: str) -> str:
    return READABLE_FEAT_NAMES.get(name, name.replace("feat_", "").replace("_", " ").title())


def _f1(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str,
) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if average == "binary":
        return float(f1_score(yt, yp, average="binary", pos_label=1, zero_division=0))
    return float(f1_score(yt, yp, average=average, zero_division=0))


def fit_lr_and_xgb(
    df: pd.DataFrame,
    *,
    random_state: int,
    test_size: float,
    calibration_cv: int,
    f1_average: str,
) -> dict[str, Any]:
    """Train LR, tuned XGBoost, and Platt calibration; return metrics and arrays for plots."""
    feat_cols = feat_columns(df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["target_label"].astype(int)

    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
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
                    random_state=random_state,
                ),
            ),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    pred_lr = lr_pipe.predict(X_test)
    test_f1_lr = _f1(y_test, pred_lr, average=f1_average)

    pred_always = np.ones(len(y_test), dtype=int)
    test_f1_always = _f1(y_test, pred_always, average=f1_average)

    spw = xgb_scale_pos_weight(y_train)
    xgb_search_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    random_state=random_state,
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
    grid = GridSearchCV(
        estimator=xgb_search_pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    pred_xgb = best.predict(X_test)
    test_f1_xgb = _f1(y_test, pred_xgb, average=f1_average)
    test_f1_xgb_binary = _f1(y_test, pred_xgb, average="binary")
    test_auc_xgb = safe_roc_auc(y_test.values, best.predict_proba(X_test)[:, 1])

    y_test_arr = y_test.values.astype(int)

    if "chosen_token_prob" not in df.columns:
        raise RuntimeError("Training matrix must include 'chosen_token_prob' for reliability plots.")
    cp_test = df.iloc[idx_test]["chosen_token_prob"].astype(float).values
    prob_base = np.clip(1.0 - cp_test, 1e-7, 1.0 - 1e-7)

    xgb_platt = CalibratedClassifierCV(
        estimator=clone(best),
        method="sigmoid",
        cv=calibration_cv,
    )
    xgb_platt.fit(X_train, y_train)
    prob_platt = xgb_platt.predict_proba(X_test)[:, 1]

    n_bins = 10
    ece_base = expected_calibration_error(y_test_arr, prob_base, n_bins=n_bins)
    ece_platt = expected_calibration_error(y_test_arr, prob_platt, n_bins=n_bins)

    clf = best.named_steps["clf"]
    importances = np.asarray(clf.feature_importances_, dtype=float)

    # Diagnostics: LR F1 under both averages (common confusion: ~0.59 is macro, ~0.70 is binary).
    lr_f1_binary = _f1(y_test, pred_lr, average="binary")
    lr_f1_macro = _f1(y_test, pred_lr, average="macro")

    return {
        "feat_cols": feat_cols,
        "test_f1_always": test_f1_always,
        "test_f1_lr": test_f1_lr,
        "test_f1_xgb": test_f1_xgb,
        "test_f1_xgb_binary": test_f1_xgb_binary,
        "lr_f1_binary": lr_f1_binary,
        "lr_f1_macro": lr_f1_macro,
        "test_auc_xgb": test_auc_xgb,
        "ece_base_llm": ece_base,
        "ece_platt": ece_platt,
        "y_test_arr": y_test_arr,
        "prob_base": prob_base,
        "prob_platt": prob_platt,
        "importances": importances,
        "X_train_columns": list(X_train.columns),
        "idx_test": idx_test,
    }


def plot_benchmark_f1(
    test_f1_always: float,
    test_f1_lr: float,
    test_f1_xgb: float,
    out_path: Path,
    *,
    dpi: int,
    f1_average: str,
) -> None:
    labels = ["Always-RAG", "Logistic Regression\n(Baseline)", "Tuned XGBoost\n(Ours)"]
    values = [test_f1_always, test_f1_lr, test_f1_xgb]
    colors = ["#95a5a6", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="0.2", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    if f1_average == "binary":
        ylabel = "Test F1 (binary, positive class = model wrong)"
        subtitle = "Same definition as trainer.py / sklearn default for binary classification."
    else:
        ylabel = "Test F1 (macro — mean of class 0 and class 1 F1)"
        subtitle = "Macro is lower than binary when classes are imbalanced; do not mix with mid-term table."
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Router benchmark (test set)", fontsize=13, fontweight="semibold")
    ax.text(
        0.5,
        -0.14,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.35",
    )
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
        )
    sns.despine(ax=ax, top=True, right=True)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_smoking_gun_boxplot(df: pd.DataFrame, out_path: Path, *, dpi: int) -> None:
    if "feat_top2_margin" not in df.columns:
        raise RuntimeError("Column 'feat_top2_margin' missing from matrix.")
    plot_df = df[["feat_top2_margin", "target_label"]].copy()
    plot_df = plot_df.rename(
        columns={
            "feat_top2_margin": "Top-2 margin",
            "target_label": "Outcome",
        }
    )
    plot_df["Outcome"] = plot_df["Outcome"].map({0: "Correct (label 0)", 1: "Wrong (label 1)"})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(
        data=plot_df,
        x="Outcome",
        y="Top-2 margin",
        order=["Correct (label 0)", "Wrong (label 1)"],
        palette=["#3498db", "#e74c3c"],
        ax=ax,
        width=0.5,
        linewidth=1.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("feat_top2_margin (internal uncertainty)", fontsize=12)
    ax.set_title(
        "Separation signal: top-2 margin when the base model is wrong vs. correct",
        fontsize=12,
        fontweight="semibold",
    )
    plt.setp(ax.get_xticklabels(), fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_top15(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    *,
    dpi: int,
    top_k: int = 15,
) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    names = [readable_feat(feature_names[i]) for i in order]
    vals = importances[order]
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.38 * top_k)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals[::-1], color="steelblue", edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Feature importance (XGBoost)", fontsize=12)
    ax.set_title(
        f"Top {top_k} features — 22-feature tuned router",
        fontsize=13,
        fontweight="semibold",
    )
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_base_vs_calibrated(
    y_true: np.ndarray,
    prob_base: np.ndarray,
    prob_platt: np.ndarray,
    ece_base: float,
    ece_platt: float,
    out_path: Path,
    *,
    dpi: int,
    n_bins: int = 10,
) -> None:
    y_true = np.asarray(y_true).astype(int).ravel()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", zorder=1)

    series = [
        (
            "Base LLM (1 − p_first)",
            np.clip(prob_base, 1e-7, 1 - 1e-7),
            ece_base,
            "#e67e22",
        ),
        (
            "Calibrated router (Platt)",
            np.clip(prob_platt, 1e-7, 1 - 1e-7),
            ece_platt,
            "#27ae60",
        ),
    ]
    for label, probs, ece, color in series:
        pt, pp = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        leg_label = f"{label}  |  ECE = {ece:.4f}"
        ax.plot(pp, pt, marker="o", linewidth=2.0, markersize=7, label=leg_label, color=color)

    ax.set_xlabel("Mean predicted P(model wrong)", fontsize=13)
    ax.set_ylabel("Observed fraction (model wrong)", fontsize=13)
    ax.set_title("Reliability: base LM vs. calibrated router (test set)", fontsize=14, fontweight="semibold")
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    sns.despine(ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_ablation_tables(
    rows: list[dict[str, Any]],
    out_md: Path,
    out_tex: Path,
) -> None:
    """rows: list of dicts with keys model, test_f1, test_auc, ece_platt"""
    lines_md = [
        "# Ablation: 22 vs. 38 features",
        "",
        "| Model | Test F1 | Test AUC | ECE (Platt) |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines_md.append(
            f"| {r['model']} | {r['test_f1']:.4f} | {r['test_auc']:.4f} | {r['ece_platt']:.4f} |"
        )
    lines_md.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines_md), encoding="utf-8")

    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation: 22-feature vs.\ 38-feature router (test set).}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & Test F1 & Test AUC & ECE (Platt) \\",
        r"\hline",
    ]
    for r in rows:
        model_tex = r["model"].replace("%", r"\%")
        tex.append(
            f"{model_tex} & {r['test_f1']:.4f} & {r['test_auc']:.4f} & {r['ece_platt']:.4f} \\\\"
        )
    tex.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mid-term report figures")
    parser.add_argument(
        "--matrix-22",
        type=Path,
        default=Path("data/router_training_matrix.pkl"),
        help="22-feature training matrix PKL",
    )
    parser.add_argument(
        "--matrix-38",
        type=Path,
        default=Path("data/router_training_matrix_semantic.pkl"),
        help="38-feature (semantic ablation) training matrix PKL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/report"),
        help="Output directory for figures and tables",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--f1-average",
        choices=("binary", "macro"),
        default="binary",
        help=(
            "F1 averaging for benchmark bars. Use 'binary' (default) to match trainer.py; "
            "'macro' is often ~0.59 for LR and is easy to confuse with binary ~0.70."
        ),
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12

    out = args.out_dir
    dpi = args.dpi

    df22 = load_training_matrix(args.matrix_22)
    res22 = fit_lr_and_xgb(
        df22,
        random_state=args.random_state,
        test_size=args.test_size,
        calibration_cv=args.calibration_cv,
        f1_average=args.f1_average,
    )
    print(
        "[Logistic Regression — test set] "
        f"F1 binary={res22['lr_f1_binary']:.4f}  |  "
        f"F1 macro={res22['lr_f1_macro']:.4f}\n"
        "  (trainer.py table uses binary F1; ~0.59 usually means macro was reported elsewhere.)"
    )

    plot_benchmark_f1(
        res22["test_f1_always"],
        res22["test_f1_lr"],
        res22["test_f1_xgb"],
        out / "benchmark_f1.png",
        dpi=dpi,
        f1_average=args.f1_average,
    )

    plot_smoking_gun_boxplot(df22, out / "smoking_gun_top2_margin_boxplot.png", dpi=dpi)

    plot_feature_importance_top15(
        res22["X_train_columns"],
        res22["importances"],
        out / "feature_importance_top15.png",
        dpi=dpi,
        top_k=15,
    )

    plot_reliability_base_vs_calibrated(
        res22["y_test_arr"],
        res22["prob_base"],
        res22["prob_platt"],
        res22["ece_base_llm"],
        res22["ece_platt"],
        out / "reliability_base_vs_calibrated.png",
        dpi=dpi,
    )

    df38 = load_training_matrix(args.matrix_38)
    res38 = fit_lr_and_xgb(
        df38,
        random_state=args.random_state,
        test_size=args.test_size,
        calibration_cv=args.calibration_cv,
        f1_average=args.f1_average,
    )

    ablation_rows = [
        {
            "model": "22 features (production)",
            "test_f1": res22["test_f1_xgb_binary"],
            "test_auc": res22["test_auc_xgb"],
            "ece_platt": res22["ece_platt"],
        },
        {
            "model": "38 features (semantic PCA ablation)",
            "test_f1": res38["test_f1_xgb_binary"],
            "test_auc": res38["test_auc_xgb"],
            "ece_platt": res38["ece_platt"],
        },
    ]
    write_ablation_tables(ablation_rows, out / "ablation_table.md", out / "ablation_table.tex")

    print(f"Wrote report assets under {out.resolve()}")
    print("  benchmark_f1.png")
    print("  smoking_gun_top2_margin_boxplot.png")
    print("  feature_importance_top15.png")
    print("  reliability_base_vs_calibrated.png")
    print("  ablation_table.md, ablation_table.tex")


if __name__ == "__main__":
    main()
