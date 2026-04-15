"""
Static mid-term report charts (F1 progression, AUC comparison, ECE).

Writes PNGs under ``outputs/charts/``. Values match the project's reported benchmarks
(22-feature router, test split); edit constants below if you re-run training and metrics change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette: baselines (blue), tuned router (orange), calibrated (green)
COLOR_BASELINE = "#2E86AB"  # blue
COLOR_TUNED = "#E67E22"  # orange
COLOR_CALIBRATED = "#27AE60"  # green


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_f1_progression(out_path: Path, *, dpi: int) -> None:
    labels = [
        "Previous Baseline\n(6 Linguistic feats + LR)",
        "Our LR Baseline\n(22 Multi-modal feats)",
        "Our Tuned Router\n(22 Multi-modal feats + XGB)",
    ]
    values = [0.435, 0.6967, 0.7132]
    colors = [COLOR_BASELINE, COLOR_BASELINE, COLOR_TUNED]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="0.25", linewidth=0.8, width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Test F1 (positive class = model wrong)", fontsize=13)
    ax.set_title("F1 progression: feature engineering + tuned router", fontsize=15, fontweight="semibold", pad=12)
    ax.set_ylim(0, 0.95)
    ax.tick_params(axis="y", labelsize=12)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
        )

    # Relative gain from previous baseline to our LR baseline: (0.6967 - 0.435) / 0.435
    rel_gain = (values[1] - values[0]) / values[0]
    ax.annotate(
        f"≈ +{rel_gain * 100:.0f}% F1 vs.\nprevious baseline\n(new model-aware + multi-modal feats)",
        xy=(0.5, 0.62),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", edgecolor="0.5", alpha=0.95),
    )

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_auc_comparison(out_path: Path, *, dpi: int) -> None:
    labels = [
        "Always-RAG (Heuristic)",
        "Logistic Regression",
        "Random Forest",
        "XGBoost Tuned",
    ]
    values = [0.50, 0.8081, 0.7955, 0.8219]
    colors = [COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_TUNED]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="0.25", linewidth=0.7, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Test ROC-AUC", fontsize=13)
    ax.set_title("Test AUC comparison (router scores)", fontsize=15, fontweight="semibold", pad=12)
    ax.set_xlim(0.45, 0.92)
    ax.tick_params(axis="x", labelsize=12)
    ax.axvline(0.5, color="0.6", linestyle="--", linewidth=1, zorder=0)

    for bar, v in zip(bars, values):
        ax.text(
            v + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="medium",
        )

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ece_calibration(out_path: Path, *, dpi: int) -> None:
    labels = [
        "Vanilla Qwen Confidence",
        "XGBoost (Uncalibrated)",
        "XGBoost + Platt (Calibrated)",
    ]
    values = [0.3222, 0.1012, 0.0638]
    colors = [COLOR_BASELINE, COLOR_TUNED, COLOR_CALIBRATED]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="0.25", linewidth=0.8, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Expected calibration error (ECE) — lower is better", fontsize=13)
    ax.set_title("Calibration: post-hoc Platt scaling reduces ECE", fontsize=15, fontweight="semibold", pad=12)
    ax.set_ylim(0, max(values) * 1.35)
    ax.tick_params(axis="y", labelsize=12)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
        )

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mid-term report static charts")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/charts"),
        help="Output directory for PNG files",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out = args.out_dir
    dpi = args.dpi

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "axes.titlesize": 15,
            "axes.labelsize": 13,
        }
    )

    plot_f1_progression(out / "F1_Progression.png", dpi=dpi)
    plot_auc_comparison(out / "AUC_Comparison.png", dpi=dpi)
    plot_ece_calibration(out / "ECE_Calibration_Error.png", dpi=dpi)

    print(f"Wrote charts under {out.resolve()}:")
    print("  F1_Progression.png")
    print("  AUC_Comparison.png")
    print("  ECE_Calibration_Error.png")


if __name__ == "__main__":
    main()
