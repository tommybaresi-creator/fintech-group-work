"""
show_results.py — one-line model runner for the project notebook.

Usage in notebook
-----------------
    from utils.show_results import show_xgboost, show_logistic_reg, show_naive_bayes
    from utils.show_results import show_random_forest, show_mlp, show_classifier_chain
    from utils.show_results import show_soft_voting, show_hard_voting, show_all

    show_xgboost()          # runs XGBoost for both targets, prints all results, shows all plots
    show_logistic_reg()     # same for Logistic Regression
    show_all()              # runs every model sequentially

Each function:
    1. Runs the model script (trains, evaluates, pickles)
    2. Loads the pickled results
    3. Prints CV metrics (mean ± std) for F_E and F_B side by side
    4. Prints test set metrics at threshold=0.5 and at the PR-curve threshold
    5. Plots: PR curves, reliability diagrams (pre/post calibration), confusion matrix,
       SHAP summary (where available), learning curves (MLP), CV F1 box-plots
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import TARGETS, apply_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("show_results")

PICKLE_ROOT = Path(__file__).parent.parent / "data" / "pickled_files"

_MODEL_MAP = {
    "xgboost":          ("utils.xgboost_shap",    "xgboost_shap"),
    "logistic_reg":     ("utils.linear_reg",       "linear_reg"),
    "naive_bayes":      ("utils.naive_bayes",       "naive_bayes"),
    "random_forest":    ("utils.rand_forest",       "rand_forest"),
    "mlp":              ("utils.mlp",               "mlp"),
    "classifier_chain": ("utils.classifier_chain",  "classifier_chain"),
    "soft_voting":      ("utils.soft_voting_ens",   "soft_voting_ens"),
    "hard_voting":      ("utils.hard_voting_ens",   "hard_voting_ens"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_model(module_path: str) -> None:
    """Import and run a model script's main() function."""
    mod = importlib.import_module(module_path)
    logger.info("Running %s...", module_path)
    mod.main()


def _load_result(folder: str, target: str) -> Optional[dict]:
    """Load a pickled result dict. Returns None if not found."""
    path = PICKLE_ROOT / folder / f"{target.lower()}.joblib"
    if not path.exists():
        logger.warning("No pickle found at %s", path)
        return None
    return joblib.load(path)


def _load_chain_result() -> Optional[dict]:
    path = PICKLE_ROOT / "classifier_chain" / "both_targets.joblib"
    if not path.exists():
        logger.warning("No pickle found at %s", path)
        return None
    return joblib.load(path)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def _print_metrics_table(result: dict) -> None:
    """Print CV and test metrics, with ablation comparison if available."""
    model_name  = result["model_name"]
    target_name = result["target_name"]
    cv_sum      = result["cv_metrics_summary"]
    test_m      = result["test_metrics"]
    ablation    = result.get("ablation")

    print(f"\n{'='*64}")
    print(f"  {model_name}  |  {target_name}")
    print(f"{'='*64}")

    # CV metrics
    print(f"\n  {'Metric':<12} {'CV Mean±Std':>16}  {'Test@0.5':>10}")
    print(f"  {'-'*42}")
    for m in ["accuracy", "precision", "recall", "f1"]:
        cv_str   = _fmt(cv_sum[m]["mean"], cv_sum[m]["std"])
        test_str = f"{test_m[m]:.3f}"
        print(f"  {m:<12} {cv_str:>16}  {test_str:>10}")

    # PR-curve threshold metrics
    thr_info = result.get("threshold_info")
    if thr_info:
        print(f"\n  PR-curve threshold = {thr_info['threshold']:.3f}  "
              f"(Precision≥{0.75:.2f} constraint)")
        print(f"  {'':12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'At threshold':<12} {thr_info['precision']:>10.3f} "
              f"{thr_info['recall']:>10.3f} {thr_info['f1']:>10.3f}")

    # Brier score
    brier_pre  = result.get("brier_score_pre_cal")
    brier_post = result.get("brier_score")
    if brier_post is not None:
        if brier_pre is not None:
            print(f"\n  Brier score: {brier_pre:.4f} (pre-cal) → {brier_post:.4f} (post-cal)  "
                  f"[baseline 0.25]")
        else:
            print(f"\n  Brier score: {brier_post:.4f}  [baseline 0.25]")

    # Ablation comparison
    if ablation:
        eng  = ablation.get("engineered", {})
        base = ablation.get("baseline", {})
        if eng and base:
            eng_f1  = eng["cv_metrics_summary"]["f1"]
            base_f1 = base["cv_metrics_summary"]["f1"]
            delta   = eng_f1["mean"] - base_f1["mean"]
            print(f"\n  Ablation — F1 CV mean:")
            print(f"    F_E: {_fmt(eng_f1['mean'],  eng_f1['std'])}")
            print(f"    F_B: {_fmt(base_f1['mean'], base_f1['std'])}")
            print(f"    ΔF1 (F_E − F_B) = {delta:+.3f}")


def _plot_pr_curve(result: dict) -> None:
    thr_info = result.get("threshold_info")
    if thr_info is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr_info["recalls"], thr_info["precisions"], lw=1.5, label="PR curve")
    ax.axhline(0.75, color="red", linestyle="--", lw=1, label="Precision floor (0.75)")
    ax.scatter(
        thr_info["recall"], thr_info["precision"],
        s=80, zorder=5, color="red", label=f"Selected (thr={thr_info['threshold']:.3f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — {result['model_name']}  |  {result['target_name']}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_calibration(result: dict) -> None:
    """Reliability diagram. Requires y_test_true and predict_proba — skips if unavailable."""
    model     = result.get("model")
    y_true    = result.get("y_test_true")
    scaler    = result.get("scaler")
    feat_names = result.get("feature_names")

    # We only have y_test_true and y_test_pred in the pickle; full proba not stored.
    # Plot a simplified reliability check using stored brier scores as text annotation.
    brier = result.get("brier_score")
    brier_pre = result.get("brier_score_pre_cal")
    if brier is None:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.5, 0.6, f"Post-calibration Brier: {brier:.4f}", ha="center", fontsize=13)
    if brier_pre:
        ax.text(0.5, 0.4, f"Pre-calibration Brier:  {brier_pre:.4f}", ha="center", fontsize=13)
    ax.text(0.5, 0.2, "Baseline (no-skill): 0.2500", ha="center", fontsize=11, color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"Calibration — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout()
    plt.show()


def _plot_confusion_matrix(result: dict) -> None:
    y_true = result.get("y_test_true")
    y_pred = result.get("y_test_pred")
    if y_true is None or y_pred is None:
        return

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout()
    plt.show()


def _plot_cv_f1_boxplot(results_by_target: dict) -> None:
    """Box-plot of per-fold F1 scores across targets."""
    fig, axes = plt.subplots(1, len(results_by_target), figsize=(5 * len(results_by_target), 4))
    if len(results_by_target) == 1:
        axes = [axes]

    for ax, (target, result) in zip(axes, results_by_target.items()):
        if result is None:
            continue
        f1_folds = result["cv_metrics_raw"]["f1"]
        ax.boxplot(f1_folds, widths=0.5)
        ax.set_title(f"{target}")
        ax.set_ylabel("F1 (per fold)")
        ax.set_xticks([1]); ax.set_xticklabels([result["model_name"]])
        ax.axhline(np.mean(f1_folds), color="red", linestyle="--", lw=1, label=f"mean={np.mean(f1_folds):.3f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CV F1 distribution (10 folds)", fontsize=11)
    plt.tight_layout()
    plt.show()


def _plot_shap(result: dict) -> None:
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plot")
        return

    shap_values = result.get("shap_values")
    shap_test_X = result.get("shap_test_X")
    if shap_values is None or shap_test_X is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, shap_test_X, show=False)
    ax.set_title(f"SHAP summary — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout()
    plt.show()


def _plot_ablation_bar(result: dict) -> None:
    """Bar chart comparing F_E vs F_B CV F1."""
    ablation = result.get("ablation")
    if not ablation:
        return
    eng  = ablation.get("engineered")
    base = ablation.get("baseline")
    if not eng or not base:
        return

    labels = ["F_E (engineered)", "F_B (baseline)"]
    means  = [eng["cv_metrics_summary"]["f1"]["mean"],  base["cv_metrics_summary"]["f1"]["mean"]]
    stds   = [eng["cv_metrics_summary"]["f1"]["std"],   base["cv_metrics_summary"]["f1"]["std"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452"], width=0.4)
    ax.set_ylabel("CV F1 (mean ± std)")
    ax.set_title(f"Feature set ablation — {result['model_name']}  |  {result['target_name']}")
    ax.set_ylim(0, 1)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def _show_single_result(result: dict) -> None:
    """Print metrics and show all plots for a single result dict."""
    _print_metrics_table(result)
    _plot_pr_curve(result)
    _plot_calibration(result)
    _plot_confusion_matrix(result)
    _plot_cv_f1_boxplot({result["target_name"]: result})
    _plot_ablation_bar(result)
    _plot_shap(result)


# ---------------------------------------------------------------------------
# Public API — one function per model
# ---------------------------------------------------------------------------


def _show_model(key: str, retrain: bool = True) -> None:
    module_path, folder = _MODEL_MAP[key]

    if retrain:
        _run_model(module_path)

    if key == "classifier_chain":
        result = _load_chain_result()
        if result is None:
            return
        # Chain result has per-target sub-dicts
        for target in ["AccumulationInvestment", "IncomeInvestment"]:
            sub = {
                "model_name":         result["model_name"],
                "target_name":        target,
                "cv_metrics_raw":     result["cv_metrics_raw"][target],
                "cv_metrics_summary": result["cv_metrics_summary"][target],
                "test_metrics":       result["test_metrics"][target],
                "y_test_true":        result["y_test_true"][:, ["AccumulationInvestment", "IncomeInvestment"].index(target)],
                "y_test_pred":        result["y_test_pred"][:, ["AccumulationInvestment", "IncomeInvestment"].index(target)],
                "feature_names":      result["feature_names"],
                "ablation":           None,
                "brier_score":        None,
                "threshold_info":     None,
            }
            _show_single_result(sub)
    else:
        results_by_target = {}
        for target in TARGETS:
            result = _load_result(folder, target)
            if result is not None:
                results_by_target[target] = result

        for target, result in results_by_target.items():
            _show_single_result(result)

        if len(results_by_target) > 1:
            _plot_cv_f1_boxplot(results_by_target)


def show_xgboost(retrain: bool = True) -> None:
    """Run XGBoost for both targets and show all results and plots."""
    _show_model("xgboost", retrain)


def show_logistic_reg(retrain: bool = True) -> None:
    """Run Logistic Regression for both targets and show all results and plots."""
    _show_model("logistic_reg", retrain)


def show_naive_bayes(retrain: bool = True) -> None:
    """Run Gaussian Naive Bayes for both targets and show all results and plots."""
    _show_model("naive_bayes", retrain)


def show_random_forest(retrain: bool = True) -> None:
    """Run Random Forest for both targets and show all results and plots."""
    _show_model("random_forest", retrain)


def show_mlp(retrain: bool = True) -> None:
    """Run MLP for both targets and show all results and plots."""
    _show_model("mlp", retrain)


def show_classifier_chain(retrain: bool = True) -> None:
    """Run Classifier Chain and show all results and plots."""
    _show_model("classifier_chain", retrain)


def show_soft_voting(retrain: bool = True) -> None:
    """Run Soft Voting Ensemble for both targets and show all results and plots."""
    _show_model("soft_voting", retrain)


def show_hard_voting(retrain: bool = True) -> None:
    """Run Hard Voting Ensemble for both targets and show all results and plots."""
    _show_model("hard_voting", retrain)


def show_all(retrain: bool = True) -> None:
    """Run all models sequentially and show all results."""
    for key in _MODEL_MAP:
        logger.info("\n%s\n  Running: %s\n%s", "="*60, key, "="*60)
        _show_model(key, retrain)