"""
show_results.py — one-line model runner for the project notebook.

Usage
-----
    from utils.show_results import (
        show_xgboost, show_logistic_reg, show_naive_bayes,
        show_random_forest, show_mlp, show_classifier_chain,
        show_soft_voting, show_hard_voting, show_all, show_winner
    )

    show_xgboost()      # trains, evaluates, prints full story, shows plots
    show_all()          # runs every model
    show_winner()       # loads all pickles, runs Wilcoxon tests, declares winner

Each function narrates the full pipeline:
    splitting → scaling → cross-validating → hyperparameter tuning →
    evaluating → feature set comparison → propensity distribution → plots
"""

import importlib
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import (
    BASELINE_FEATURE_NAMES,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    FEATURE_NAMES,
    PICKLE_ROOT,
    PRECISION_FLOOR,
    TARGETS,
    load_result,
    segment_by_confidence,
    summarise_cv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("show_results")

_MODEL_MAP = {
    "xgboost":          ("utils.xgboost_shap",   "xgboost_shap"),
    "logistic_reg":     ("utils.linear_reg",      "linear_reg"),
    "naive_bayes":      ("utils.naive_bayes",      "naive_bayes"),
    "random_forest":    ("utils.rand_forest",      "rand_forest"),
    "mlp":              ("utils.mlp",              "mlp"),
    "classifier_chain": ("utils.classifier_chain", "classifier_chain"),
    "soft_voting":      ("utils.soft_voting_ens",  "soft_voting_ens"),
    "hard_voting":      ("utils.hard_voting_ens",  "hard_voting_ens"),
}


# ---------------------------------------------------------------------------
# Narrative printing
# ---------------------------------------------------------------------------


def _separator(title: str = "", width: int = 64) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'='*pad} {title} {'='*pad}")
    else:
        print("=" * width)


def _step(msg: str) -> None:
    print(f"\n  ▸ {msg}")


def _result(msg: str) -> None:
    print(f"    → {msg}")


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def _print_full_story(result: dict) -> None:
    """Print the full pipeline narrative and all metrics for one result."""
    model_name  = result["model_name"]
    target_name = result["target_name"]
    cv_sum      = result["cv_metrics_summary"]
    test_m      = result["test_metrics"]
    ablation    = result.get("ablation")
    best_params = result.get("best_params")
    thr_info    = result.get("threshold_info")
    brier_pre   = result.get("brier_score_pre_cal")
    brier_post  = result.get("brier_score")
    y_proba     = result.get("y_test_proba")

    _separator(f"{model_name}  |  {target_name}")

    # ---- Pipeline narrative ----
    _step("Data split: 80% training / 20% held-out test set, stratified on target class ratio")

    feat_set = result.get("feature_names", [])
    if feat_set == FEATURE_NAMES:
        _step("Feature set: F_E — 10 engineered features (lifecycle + savings capacity)")
    elif feat_set == BASELINE_FEATURE_NAMES:
        _step("Feature set: F_B — 7 professor's baseline features")
    else:
        _step(f"Feature set: {len(feat_set)} features")

    scaler = result.get("scaler")
    if scaler is not None:
        scaler_name = type(scaler).__name__
        _step(f"Scaling: {scaler_name} fitted on training set only (test set transformation applied separately — no leakage)")
    else:
        _step("Scaling: none (tree-based model, invariant to monotonic transforms) or inside Pipeline components")

    n_folds = len(result["cv_metrics_raw"]["f1"])
    _step(f"Cross-validation: {n_folds}-fold StratifiedKFold (model cloned each fold, scaler refits per fold inside Pipeline)")

    if best_params:
        _step("Hyperparameter tuning: RandomizedSearchCV (inner 3-fold CV) — best configuration below")
        for k, v in best_params.items():
            _result(f"{k} = {v}")
    else:
        _step("Hyperparameter tuning: default parameters (no tuning for this model)")

    if brier_pre is not None:
        _step("Calibration: post-hoc isotonic regression (cv=5) applied to correct probability bias")
        _result(f"Brier score: {brier_pre:.4f} (pre) → {brier_post:.4f} (post)  [no-skill baseline: 0.25]")
        improvement_pct = (brier_pre - brier_post) / brier_pre * 100
        _result(f"Calibration improved Brier by {improvement_pct:.1f}%")
    elif brier_post is not None:
        _step("Calibration quality:")
        _result(f"Brier score: {brier_post:.4f}  [no-skill baseline: 0.25]")

    # ---- CV metrics ----
    print(f"\n  {'─'*58}")
    print(f"  {'Metric':<14} {'CV Mean ± Std':>18}  {'Test @ thr=0.5':>16}")
    print(f"  {'─'*58}")
    for m in ["accuracy", "precision", "recall", "f1"]:
        cv_str   = _fmt(cv_sum[m]["mean"], cv_sum[m]["std"])
        test_str = f"{test_m[m]:.3f}"
        marker   = "  ◄ PRIMARY" if m == "f1" else ""
        print(f"  {m:<14} {cv_str:>18}  {test_str:>16}{marker}")
    print(f"  {'─'*58}")

    # ---- PR-curve threshold ----
    if thr_info:
        print(f"\n  MiFID II threshold selection (Precision ≥ {PRECISION_FLOOR:.2f} constraint):")
        print(f"  Operating threshold: {thr_info['threshold']:.3f}  "
              f"(vs naive 0.5 — {'lower' if thr_info['threshold'] < 0.5 else 'higher'}, reflects class distribution)")
        print(f"  {'':14} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print(f"  {'At threshold':<14} {thr_info['precision']:>12.3f} "
              f"{thr_info['recall']:>10.3f} {thr_info['f1']:>10.3f}")

    # ---- Ablation ----
    if ablation:
        eng  = ablation.get("engineered", {})
        base = ablation.get("baseline", {})
        if eng and base:
            eng_f1  = eng["cv_metrics_summary"]["f1"]
            base_f1 = base["cv_metrics_summary"]["f1"]
            delta   = eng_f1["mean"] - base_f1["mean"]
            winner  = "F_E" if delta > 0 else "F_B" if delta < -0.002 else "EQUIVALENT"
            print(f"\n  Feature set ablation (CV F1):")
            print(f"    F_E (engineered):  {_fmt(eng_f1['mean'], eng_f1['std'])}")
            print(f"    F_B (baseline):    {_fmt(base_f1['mean'], base_f1['std'])}")
            print(f"    ΔF1 (F_E − F_B):  {delta:+.3f}  →  Winner: {winner}")
            if winner == "F_B":
                print(f"    Interpretation: tree model learns interactions natively — F_E adds noise")
            elif winner == "F_E":
                print(f"    Interpretation: engineered features provide lift — linear model benefits from explicit interactions")
            else:
                print(f"    Interpretation: feature sets are statistically equivalent for this model")

    # ---- Propensity scores ----
    if y_proba is not None:
        seg = segment_by_confidence(y_proba)
        n   = len(y_proba)
        print(f"\n  Propensity score distribution (test set, n={n}):")
        print(f"    mean={y_proba.mean():.3f}  std={y_proba.std():.3f}  "
              f"min={y_proba.min():.3f}  max={y_proba.max():.3f}")
        print(f"    High confidence  (>{CONFIDENCE_HIGH:.2f}):  "
              f"{seg['high']['count']:4d}  ({seg['high']['count']/n*100:.1f}%)  → automate recommendation")
        print(f"    Uncertain  ({CONFIDENCE_LOW:.2f}–{CONFIDENCE_HIGH:.2f}):  "
              f"{seg['unsure']['count']:4d}  ({seg['unsure']['count']/n*100:.1f}%)  → route to human advisor")
        print(f"    Low propensity   (<{CONFIDENCE_LOW:.2f}):  "
              f"{seg['low']['count']:4d}  ({seg['low']['count']/n*100:.1f}%)  → no action")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_pr_curve(result: dict) -> None:
    thr_info = result.get("threshold_info")
    if thr_info is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr_info["recalls"], thr_info["precisions"], lw=1.5, label="PR curve")
    ax.axhline(PRECISION_FLOOR, color="red", linestyle="--", lw=1,
               label=f"MiFID II precision floor ({PRECISION_FLOOR:.2f})")
    ax.scatter(thr_info["recall"], thr_info["precision"], s=80, zorder=5,
               color="red", label=f"Selected threshold={thr_info['threshold']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — {result['model_name']}  |  {result['target_name']}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_confusion_matrix(result: dict) -> None:
    y_true = result.get("y_test_true")
    y_pred = result.get("y_test_pred")
    if y_true is None or y_pred is None:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout(); plt.show()


def _plot_cv_f1_boxplot(results_by_target: dict) -> None:
    n = len(results_by_target)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (target, result) in zip(axes, results_by_target.items()):
        if result is None:
            continue
        f1_folds = result["cv_metrics_raw"]["f1"]
        ax.boxplot(f1_folds, widths=0.5)
        ax.axhline(np.mean(f1_folds), color="red", linestyle="--", lw=1,
                   label=f"mean={np.mean(f1_folds):.3f}")
        ax.set_title(target); ax.set_ylabel("F1 (per fold)")
        ax.set_xticks([1]); ax.set_xticklabels([result["model_name"]], fontsize=8)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("CV F1 distribution (10 folds)", fontsize=11)
    plt.tight_layout(); plt.show()


def _plot_ablation_bar(result: dict) -> None:
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
    colors = ["#4C72B0" if means[0] >= means[1] else "#DD8452", "#DD8452" if means[0] >= means[1] else "#4C72B0"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, width=0.4)
    ax.set_ylabel("CV F1 (mean ± std)")
    ax.set_title(f"Feature set ablation — {result['model_name']}  |  {result['target_name']}")
    ax.set_ylim(max(0, min(means) - 0.05), min(1, max(means) + 0.08))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + stds[0] + 0.01,
                f"{mean:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_propensity_histogram(result: dict) -> None:
    y_proba = result.get("y_test_proba")
    if y_proba is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_proba, bins=30, edgecolor="white", color="#4C72B0", alpha=0.8)
    ax.axvline(CONFIDENCE_HIGH, color="green",  linestyle="--", lw=1.5, label=f"Automate (>{CONFIDENCE_HIGH:.2f})")
    ax.axvline(CONFIDENCE_LOW,  color="orange", linestyle="--", lw=1.5, label=f"Low (<{CONFIDENCE_LOW:.2f})")
    ax.set_xlabel("Propensity score"); ax.set_ylabel("Count")
    ax.set_title(f"Propensity distribution — {result['model_name']}  |  {result['target_name']}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_shap(result: dict) -> None:
    try:
        import shap
    except ImportError:
        return
    shap_values = result.get("shap_values")
    shap_test_X = result.get("shap_test_X")
    if shap_values is None or shap_test_X is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, shap_test_X, show=False)
    ax.set_title(f"SHAP summary — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout(); plt.show()


def _show_all_plots(result: dict) -> None:
    _plot_pr_curve(result)
    _plot_confusion_matrix(result)
    _plot_propensity_histogram(result)
    _plot_ablation_bar(result)
    _plot_shap(result)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def _run_and_load(key: str, retrain: bool) -> dict:
    module_path, folder = _MODEL_MAP[key]

    if retrain:
        mod = importlib.import_module(module_path)
        logger.info("Training %s...", module_path)
        mod.main()

    if key == "classifier_chain":
        path = PICKLE_ROOT / folder / "both_targets.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No pickle at {path}")
        with open(path, "rb") as f:
            return {"_chain": pickle.load(f)}

    results = {}
    for target in TARGETS:
        try:
            results[target] = load_result(folder, target)
        except FileNotFoundError:
            logger.warning("No pickle for %s / %s", folder, target)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _show_model(key: str, retrain: bool = True) -> None:
    data = _run_and_load(key, retrain)

    if "_chain" in data:
        chain_result = data["_chain"]
        for i, target in enumerate(["AccumulationInvestment", "IncomeInvestment"]):
            sub = {
                "model_name":         chain_result["model_name"],
                "target_name":        target,
                "cv_metrics_raw":     chain_result["cv_metrics_raw"][target],
                "cv_metrics_summary": chain_result["cv_metrics_summary"][target],
                "test_metrics":       chain_result["test_metrics"][target],
                "y_test_true":        chain_result["y_test_true"][:, i],
                "y_test_pred":        chain_result["y_test_pred"][:, i],
                "y_test_proba":       chain_result.get("y_test_proba", {}).get(target),
                "feature_names":      chain_result["feature_names"],
                "scaler":             chain_result.get("scaler"),
                "ablation":           None,
                "best_params":        chain_result.get("best_params"),
                "threshold_info":     None,
                "brier_score":        None,
            }
            _print_full_story(sub)
            _show_all_plots(sub)
        _plot_cv_f1_boxplot({
            t: {
                "cv_metrics_raw": chain_result["cv_metrics_raw"][t],
                "model_name": chain_result["model_name"],
            }
            for t in ["AccumulationInvestment", "IncomeInvestment"]
        })
        return

    for target, result in data.items():
        _print_full_story(result)
        _show_all_plots(result)

    if len(data) > 1:
        _plot_cv_f1_boxplot(data)


def show_xgboost(retrain: bool = True) -> None:
    """Run XGBoost for both targets and show full pipeline story + plots."""
    _show_model("xgboost", retrain)


def show_logistic_reg(retrain: bool = True) -> None:
    """Run Logistic Regression for both targets and show full pipeline story + plots."""
    _show_model("logistic_reg", retrain)


def show_naive_bayes(retrain: bool = True) -> None:
    """Run Gaussian Naive Bayes for both targets and show full pipeline story + plots."""
    _show_model("naive_bayes", retrain)


def show_random_forest(retrain: bool = True) -> None:
    """Run Random Forest for both targets and show full pipeline story + plots."""
    _show_model("random_forest", retrain)


def show_mlp(retrain: bool = True) -> None:
    """Run MLP for both targets and show full pipeline story + plots."""
    _show_model("mlp", retrain)


def show_classifier_chain(retrain: bool = True) -> None:
    """Run Classifier Chain and show full pipeline story + plots."""
    _show_model("classifier_chain", retrain)


def show_soft_voting(retrain: bool = True) -> None:
    """Run Soft Voting Ensemble for both targets and show full pipeline story + plots."""
    _show_model("soft_voting", retrain)


def show_hard_voting(retrain: bool = True) -> None:
    """Run Hard Voting Ensemble for both targets and show full pipeline story + plots."""
    _show_model("hard_voting", retrain)


def show_all(retrain: bool = True) -> None:
    """Run all models sequentially and show all results."""
    for key in _MODEL_MAP:
        _separator(f"MODEL: {key.upper()}", width=64)
        _show_model(key, retrain)


def show_winner() -> None:
    """
    Load all pickles, run pairwise Wilcoxon tests, declare winner per target.

    This is the formal model selection procedure from Section 3.5 of the paper.
    Primary metric: CV F1. Hard constraint: Test Precision >= 0.75.
    Tie-breaking: Occam's razor (simpler model wins if p > 0.05).
    """
    _separator("MODEL SELECTION — WILCOXON SIGNED-RANK TESTS", width=64)

    folders = {
        "XGBoost":       "xgboost_shap",
        "LR":            "linear_reg",
        "NaiveBayes":    "naive_bayes",
        "RandomForest":  "rand_forest",
        "MLP":           "mlp",
        "SoftVoting":    "soft_voting_ens",
        "HardVoting":    "hard_voting_ens",
    }

    for target in TARGETS:
        _separator(target)
        results = {}
        for name, folder in folders.items():
            try:
                r = load_result(folder, target)
                f1_folds = r["cv_metrics_raw"]["f1"]
                test_prec = r["test_metrics"]["precision"]
                results[name] = {
                    "f1_folds":   f1_folds,
                    "cv_f1_mean": np.mean(f1_folds),
                    "cv_f1_std":  np.std(f1_folds),
                    "test_precision": test_prec,
                    "test_f1":    r["test_metrics"]["f1"],
                    "passes_constraint": test_prec >= PRECISION_FLOOR,
                }
            except FileNotFoundError:
                logger.warning("  Skipping %s — no pickle found", name)

        if not results:
            print("  No results found — run models first.")
            continue

        # Filter to models passing the precision constraint
        passing = {k: v for k, v in results.items() if v["passes_constraint"]}
        failing = {k: v for k, v in results.items() if not v["passes_constraint"]}

        print(f"\n  {'Model':<20} {'CV F1 mean±std':>18}  {'Test Prec':>10}  {'Test F1':>8}  {'MiFID OK':>9}")
        print(f"  {'─'*70}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["cv_f1_mean"]):
            ok = "✓" if r["passes_constraint"] else "✗"
            print(f"  {name:<20} {_fmt(r['cv_f1_mean'], r['cv_f1_std']):>18}  "
                  f"{r['test_precision']:>10.3f}  {r['test_f1']:>8.3f}  {ok:>9}")

        if failing:
            print(f"\n  ✗ Excluded (Precision < {PRECISION_FLOOR:.2f}): {', '.join(failing.keys())}")

        if not passing:
            print("\n  No model passes the MiFID II precision constraint on test set.")
            # Fall back to all models
            passing = results

        # Rank passing models by CV F1
        ranked = sorted(passing.items(), key=lambda x: -x[1]["cv_f1_mean"])
        best_name, best_data = ranked[0]
        second_name, second_data = ranked[1] if len(ranked) > 1 else (None, None)

        print(f"\n  Top candidate: {best_name}  CV F1 = {best_data['cv_f1_mean']:.3f}")

        if second_name is not None:
            # Wilcoxon test between top 2
            try:
                stat, p_val = wilcoxon(best_data["f1_folds"], second_data["f1_folds"])
                print(f"  Wilcoxon test vs {second_name}: statistic={stat:.3f}  p={p_val:.4f}")
                if p_val > 0.05:
                    print(f"  p > 0.05 → not significant. Applying Occam's razor: selecting simpler model.")
                    # Simple heuristic: fewer parameters = simpler
                    complexity = {"NaiveBayes": 1, "LR": 2, "XGBoost": 3, "RandomForest": 3,
                                  "MLP": 4, "SoftVoting": 5, "HardVoting": 5, "ClassifierChain": 4}
                    winner = min([best_name, second_name], key=lambda x: complexity.get(x, 99))
                    print(f"  ★ WINNER for {target}: {winner}  (simpler of equivalent models)")
                else:
                    print(f"  p ≤ 0.05 → significant. Best model wins.")
                    print(f"  ★ WINNER for {target}: {best_name}")
            except Exception as e:
                print(f"  Wilcoxon test failed: {e}")
                print(f"  ★ WINNER for {target}: {best_name}  (by CV F1)")
        else:
            print(f"  ★ WINNER for {target}: {best_name}")

        # Show propensity score stats for the winner
        try:
            winner_result = load_result(folders.get(best_name, ""), target)
            y_proba = winner_result.get("y_test_proba")
            if y_proba is not None:
                seg = segment_by_confidence(y_proba)
                n   = len(y_proba)
                print(f"\n  Winner propensity distribution (test set, n={n}):")
                print(f"    High (>{CONFIDENCE_HIGH:.2f}): {seg['high']['count']} ({seg['high']['count']/n*100:.1f}%)")
                print(f"    Uncertain: {seg['unsure']['count']} ({seg['unsure']['count']/n*100:.1f}%)")
                print(f"    Low (<{CONFIDENCE_LOW:.2f}): {seg['low']['count']} ({seg['low']['count']/n*100:.1f}%)")
        except Exception:
            pass