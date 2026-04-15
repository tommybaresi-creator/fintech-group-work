"""
show_results.py

"""

import importlib
import logging
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
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


def _separator(title="", width=64):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'='*pad} {title} {'='*pad}")
    else:
        print("=" * width)


def _step(msg):   print(f"\n  ▸ {msg}")
def _result(msg): print(f"    → {msg}")
def _fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"


def _effective_precision(result: dict) -> float:
    """
    C6 fix: return precision at the optimised threshold where available.
    Falls back to precision at 0.5 only when threshold selection was not possible.
    """
    thr_m = result.get("threshold_metrics")
    if thr_m is not None:
        return thr_m["precision"]
    return result["test_metrics"]["precision"]


def _print_full_story(result: dict) -> None:
    model_name  = result["model_name"]
    target_name = result["target_name"]
    cv_sum      = result["cv_metrics_summary"]
    test_m      = result["test_metrics"]
    thr_m       = result.get("threshold_metrics")
    ablation    = result.get("ablation")
    best_params = result.get("best_params")
    brier_pre   = result.get("brier_score_pre_cal")
    brier_post  = result.get("brier_score")
    no_skill    = result.get("no_skill_brier", 0.25)
    y_proba     = result.get("y_test_proba")

    _separator(f"{model_name}  |  {target_name}")

    _step("Data split: 80/20 stratified. Nested val split (from training) for threshold selection.")

    feat_set = result.get("feature_names", [])
    if feat_set == FEATURE_NAMES:
        _step("Feature set: F_E — 10 engineered features")
    elif feat_set == BASELINE_FEATURE_NAMES:
        _step("Feature set: F_B — 7 professor's baseline features")

    scaler = result.get("scaler")
    _step(f"Scaling: {type(scaler).__name__ if scaler else 'none or inside Pipeline'}")

    n_folds = len(result["cv_metrics_raw"]["f1"])
    _step(f"Cross-validation: {n_folds}-fold StratifiedKFold, true nested CV (HP tuning inside each outer fold)")

    if best_params:
        _step("Hyperparameter tuning: RandomizedSearchCV (inner 3-fold CV)")
        for k, v in best_params.items():
            if k not in ("pos_weight_note",):
                _result(f"{k} = {v}")

    if brier_post is not None:
        _step("Calibration: isotonic regression (cv=5)")
        if brier_pre is not None:
            pct = (brier_pre - brier_post) / brier_pre * 100
            _result(f"Brier: {brier_pre:.4f} → {brier_post:.4f}  "
                    f"[no-skill: {no_skill:.4f}]  Δ={pct:.1f}%")
        else:
            _result(f"Brier: {brier_post:.4f}  [no-skill: {no_skill:.4f}]")

    # CV + test metrics table
    print(f"\n  {'─'*58}")
    print(f"  {'Metric':<14} {'CV Mean ± Std':>18}  {'Test @ 0.5':>12}")
    print(f"  {'─'*58}")
    for m in ["accuracy", "precision", "recall", "f1"]:
        cv_str   = _fmt(cv_sum[m]["mean"], cv_sum[m]["std"])
        test_str = f"{test_m[m]:.3f}"
        marker   = "  ◄ PRIMARY" if m == "f1" else ""
        print(f"  {m:<14} {cv_str:>18}  {test_str:>12}{marker}")
    print(f"  {'─'*58}")

    # Unbiased threshold metrics
    if thr_m:
        eff_p = thr_m["precision"]
        mifid = "✓" if eff_p >= PRECISION_FLOOR else "✗"
        print(f"\n  MiFID II — threshold selected on VALIDATION set (unbiased):")
        print(f"  Threshold = {thr_m['threshold']:.3f}  |  "
              f"P={thr_m['precision']:.3f}  R={thr_m['recall']:.3f}  F1={thr_m['f1']:.3f}  "
              f"MiFID {mifid}")
    else:
        eff_p = test_m["precision"]
        mifid = "✓" if eff_p >= PRECISION_FLOOR else "✗"
        print(f"\n  MiFID II — precision at 0.5 threshold (no threshold optimisation): "
              f"{eff_p:.3f}  {mifid}")

    # Ablation — N7 fix: symmetric ±0.002 threshold
    if ablation:
        eng  = ablation.get("engineered", {})
        base = ablation.get("baseline", {})
        if eng and base:
            eng_f1  = eng["cv_metrics_summary"]["f1"]
            base_f1 = base["cv_metrics_summary"]["f1"]
            delta   = eng_f1["mean"] - base_f1["mean"]
            # N7 fix: symmetric thresholds
            if delta > 0.002:
                winner = "F_E"
            elif delta < -0.002:
                winner = "F_B"
            else:
                winner = "EQUIVALENT"
            print(f"\n  Feature set ablation (both sets independently tuned):")
            print(f"    F_E: {_fmt(eng_f1['mean'], eng_f1['std'])}")
            print(f"    F_B: {_fmt(base_f1['mean'], base_f1['std'])}")
            print(f"    ΔF1 (F_E − F_B): {delta:+.3f}  →  Winner: {winner}")

    # Propensity
    if y_proba is not None:
        seg = segment_by_confidence(y_proba)
        n   = len(y_proba)
        print(f"\n  Propensity (test set, n={n}):")
        print(f"    mean={y_proba.mean():.3f}  std={y_proba.std():.3f}  "
              f"[{y_proba.min():.3f}, {y_proba.max():.3f}]")
        print(f"    High (>{CONFIDENCE_HIGH:.2f}): {seg['high']['count']:4d} ({seg['high']['count']/n*100:.1f}%)  → automate")
        print(f"    Uncertain:          {seg['unsure']['count']:4d} ({seg['unsure']['count']/n*100:.1f}%)  → human")
        print(f"    Low (<{CONFIDENCE_LOW:.2f}):  {seg['low']['count']:4d} ({seg['low']['count']/n*100:.1f}%)  → no action")

    # M5 note: SHAP–calibration mismatch
    if result.get("shap_values") is not None:
        print(f"\n  NOTE (M5): SHAP values computed on uncalibrated XGBoost (TreeExplainer "
              f"requires tree structure, unavailable in CalibratedClassifierCV wrapper). "
              f"Propensity scores use the calibrated model. SHAP explains the pre-calibration "
              f"log-odds mapping, not the final output probabilities.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_pr_curve(result: dict) -> None:
    y_true  = result.get("y_test_true")
    y_proba = result.get("y_test_proba")
    thr_m   = result.get("threshold_metrics")
    if y_true is None or y_proba is None:
        return
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls[:-1], precisions[:-1], lw=1.5, label="PR curve")
    ax.axhline(PRECISION_FLOOR, color="red", linestyle="--", lw=1,
               label=f"MiFID II floor ({PRECISION_FLOOR:.2f})")
    if thr_m:
        ax.scatter(thr_m["recall"], thr_m["precision"], s=80, zorder=5, color="red",
                   label=f"Val-selected thr={thr_m['threshold']:.3f} (unbiased)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — {result['model_name']}  |  {result['target_name']}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_confusion_matrix(result: dict) -> None:
    y_true = result.get("y_test_true"); y_pred = result.get("y_test_pred")
    if y_true is None or y_pred is None: return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {result['model_name']}  |  {result['target_name']}")
    plt.tight_layout(); plt.show()


def _plot_cv_f1_boxplot(results_by_target: dict) -> None:
    n = len(results_by_target)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, (target, result) in zip(axes, results_by_target.items()):
        if result is None: continue
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
    if not ablation: return
    eng = ablation.get("engineered"); base = ablation.get("baseline")
    if not eng or not base: return
    means  = [eng["cv_metrics_summary"]["f1"]["mean"],  base["cv_metrics_summary"]["f1"]["mean"]]
    stds   = [eng["cv_metrics_summary"]["f1"]["std"],   base["cv_metrics_summary"]["f1"]["std"]]
    colors = ["#4C72B0" if means[0] >= means[1] else "#DD8452",
              "#DD8452" if means[0] >= means[1] else "#4C72B0"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["F_E", "F_B"], means, yerr=stds, capsize=5, color=colors, width=0.4)
    ax.set_ylabel("CV F1 (mean ± std)")
    ax.set_title(f"Ablation — {result['model_name']}  |  {result['target_name']}")
    ax.set_ylim(max(0, min(means) - 0.05), min(1, max(means) + 0.08))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + stds[0] + 0.01,
                f"{mean:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()


def _plot_propensity_histogram(result: dict) -> None:
    y_proba = result.get("y_test_proba")
    if y_proba is None: return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_proba, bins=30, edgecolor="white", color="#4C72B0", alpha=0.8)
    ax.axvline(CONFIDENCE_HIGH, color="green",  linestyle="--", lw=1.5,
               label=f"Automate (>{CONFIDENCE_HIGH:.2f})")
    ax.axvline(CONFIDENCE_LOW,  color="orange", linestyle="--", lw=1.5,
               label=f"Low (<{CONFIDENCE_LOW:.2f})")
    ax.set_xlabel("Propensity"); ax.set_ylabel("Count")
    ax.set_title(f"Propensity — {result['model_name']}  |  {result['target_name']}")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()


def _plot_shap(result: dict) -> None:
    try:
        import shap
    except ImportError:
        return
    shap_values = result.get("shap_values"); shap_test_X = result.get("shap_test_X")
    if shap_values is None or shap_test_X is None: return
    shap.summary_plot(shap_values, shap_test_X, show=True)


def _show_all_plots(result: dict) -> None:
    _plot_pr_curve(result)
    _plot_confusion_matrix(result)
    _plot_propensity_histogram(result)
    _plot_ablation_bar(result)
    _plot_shap(result)


# ---------------------------------------------------------------------------
# Model runner
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
                "best_params":        None,
                "threshold_metrics":  None,
                "brier_score":        None,
                "no_skill_brier":     None,
            }
            _print_full_story(sub)
            _show_all_plots(sub)
        return

    for target, result in data.items():
        _print_full_story(result)
        _show_all_plots(result)
    if len(data) > 1:
        _plot_cv_f1_boxplot(data)


def show_xgboost(retrain=True):       _show_model("xgboost",          retrain)
def show_logistic_reg(retrain=True):  _show_model("logistic_reg",      retrain)
def show_naive_bayes(retrain=True):   _show_model("naive_bayes",       retrain)
def show_random_forest(retrain=True): _show_model("random_forest",     retrain)
def show_mlp(retrain=True):           _show_model("mlp",               retrain)
def show_classifier_chain(retrain=True): _show_model("classifier_chain", retrain)
def show_soft_voting(retrain=True):   _show_model("soft_voting",       retrain)
def show_hard_voting(retrain=True):   _show_model("hard_voting",       retrain)
def show_all(retrain=True):
    for key in _MODEL_MAP:
        _show_model(key, retrain)


def show_winner() -> None:
    """
    Formal model selection per target.

    C6 fix: MiFID compliance gate uses precision at the OPTIMISED threshold
    (threshold_metrics["precision"]) where available, not at 0.5.

    M2 note: With 10 folds, Wilcoxon has very low power for small differences.
    Results should be read as 'cannot distinguish' rather than 'equivalent'.

    M3 fix: All pairwise Wilcoxon tests among top-3 compliant models,
    not just top-2.
    """
    _separator("MODEL SELECTION — WILCOXON SIGNED-RANK TESTS", width=64)
    print("\n  IMPORTANT NOTES:")
    print("  • MiFID II compliance tested at OPTIMISED threshold (C6 fix), not at 0.5")
    print("  • Wilcoxon with 10 folds has low power — 'p>0.05' means 'cannot distinguish'")
    print("  • All pairs among top-3 compliant models are tested (M3 fix)\n")

    folders = {
        "XGBoost":      "xgboost_shap",
        "LR":           "linear_reg",
        "NaiveBayes":   "naive_bayes",
        "RandomForest": "rand_forest",
        "MLP":          "mlp",
        "SoftVoting":   "soft_voting_ens",
        "HardVoting":   "hard_voting_ens",
    }

    for target in TARGETS:
        _separator(target)
        results = {}
        for name, folder in folders.items():
            try:
                r = load_result(folder, target)
                f1_folds = r["cv_metrics_raw"]["f1"]
                # C6 fix: use effective precision at optimised threshold
                eff_prec = _effective_precision(r)
                results[name] = {
                    "f1_folds":          f1_folds,
                    "cv_f1_mean":        np.mean(f1_folds),
                    "cv_f1_std":         np.std(f1_folds, ddof=1),
                    "eff_precision":     eff_prec,
                    "test_f1":           r["test_metrics"]["f1"],
                    "thr_f1":            r.get("threshold_metrics", {}).get("f1"),
                    "passes_constraint": eff_prec >= PRECISION_FLOOR,
                }
            except FileNotFoundError:
                logger.warning("  Skipping %s — no pickle", name)

        if not results:
            print("  No results found."); continue

        passing = {k: v for k, v in results.items() if v["passes_constraint"]}
        failing = {k: v for k, v in results.items() if not v["passes_constraint"]}

        print(f"\n  {'Model':<20} {'CV F1':>16}  {'Eff Prec':>10}  {'Test F1':>8}  {'Thr F1':>8}  {'MiFID':>6}")
        print(f"  {'─'*72}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["cv_f1_mean"]):
            ok     = "✓" if r["passes_constraint"] else "✗"
            thr_f1 = f"{r['thr_f1']:.3f}" if r["thr_f1"] is not None else "  N/A"
            print(f"  {name:<20} {_fmt(r['cv_f1_mean'], r['cv_f1_std']):>16}  "
                  f"{r['eff_precision']:>10.3f}  {r['test_f1']:>8.3f}  {thr_f1:>8}  {ok:>6}")

        if failing:
            print(f"\n  ✗ Excluded: {', '.join(failing.keys())}")

        candidates = passing if passing else results
        ranked = sorted(candidates.items(), key=lambda x: -x[1]["cv_f1_mean"])

        # M3 fix: all pairs among top-3
        top3 = ranked[:3]
        complexity = {"NaiveBayes": 1, "LR": 2, "XGBoost": 3, "RandomForest": 3,
                      "MLP": 4, "SoftVoting": 5, "HardVoting": 5}

        print(f"\n  Pairwise Wilcoxon tests (top-{min(3,len(top3))}):")
        all_equivalent = True
        for i in range(len(top3)):
            for j in range(i + 1, len(top3)):
                n_a, d_a = top3[i]; n_b, d_b = top3[j]
                try:
                    stat, p_val = wilcoxon(d_a["f1_folds"], d_b["f1_folds"])
                    sig = "not significant" if p_val > 0.05 else "SIGNIFICANT"
                    print(f"    {n_a} vs {n_b}: stat={stat:.1f}  p={p_val:.4f}  [{sig}]")
                    if p_val <= 0.05:
                        all_equivalent = False
                except Exception as e:
                    print(f"    {n_a} vs {n_b}: test failed ({e})")

        best_name = ranked[0][0]
        if all_equivalent and len(top3) > 1:
            # All equivalent — Occam's razor over top-3
            winner = min([n for n, _ in top3], key=lambda x: complexity.get(x, 99))
            print(f"\n  All top models statistically equivalent → Occam's razor.")
            print(f"  ★ WINNER for {target}: {winner}")
        else:
            print(f"\n  ★ WINNER for {target}: {best_name}")

        # Winner propensity
        try:
            winner_result = load_result(folders.get(best_name, ""), target)
            y_proba = winner_result.get("y_test_proba")
            if y_proba is not None:
                seg = segment_by_confidence(y_proba)
                n   = len(y_proba)
                print(f"\n  Winner propensity (n={n}):")
                print(f"    High: {seg['high']['count']} ({seg['high']['count']/n*100:.1f}%)")
                print(f"    Uncertain: {seg['unsure']['count']} ({seg['unsure']['count']/n*100:.1f}%)")
                print(f"    Low: {seg['low']['count']} ({seg['low']['count']/n*100:.1f}%)")
        except Exception:
            pass