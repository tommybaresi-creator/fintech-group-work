"""
Diagnostic plots for the KYC pipeline report.
Run after bestmodel_accumulation and bestmodel_income notebooks.

Produces three figures:
  1. Precision-Recall curves (both targets overlaid, threshold marked)
  2. Calibration curves before/after isotonic regression
  3. SHAP beeswarm (one per target)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import shap
from pathlib import Path
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PICKLE_DIR = Path("Data/pickled_files")
TARGETS = {
    "AccumulationInvestment": "xgboost_shap",
    "IncomeInvestment":       "xgboost_shap",
}
COLORS = {
    "AccumulationInvestment": "#2E86AB",
    "IncomeInvestment":       "#E84855",
}
MIFID_PRECISION_FLOOR = 0.75

Path("figures").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

results = {}
for target, folder in TARGETS.items():
    path = PICKLE_DIR / folder / f"{target.lower()}.pkl"
    results[target] = joblib.load(path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def target_label(t):
    return "Accumulation" if "Accum" in t else "Income"


# ===========================================================================
# Figure 1 — Precision-Recall Curves
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (target, r) in zip(axes, results.items()):
    y_true  = r["y_test_true"]
    y_proba = r["y_test_proba"]

    # key is threshold_metrics, not threshold_info
    thresh = r["threshold_metrics"]["threshold"]
    t_prec = r["threshold_metrics"]["precision"]
    t_rec  = r["threshold_metrics"]["recall"]

    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    color = COLORS[target]
    ax.plot(rec, prec, color=color, lw=2, label=f"{target_label(target)} PR curve")

    # MiFID II floor
    ax.axhline(MIFID_PRECISION_FLOOR, color="gray", ls="--", lw=1.2,
               label=f"MiFID II floor (P = {MIFID_PRECISION_FLOOR})")

    # Operating threshold point
    ax.scatter([t_rec], [t_prec], color=color, s=100, zorder=5,
               edgecolors="black", linewidths=0.8,
               label=f"Threshold = {thresh:.3f}  (P={t_prec:.2f}, R={t_rec:.2f})")

    # Shade infeasible region below MiFID floor
    ax.axhspan(0, MIFID_PRECISION_FLOOR, alpha=0.05, color="red")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"{target_label(target)}Investment\nPrecision-Recall Curve", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)

fig.suptitle("Precision-Recall Curves with MiFID II Constraint",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/pr_curves.pdf", bbox_inches="tight", dpi=150)
plt.savefig("figures/pr_curves.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved -> figures/pr_curves.pdf / .png")


# ===========================================================================
# Figure 2 — Calibration Curves
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (target, r) in zip(axes, results.items()):
    y_true     = r["y_test_true"]
    y_proba    = r["y_test_proba"]
    brier_post = r["brier_score"]
    brier_pre  = r["brier_score_pre_cal"]
    no_skill   = r["no_skill_brier"]
    color      = COLORS[target]

    # Post-calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos,
            color=color, lw=2, marker="o", ms=5,
            label=f"After calibration  (Brier = {brier_post:.3f})")

    # Pre-calibration: reconstruct probas from the inner uncalibrated estimator
    X_test = r["shap_test_X"].values
    try:
        raw_model = r["model"].estimator
        y_proba_pre = raw_model.predict_proba(X_test)[:, 1]
        frac_pos_pre, mean_pred_pre = calibration_curve(y_true, y_proba_pre, n_bins=10)
        ax.plot(mean_pred_pre, frac_pos_pre,
                color=color, lw=2, marker="s", ms=5, ls="--", alpha=0.55,
                label=f"Before calibration (Brier = {brier_pre:.3f})")
    except Exception:
        ax.annotate(f"Pre-cal Brier = {brier_pre:.3f}",
                    xy=(0.05, 0.88), xycoords="axes fraction",
                    fontsize=8.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.7))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")

    # No-skill baseline
    pos_rate = y_true.mean()
    ax.axhline(pos_rate, color="gray", ls=":", lw=1, alpha=0.6)
    ax.annotate(f"Positive rate = {pos_rate:.2f}  |  No-skill Brier = {no_skill:.3f}",
                xy=(0.01, pos_rate + 0.02), fontsize=7.5, color="gray")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"{target_label(target)}Investment\nCalibration Curve", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

fig.suptitle("Calibration Curves - Isotonic Regression Post-Hoc Calibration",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/calibration_curves.pdf", bbox_inches="tight", dpi=150)
plt.savefig("figures/calibration_curves.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved -> figures/calibration_curves.pdf / .png")


# ===========================================================================
# Figure 3 — SHAP Beeswarm
# ===========================================================================

fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5)

for col, (target, r) in enumerate(results.items()):
    shap_values   = r["shap_values"]    # (1000, 7)
    X_test        = r["shap_test_X"]    # DataFrame (1000, 7)
    feature_names = r["feature_names"]

    explanation = shap.Explanation(
        values=shap_values,
        data=X_test.values,
        feature_names=feature_names,
    )

    ax = fig.add_subplot(gs[0, col])
    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)
    ax.set_title(f"{target_label(target)}Investment\nSHAP Feature Importance",
                 fontsize=12, fontweight="bold", pad=12)

fig.suptitle("XGBoost SHAP Values - Winning Model (Test Set)",
             fontsize=13, fontweight="bold", y=1.02)
plt.savefig("figures/shap_beeswarm.pdf", bbox_inches="tight", dpi=150)
plt.savefig("figures/shap_beeswarm.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved -> figures/shap_beeswarm.pdf / .png")