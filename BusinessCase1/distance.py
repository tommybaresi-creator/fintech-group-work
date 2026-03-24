import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Gower helper ──────────────────────────────────────────────────────────────

def build_gower_cat_mask(
    df: pd.DataFrame,
    categorical_cols: List[str],
) -> np.ndarray:
    """
    Build the boolean cat_features mask required by gower.gower_matrix,
    guaranteeing it matches the actual column order of df.

    Parameters
    ----------
    df               : the DataFrame we will pass to gower.gower_matrix
    categorical_cols : names of columns that should be treated as categorical

    Returns
    -------
    mask : boolean array of shape (len(df.columns),)
           mask[i] is True  ↔ df.columns[i] is categorical
           mask[i] is False ↔ df.columns[i] is numerical
    """
    cat_set = set(categorical_cols)

    # ── Validate that every requested cat col actually exists in df ───────────
    missing = cat_set - set(df.columns)
    if missing:
        raise ValueError(
            f"build_gower_cat_mask: the following categorical columns are not "
            f"present in the DataFrame: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # ── Build mask in df.columns order (the only correct order) ──────────────
    mask = np.array([col in cat_set for col in df.columns])

    # ── Log the full mapping so it can be visually verified ──────────────────
    logger.info("── Gower cat_features mask ──")
    logger.info("  DataFrame columns (%d total):", len(df.columns))
    for i, col in enumerate(df.columns):
        col_type = "CATEGORICAL" if mask[i] else "numerical"
        logger.info("    [%d] %-30s → %s", i, col, col_type)

    n_cat = mask.sum()
    n_num = (~mask).sum()
    logger.info(
        "  Summary: %d numerical, %d categorical  (mask=%s)",
        n_num, n_cat, mask.astype(int).tolist(),
    )

    # ── Sanity: warn if no categoricals or no numericals ─────────────────────
    if n_cat == 0:
        logger.warning(
            "build_gower_cat_mask: no categorical columns selected — "
            "Gower will behave like a pure numerical distance."
        )
    if n_num == 0:
        logger.warning(
            "build_gower_cat_mask: no numerical columns selected — "
            "Gower will behave like a pure categorical distance."
        )

    return mask


# ── Input pre-flight checks ───────────────────────────────────────────────────
#
# Each function inspects the array BEFORE any distance is computed.
# They log a per-column report (INFO pass, WARNING/ERROR fail) and return
# True (all clean) / False (problems found).
# Matrix functions call the appropriate check at the very top so problems
# surface immediately with a clear message rather than silent wrong results.


def _check_input_numerical(
    X: np.ndarray,
    name: str = "X_num",
    tol: float = 1e-6,
) -> bool:
    """
    Pre-flight checks for a min-max scaled numerical matrix.

    Checks
    ------
    1. dtype is float
    2. No NaN per column
    3. No Inf per column
    4. All values in [0, 1]
    5. No constant columns (range == 0)
    6. No all-zero columns
    """
    logger.info("── Pre-flight check: numerical '%s'  shape=%s ──", name, X.shape)
    passed = True
    n_samples, n_features = X.shape

    # 1. dtype
    if not np.issubdtype(X.dtype, np.floating):
        logger.error(
            "[%s] FAIL dtype: expected float, got %s — "
            "cast with .astype(float) or rerun MinMaxScaler", name, X.dtype,
        )
        passed = False
    else:
        logger.info("[%s] OK   dtype: %s", name, X.dtype)

    # 2. NaN per column
    nan_per_col = np.isnan(X).sum(axis=0)
    bad_nan = np.where(nan_per_col > 0)[0].tolist()
    if bad_nan:
        logger.error(
            "[%s] FAIL NaN: %d column(s) contain NaN — cols %s — impute before scaling",
            name, len(bad_nan), bad_nan,
        )
        passed = False
    else:
        logger.info("[%s] OK   NaN: none", name)

    # 3. Inf per column
    inf_per_col = np.isinf(X).sum(axis=0)
    bad_inf = np.where(inf_per_col > 0)[0].tolist()
    if bad_inf:
        logger.error("[%s] FAIL Inf: %d column(s) — cols %s", name, len(bad_inf), bad_inf)
        passed = False
    else:
        logger.info("[%s] OK   Inf: none", name)

    # 4. Range [0, 1]
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    below   = np.where(col_min < -tol)[0].tolist()
    above   = np.where(col_max >  1.0 + tol)[0].tolist()
    if below or above:
        logger.error(
            "[%s] FAIL range [0,1]: %d col(s) below 0 %s, %d col(s) above 1 %s — "
            "apply MinMaxScaler", name, len(below), below, len(above), above,
        )
        passed = False
    else:
        logger.info(
            "[%s] OK   range [0,1]: global min=%.6f  max=%.6f",
            name, float(col_min.min()), float(col_max.max()),
        )

    # 5. Constant columns
    col_range  = col_max - col_min
    const_cols = np.where(col_range < tol)[0].tolist()
    if const_cols:
        logger.warning(
            "[%s] WARN constant: %d col(s) are constant (range≈0) — %s — "
            "contribute nothing to distances", name, len(const_cols), const_cols,
        )
    else:
        logger.info("[%s] OK   constant: no constant columns", name)

    # 6. All-zero columns
    allzero = np.where(col_max < tol)[0].tolist()
    if allzero:
        logger.warning(
            "[%s] WARN all-zero: %d col(s) are all zero — %s", name, len(allzero), allzero,
        )

    status = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    (logger.info if passed else logger.error)(
        "[%s] numerical pre-flight: %s (%d features)", name, status, n_features
    )
    return passed


def _check_input_categorical_int(
    X: np.ndarray,
    name: str = "X_cat",
) -> bool:
    """
    Pre-flight checks for a label-encoded categorical matrix (Hamming).

    Checks
    ------
    1. dtype is integer
    2. No negative codes
    3. Per column: >= 2 unique values
    4. Per column: codes contiguous from 0 (0..n_classes-1)
    """
    logger.info("── Pre-flight check: label-encoded categorical '%s'  shape=%s ──", name, X.shape)
    passed = True
    n_samples, n_features = X.shape

    # 1. dtype integer
    if not np.issubdtype(X.dtype, np.integer):
        logger.error(
            "[%s] FAIL dtype: expected integer, got %s — "
            "apply LabelEncoder and cast to int", name, X.dtype,
        )
        passed = False
    else:
        logger.info("[%s] OK   dtype: %s", name, X.dtype)

    # 2. Negative codes
    if np.issubdtype(X.dtype, np.integer):
        neg_cols = np.where((X < 0).any(axis=0))[0].tolist()
        if neg_cols:
            logger.error(
                "[%s] FAIL negative codes: cols %s — re-run LabelEncoder", name, neg_cols,
            )
            passed = False
        else:
            logger.info("[%s] OK   negative codes: none", name)

    # 3 & 4. Per-column uniqueness and contiguity
    for j in range(n_features):
        col    = X[:, j]
        unique = np.unique(col)
        n_uniq = len(unique)

        if n_uniq < 2:
            logger.warning(
                "[%s] WARN col %d: only %d unique value(s) %s — constant, "
                "contributes nothing to Hamming", name, j, n_uniq, unique.tolist(),
            )
        else:
            logger.info(
                "[%s] OK   col %d: %d unique values %s", name, j, n_uniq, unique.tolist(),
            )

        if not np.array_equal(unique, np.arange(n_uniq)):
            logger.warning(
                "[%s] WARN col %d: codes not contiguous 0..%d — got %s — "
                "re-fit LabelEncoder on full dataset", name, j, n_uniq - 1, unique.tolist(),
            )

    status = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    (logger.info if passed else logger.error)(
        "[%s] categorical-int pre-flight: %s (%d features)", name, status, n_features
    )
    return passed


def _check_input_binary(
    X: np.ndarray,
    name: str = "X_bin",
    tol: float = 1e-6,
) -> bool:
    """
    Pre-flight checks for a one-hot encoded binary matrix (Tanimoto).

    Checks
    ------
    1. Values strictly in {0, 1} — checked before any cast
    2. No NaN or Inf
    3. No all-zero columns (category never appears)
    4. No all-one columns (category always appears, contributes nothing)
    5. Row-sum report (one-hot groups should each sum to 1 per row)
    """
    logger.info("── Pre-flight check: binary (one-hot) '%s'  shape=%s ──", name, X.shape)
    passed = True
    n_samples, n_features = X.shape

    # 1. Binary values — before any cast so 0.5 is caught, not silently floored
    unique_vals = np.unique(X)
    non_binary  = [v for v in unique_vals if v not in (0, 1)]
    if non_binary:
        logger.error(
            "[%s] FAIL binary values: found %s — "
            "do not apply float scaling to one-hot columns", name, non_binary,
        )
        passed = False
    else:
        logger.info("[%s] OK   binary values: only {0, 1}", name)

    # 2. NaN / Inf
    X_f   = X.astype(float)
    n_nan = int(np.isnan(X_f).sum())
    n_inf = int(np.isinf(X_f).sum())
    if n_nan or n_inf:
        logger.error("[%s] FAIL NaN/Inf: %d NaN, %d Inf", name, n_nan, n_inf)
        passed = False
    else:
        logger.info("[%s] OK   NaN/Inf: none", name)

    # 3 & 4. All-zero / all-one columns
    col_sum      = X.sum(axis=0)
    allzero_cols = np.where(col_sum == 0)[0].tolist()
    allone_cols  = np.where(col_sum == n_samples)[0].tolist()
    if allzero_cols:
        logger.warning(
            "[%s] WARN all-zero cols: %s — category never present, "
            "inflates sparsity", name, allzero_cols,
        )
    if allone_cols:
        logger.warning(
            "[%s] WARN all-one cols: %s — category always present, "
            "contributes nothing to Tanimoto", name, allone_cols,
        )
    if not allzero_cols and not allone_cols:
        col_mean = col_sum / n_samples
        logger.info(
            "[%s] OK   column activity: min_freq=%.3f  max_freq=%.3f  mean_freq=%.3f",
            name, float(col_mean.min()), float(col_mean.max()), float(col_mean.mean()),
        )

    # 5. Row-sum report
    row_sum  = X.sum(axis=1)
    min_rsum = int(row_sum.min())
    max_rsum = int(row_sum.max())
    logger.info(
        "[%s] INFO row sums: min=%d  max=%d  mean=%.2f — "
        "each group of one-hot cols should sum to exactly 1 per row",
        name, min_rsum, max_rsum, float(row_sum.mean()),
    )
    if min_rsum == 0:
        logger.warning(
            "[%s] WARN %d row(s) have row sum=0 — "
            "Tanimoto returns 0.0 for these pairs", name, int((row_sum == 0).sum()),
        )

    status = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    (logger.info if passed else logger.error)(
        "[%s] binary pre-flight: %s (%d features)", name, status, n_features
    )
    return passed


# ── Distance matrix output validation ────────────────────────────────────────

def check_distance_matrix(D: np.ndarray, name: str = "D", tol: float = 1e-6) -> bool:
    """
    Post-computation sanity checks on a distance matrix.

    Checks: square, zero diagonal, symmetry, non-negativity,
    range [0, 1], no NaN/Inf, off-diagonal degenerate check.
    """
    logger.info("── Checking distance matrix: %s  shape=%s ──", name, D.shape)
    passed = True

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        logger.error("[%s] FAIL shape: not square — got %s", name, D.shape)
        return False
    n = D.shape[0]

    max_diag = np.abs(np.diag(D)).max()
    if max_diag > tol:
        logger.error("[%s] FAIL diagonal: max |D[i,i]|=%.2e", name, max_diag)
        passed = False
    else:
        logger.info("[%s] OK   diagonal: %.2e", name, max_diag)

    max_asym = np.abs(D - D.T).max()
    if max_asym > tol:
        logger.error("[%s] FAIL symmetry: max asymmetry=%.2e", name, max_asym)
        passed = False
    else:
        logger.info("[%s] OK   symmetry: %.2e", name, max_asym)

    n_neg = int((D < -tol).sum())
    if n_neg:
        logger.error("[%s] FAIL non-negativity: %d negative entries", name, n_neg)
        passed = False
    else:
        logger.info("[%s] OK   non-negativity: min=%.6f", name, D.min())

    d_max = D.max()
    if d_max > 1.0 + tol:
        logger.error("[%s] FAIL range: max=%.6f > 1.0", name, d_max)
        passed = False
    else:
        logger.info("[%s] OK   range: [%.6f, %.6f]", name, D.min(), d_max)

    n_nan = int(np.isnan(D).sum())
    n_inf = int(np.isinf(D).sum())
    if n_nan or n_inf:
        logger.error("[%s] FAIL NaN/Inf: %d NaN, %d Inf", name, n_nan, n_inf)
        passed = False
    else:
        logger.info("[%s] OK   NaN/Inf: none", name)

    off  = D[np.triu_indices(n, k=1)]
    zf   = float((off < tol).sum()) / len(off) if len(off) else 0.0
    if zf > 0.5:
        logger.warning(
            "[%s] WARN degenerate: %.1f%% off-diagonal ~0 (mean=%.4f)",
            name, zf * 100, off.mean(),
        )
    else:
        logger.info(
            "[%s] OK   off-diagonal: mean=%.4f  ~0 frac=%.1f%%",
            name, off.mean(), zf * 100,
        )

    status = "ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED"
    (logger.info if passed else logger.error)("[%s] %s", name, status)
    return passed


# ── Pairwise scalar distances ─────────────────────────────────────────────────
#
# All return a value in [0, 1] (assuming X_num is min-max scaled).
# Normalized = divided by weighted number of features so alpha blending
# in mixed_distance_matrix is meaningful on a common scale.


def hamming_distance(
    x_cat: np.ndarray,
    y_cat: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted Hamming distance for label-encoded categoricals. Result in [0, 1]."""
    x_cat = np.asarray(x_cat)
    y_cat = np.asarray(y_cat)
    if x_cat.shape != y_cat.shape:
        raise ValueError("`x_cat` and `y_cat` must have the same shape.")
    weights    = _validate_weights(weights, x_cat.shape[0])
    mismatches = (x_cat != y_cat).astype(float)
    result     = float(np.sum(weights * mismatches) / weights.sum())
    logger.debug("hamming_distance → %.6f", result)
    return result


def tanimoto_distance(
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted Tanimoto distance for one-hot encoded categoricals. Result in [0, 1]."""
    x_bin = np.asarray(x_bin)
    y_bin = np.asarray(y_bin)
    if not np.all(np.isin(x_bin, [0, 1])):
        raise ValueError("`x_bin` must contain only binary values (0/1).")
    if not np.all(np.isin(y_bin, [0, 1])):
        raise ValueError("`y_bin` must contain only binary values (0/1).")
    x_bin = x_bin.astype(np.int32)
    y_bin = y_bin.astype(np.int32)
    if x_bin.shape != y_bin.shape:
        raise ValueError("`x_bin` and `y_bin` must have the same shape.")
    weights      = _validate_weights(weights, x_bin.shape[0])
    intersection = np.sum(weights * ((x_bin == 1) & (y_bin == 1)))
    union        = np.sum(weights * ((x_bin == 1) | (y_bin == 1)))
    result       = 0.0 if union == 0 else float(1.0 - intersection / union)
    logger.debug("tanimoto_distance → %.6f", result)
    return result


def L1_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted normalized L1 (Manhattan) distance. Assumes [0,1] input. Result in [0, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    result  = float(np.sum(weights * np.abs(x - y)) / weights.sum())
    logger.debug("L1_distance → %.6f", result)
    return result


def L2_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted normalized L2 (Euclidean) distance. Assumes [0,1] input. Result in [0, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    result  = float(np.sqrt(np.sum(weights * (x - y) ** 2) / weights.sum()))
    logger.debug("L2_distance → %.6f", result)
    return result


def canberra_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted normalized Canberra distance. Result in [0, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    num     = np.abs(x - y)
    den     = np.abs(x) + np.abs(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = np.where(den != 0, num / den, 0.0)
    result  = float(np.sum(weights * term) / weights.sum())
    logger.debug("canberra_distance → %.6f", result)
    return result


# ── Distance matrices (numerical only) ───────────────────────────────────────

def numerical_distance_matrix(
    X_num: np.ndarray,
    metric: str = "L1",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pairwise distance matrix for min-max scaled numerical variables.
    All results in [0, 1].

    Pre-flight: float dtype, no NaN/Inf, values in [0,1], no constant cols.
    """
    X_num = np.asarray(X_num, dtype=float)
    n_samples, n_features = X_num.shape

    if metric not in ("L1", "L2", "Canberra"):
        raise ValueError("`metric` must be one of: 'L1', 'L2', 'Canberra'.")

    _check_input_numerical(X_num, name=f"X_num[{metric}]")

    weights_arr = _validate_weights(weights, n_features)
    uniform     = weights is None or np.allclose(weights_arr, weights_arr[0])

    logger.info(
        "numerical_distance_matrix | metric=%s  samples=%d  features=%d  weighted=%s",
        metric, n_samples, n_features, not uniform,
    )

    if uniform:
        scipy_map = {"L1": "cityblock", "L2": "euclidean", "Canberra": "canberra"}
        D = cdist(X_num, X_num, metric=scipy_map[metric])
        if metric == "L1":
            D /= n_features
        elif metric == "L2":
            D /= np.sqrt(n_features)
        elif metric == "Canberra":
            D /= n_features
        logger.info("Used scipy fast path (uniform weights).")
    else:
        logger.info("Non-uniform weights — using weighted loop.")
        fn = {"L1": L1_distance, "L2": L2_distance, "Canberra": canberra_distance}[metric]
        D  = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = fn(X_num[i], X_num[j], weights=weights_arr)
                D[j, i] = D[i, j]

    logger.info("numerical_distance_matrix | done — shape=%s  min=%.4f  max=%.4f",
                D.shape, D.min(), D.max())
    return D


# ── Distance matrices (categorical only) ─────────────────────────────────────

def hamming_distance_matrix(
    X_cat: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pairwise Hamming distance matrix for label-encoded categoricals.
    Result in [0, 1].

    Pre-flight: integer dtype, no negatives, >=2 unique values per col,
    contiguous codes from 0.
    """
    X_cat = np.asarray(X_cat)
    n_samples, n_features = X_cat.shape

    _check_input_categorical_int(X_cat, name="X_cat[Hamming]")

    weights_arr = _validate_weights(weights, n_features)
    uniform     = weights is None or np.allclose(weights_arr, weights_arr[0])

    logger.info(
        "hamming_distance_matrix | samples=%d  features=%d  weighted=%s",
        n_samples, n_features, not uniform,
    )

    if uniform:
        D = cdist(X_cat, X_cat, metric="hamming")
        logger.info("Used scipy fast path (uniform weights).")
    else:
        logger.info("Non-uniform weights — using weighted loop.")
        D = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = hamming_distance(X_cat[i], X_cat[j], weights=weights_arr)
                D[j, i] = D[i, j]

    logger.info("hamming_distance_matrix | done — shape=%s  min=%.4f  max=%.4f",
                D.shape, D.min(), D.max())
    return D


def tanimoto_distance_matrix(
    X_bin: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pairwise Tanimoto distance matrix for one-hot encoded categoricals.
    Result in [0, 1].

    Pre-flight: binary values only (before cast), no NaN/Inf,
    no all-zero/all-one cols, row-sum report.
    """
    X_bin = np.asarray(X_bin)

    _check_input_binary(X_bin, name="X_bin[Tanimoto]")

    X_bin = X_bin.astype(np.int32)
    n_samples, n_features = X_bin.shape
    weights_arr = _validate_weights(weights, n_features)

    logger.info(
        "tanimoto_distance_matrix | samples=%d  features=%d",
        n_samples, n_features,
    )

    D = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            D[i, j] = tanimoto_distance(X_bin[i], X_bin[j], weights=weights_arr)
            D[j, i] = D[i, j]

    logger.info("tanimoto_distance_matrix | done — shape=%s  min=%.4f  max=%.4f",
                D.shape, D.min(), D.max())
    return D


# ── Mixed distance (block-level) ──────────────────────────────────────────────

def mixed_distance(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y_num: np.ndarray,
    y_cat: np.ndarray,
    alpha: float,
    num_dist: str = "L1",
    cat_dist: str = "Hamming",
    num_weights: Optional[np.ndarray] = None,
    cat_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Mixed distance between two observations.

    Both blocks normalized to [0, 1]. X_num must be min-max scaled.
    Returns alpha * d_num + (1 - alpha) * d_cat in [0, 1].

    Neutral alpha (equal feature weighting, comparable to Gower):
        alpha = n_num / (n_num + n_cat)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("`alpha` must be between 0 and 1.")

    num_fn = {"L1": L1_distance, "L2": L2_distance, "Canberra": canberra_distance}
    cat_fn = {"Hamming": hamming_distance, "Tanimoto": tanimoto_distance}

    if num_dist not in num_fn:
        raise ValueError("`num_dist` must be one of: 'L1', 'L2', 'Canberra'.")
    if cat_dist not in cat_fn:
        raise ValueError("`cat_dist` must be one of: 'Hamming', 'Tanimoto'.")

    d_num  = num_fn[num_dist](x_num, y_num, num_weights)
    d_cat  = cat_fn[cat_dist](x_cat, y_cat, cat_weights)
    result = float(alpha * d_num + (1.0 - alpha) * d_cat)
    logger.debug(
        "mixed_distance | %s+%s  alpha=%.3f  d_num=%.4f  d_cat=%.4f  → %.6f",
        num_dist, cat_dist, alpha, d_num, d_cat, result,
    )
    return result


def _compute_mixed_row(
    i: int,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    alpha: float,
    num_dist: str,
    cat_dist: str,
    num_weights: Optional[np.ndarray],
    cat_weights: Optional[np.ndarray],
) -> list:
    """One upper-triangle row for joblib parallelism."""
    X_cat = np.asarray(X_cat, dtype=np.int32)
    n     = len(X_num)
    row   = np.zeros(n, dtype=float)
    for j in range(i + 1, n):
        row[j] = mixed_distance(
            X_num[i], X_cat[i], X_num[j], X_cat[j],
            alpha=alpha, num_dist=num_dist, cat_dist=cat_dist,
            num_weights=num_weights, cat_weights=cat_weights,
        )
    return row.tolist()


def mixed_distance_matrix(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    alpha: float,
    num_dist: str = "L1",
    cat_dist: str = "Hamming",
    num_weights: Optional[np.ndarray] = None,
    cat_weights: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Pairwise mixed distance matrix.

    X_num must be min-max scaled to [0, 1].
    X_cat: label-encoded integers for Hamming, binary integers for Tanimoto.

    Neutral alpha (mimics Gower equal-feature weighting):
        alpha = X_num.shape[1] / (X_num.shape[1] + X_cat.shape[1])

    Pre-flight:
      X_num            → float dtype, [0,1], no NaN/Inf, no constant cols
      X_cat (Hamming)  → integer dtype, no negatives, contiguous codes
      X_cat (Tanimoto) → binary values only, no NaN/Inf, column activity
    """
    X_num = np.asarray(X_num, dtype=float)
    X_cat = np.asarray(X_cat)   # keep original dtype for pre-flight

    if X_num.shape[0] != X_cat.shape[0]:
        raise ValueError("`X_num` and `X_cat` must have the same number of rows.")

    n_samples = X_num.shape[0]
    n_num     = X_num.shape[1]
    n_cat     = X_cat.shape[1]

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _check_input_numerical(X_num, name=f"X_num[{num_dist}+{cat_dist}]")

    if cat_dist == "Hamming":
        _check_input_categorical_int(X_cat, name=f"X_cat[{num_dist}+Hamming]")
    elif cat_dist == "Tanimoto":
        _check_input_binary(X_cat, name=f"X_cat[{num_dist}+Tanimoto]")

    X_cat = X_cat.astype(np.int32)

    logger.info(
        "mixed_distance_matrix | %s+%s  alpha=%.3f  "
        "samples=%d  num_features=%d  cat_features=%d  n_jobs=%d",
        num_dist, cat_dist, alpha, n_samples, n_num, n_cat, n_jobs,
    )

    alpha_neutral = n_num / (n_num + n_cat)
    if abs(alpha - alpha_neutral) > 0.15:
        logger.warning(
            "alpha=%.3f deviates from neutral baseline %.3f (n_num=%d, n_cat=%d) — "
            "numerical block is %s.",
            alpha, alpha_neutral, n_num, n_cat,
            "over-weighted" if alpha > alpha_neutral else "under-weighted",
        )
    else:
        logger.info("alpha=%.3f  neutral baseline=%.3f", alpha, alpha_neutral)

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_mixed_row)(
            i, X_num, X_cat, alpha, num_dist, cat_dist, num_weights, cat_weights,
        )
        for i in range(n_samples)
    )

    D = np.array(rows, dtype=float)
    D = D + D.T

    logger.info("mixed_distance_matrix | done — shape=%s  min=%.4f  max=%.4f",
                D.shape, D.min(), D.max())
    return D


# ── Helper ────────────────────────────────────────────────────────────────────

def _validate_weights(
    weights: Optional[np.ndarray],
    n_features: int,
) -> np.ndarray:
    """Validate weights array, defaulting to uniform ones if None."""
    if weights is None:
        return np.ones(n_features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != n_features:
        raise ValueError(f"`weights` must have length {n_features}, got {weights.shape[0]}.")
    if np.any(weights < 0):
        raise ValueError("`weights` must be non-negative.")
    if weights.sum() == 0:
        raise ValueError("Sum of `weights` must be > 0.")
    return weights