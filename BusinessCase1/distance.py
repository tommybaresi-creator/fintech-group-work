import numpy as np
import logging
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def hamming_distance(
    x_cat: np.ndarray,
    y_cat: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    x_cat = np.asarray(x_cat)
    y_cat = np.asarray(y_cat)

    if x_cat.shape != y_cat.shape:
        raise ValueError("`x_cat` and `y_cat` must have the same shape.")

    n_features = x_cat.shape[0]
    weights = _validate_weights(weights, n_features)

    mismatches = (x_cat != y_cat).astype(float)
    return float(np.sum(weights * mismatches) / weights.sum())


def tanimoto_distance(
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    x_bin = np.asarray(x_bin)
    y_bin = np.asarray(y_bin)

    if x_bin.shape != y_bin.shape:
        raise ValueError("`x_bin` and `y_bin` must have the same shape.")
    if not np.all(np.isin(x_bin, [0, 1, False, True])):
        raise ValueError("`x_bin` must contain only binary values (0/1 or False/True).")
    if not np.all(np.isin(y_bin, [0, 1, False, True])):
        raise ValueError("`y_bin` must contain only binary values (0/1 or False/True).")

    x_bin = x_bin.astype(int)
    y_bin = y_bin.astype(int)
    weights = _validate_weights(weights, x_bin.shape[0])

    intersection = np.sum(weights * ((x_bin == 1) & (y_bin == 1)))
    union        = np.sum(weights * ((x_bin == 1) | (y_bin == 1)))

    return 0.0 if union == 0 else float(1.0 - intersection / union)


def hamming_distance_matrix(
    X_cat: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the weighted Hamming distance matrix for categorical variables.
    Uses scipy.spatial.distance.cdist for vectorised computation.

    X_cat must contain the original categorical variables (label-encoded),
    one column per variable, in a fixed order.
    """
    X_cat = np.asarray(X_cat)
    n_samples, n_features = X_cat.shape
    weights = _validate_weights(weights, n_features)

    logger.info("Computing Hamming distance matrix for %d samples, %d features.", n_samples, n_features)

    if weights is None or np.allclose(weights, weights[0]):
        # Unweighted (or uniform weights): use fast scipy cdist
        D = cdist(X_cat, X_cat, metric="hamming")
    else:
        # Weighted: fall back to row-by-row loop
        logger.info("Non-uniform weights detected — using weighted loop for Hamming.")
        D = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = hamming_distance(X_cat[i], X_cat[j], weights=weights)
                D[j, i] = D[i, j]

    logger.info("Hamming distance matrix computed. Shape: %s", D.shape)
    return D


def tanimoto_distance_matrix(
    X_bin: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the weighted Tanimoto/Jaccard distance matrix for one-hot encoded
    categorical variables.
    """
    X_bin = np.asarray(X_bin)
    if not np.all(np.isin(X_bin, [0, 1, False, True])):
        raise ValueError("`X_bin` must contain only binary values (0/1 or False/True).")

    X_bin = X_bin.astype(int)
    n_samples, n_features = X_bin.shape
    weights = _validate_weights(weights, n_features)

    logger.info("Computing Tanimoto distance matrix for %d samples, %d features.", n_samples, n_features)

    D = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            D[i, j] = tanimoto_distance(X_bin[i], X_bin[j], weights=weights)
            D[j, i] = D[i, j]

    logger.info("Tanimoto distance matrix computed. Shape: %s", D.shape)
    return D


def L1_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    return float(np.sum(weights * np.abs(x - y)))


def L2_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    return float(np.sqrt(np.sum(weights * (x - y) ** 2)))


def canberra_distance(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    weights = _validate_weights(weights, x.shape[0])
    num  = np.abs(x - y)
    den  = np.abs(x) + np.abs(y)
    term = np.where(den != 0, num / den, 0.0)
    return float(np.sum(weights * term))


def numerical_distance_matrix(
    X_num: np.ndarray,
    metric: str = "L1",
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute a pairwise distance matrix for numerical variables.
    Uses scipy.spatial.distance.cdist for vectorised computation when
    weights are None or uniform.

    Parameters
    ----------
    X_num   : array-like of shape (n_samples, n_numerical_features)
    metric  : {"L1", "L2", "Canberra"}
    weights : optional feature weights — falls back to loop if provided
    """
    X_num = np.asarray(X_num, dtype=float)
    n_samples, n_features = X_num.shape
    _validate_weights(weights, n_features)  # validate only; scipy handles the rest

    logger.info(
        "Computing %s distance matrix for %d samples, %d features.",
        metric, n_samples, n_features
    )

    scipy_metric_map = {"L1": "cityblock", "L2": "euclidean", "Canberra": "canberra"}

    if metric not in scipy_metric_map:
        raise ValueError("`metric` must be one of: 'L1', 'L2', 'Canberra'.")

    if weights is None or np.allclose(weights, weights[0]):
        # Fast vectorised path
        D = cdist(X_num, X_num, metric=scipy_metric_map[metric])
    else:
        # Weighted: loop fallback
        logger.info("Non-uniform weights detected — using weighted loop for %s.", metric)
        metric_fn = {"L1": L1_distance, "L2": L2_distance, "Canberra": canberra_distance}[metric]
        D = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                D[i, j] = metric_fn(X_num[i], X_num[j], weights)
                D[j, i] = D[i, j]

    logger.info("%s distance matrix computed. Shape: %s", metric, D.shape)
    return D


def mixed_distance(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y_num: np.ndarray,
    y_cat: np.ndarray,
    alpha: float,
    num_dist: str = "L1",
    cat_dist: str = "Hamming",
    num_weights: Optional[np.ndarray] = None,
    cat_weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute a mixed distance between two observations.

    The numerical component is NOT normalised inside this function.
    Ensure X_num is min-max scaled to [0, 1] before calling so that
    alpha weights both components on the same scale.

    Returns alpha * d_num + (1 - alpha) * d_cat.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    num_fn = {"L1": L1_distance, "L2": L2_distance, "Canberra": canberra_distance}
    cat_fn = {"Hamming": hamming_distance, "Tanimoto": tanimoto_distance}

    if num_dist not in num_fn:
        raise ValueError("`num_dist` must be one of: 'L1', 'L2', 'Canberra'.")
    if cat_dist not in cat_fn:
        raise ValueError("`cat_dist` must be one of: 'Hamming', 'Tanimoto'.")

    d_num = num_fn[num_dist](x_num, y_num, num_weights)
    d_cat = cat_fn[cat_dist](x_cat, y_cat, cat_weights)

    return float(alpha * d_num + (1 - alpha) * d_cat)


def _compute_mixed_row(
    i: int,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    alpha: float,
    num_dist: str,
    cat_dist: str,
    num_weights: Optional[np.ndarray],
    cat_weights: Optional[np.ndarray]
) -> list:
    """Compute one row of the mixed distance matrix (used by joblib)."""
    n = len(X_num)
    row = np.zeros(n, dtype=float)
    for j in range(i + 1, n):
        row[j] = mixed_distance(
            X_num[i], X_cat[i], X_num[j], X_cat[j],
            alpha=alpha,
            num_dist=num_dist,
            cat_dist=cat_dist,
            num_weights=num_weights,
            cat_weights=cat_weights
        )
    return row.tolist()


def mixed_matrix_distance(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    alpha: float,
    num_dist: str = "L1",
    cat_dist: str = "Hamming",
    num_weights: Optional[np.ndarray] = None,
    cat_weights: Optional[np.ndarray] = None,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Compute the mixed distance matrix using parallel row computation.

    Parameters
    ----------
    X_num     : array-like (n_samples, n_numerical_features) — must be min-max scaled
    X_cat     : array-like (n_samples, n_categorical_features)
                  label-encoded if cat_dist='Hamming',
                  one-hot encoded if cat_dist='Tanimoto'
    alpha     : weight for numerical component; (1-alpha) for categorical
    num_dist  : {"L1", "L2", "Canberra"}
    cat_dist  : {"Hamming", "Tanimoto"}
    n_jobs    : number of parallel jobs (-1 = all cores)
    """
    X_num = np.asarray(X_num)
    X_cat = np.asarray(X_cat)

    if X_num.shape[0] != X_cat.shape[0]:
        raise ValueError("`X_num` and `X_cat` must have the same number of rows.")

    n_samples = X_num.shape[0]
    logger.info(
        "Computing mixed distance matrix (%s + %s, alpha=%.2f) for %d samples using %d jobs.",
        num_dist, cat_dist, alpha, n_samples, n_jobs
    )

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_compute_mixed_row)(
            i, X_num, X_cat, alpha, num_dist, cat_dist, num_weights, cat_weights
        )
        for i in range(n_samples)
    )

    # Reconstruct symmetric matrix from upper triangle
    D = np.array(rows, dtype=float)
    D = D + D.T  # lower triangle is 0, so adding transpose fills it

    logger.info("Mixed distance matrix computed. Shape: %s", D.shape)
    return D


# ── Helper ────────────────────────────────────────────────────────────────────

def _validate_weights(
    weights: Optional[np.ndarray],
    n_features: int
) -> np.ndarray:
    """Validate and return weights array, defaulting to ones if None."""
    if weights is None:
        return np.ones(n_features, dtype=float)

    weights = np.asarray(weights, dtype=float)

    if weights.shape[0] != n_features:
        raise ValueError(
            f"`weights` must have length {n_features}, "
            f"but got length {weights.shape[0]}."
        )
    if np.any(weights < 0):
        raise ValueError("`weights` must contain non-negative values.")
    if weights.sum() == 0:
        raise ValueError("The sum of `weights` must be greater than 0.")

    return weights