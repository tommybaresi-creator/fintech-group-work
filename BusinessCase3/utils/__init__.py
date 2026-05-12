"""
Portfolio Replication — utility package.

Import `setup_logging` once at the top of every entry point
(notebook or script) to configure the root logger.
"""

import contextlib
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a stdout handler.

    Parameters
    ----------
    level : int
        Logging level (e.g. logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-25s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


from utils.data_loader import run_data_loader
from utils.evaluation import run_evaluation
from utils.run_nn import run_nn
from utils.transaction_costs import run_transaction_cost_analysis


@contextlib.contextmanager
def inline_figures():
    """
    Context manager for Jupyter inline display of utils plot functions.

    utils plot functions save figures to disk and then call ``plt.close(fig)``
    to free memory.  Inside this context, ``plt.close`` is suppressed so that
    Jupyter's inline backend can render each figure at the end of the cell.

    Usage::

        from utils import inline_figures
        with inline_figures():
            plot_cumulative_returns(all_results)
    """
    import matplotlib.pyplot as plt
    _orig = plt.close
    plt.close = lambda *a, **kw: None
    try:
        yield
    finally:
        plt.close = _orig


__all__ = [
    "setup_logging",
    "inline_figures",
    "run_data_loader",
    "run_evaluation",
    "run_nn",
    "run_transaction_cost_analysis",
]
