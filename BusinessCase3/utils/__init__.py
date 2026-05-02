"""
Portfolio Replication — utility package.

Import `setup_logging` once at the top of every entry point
(notebook or script) to configure the root logger.
"""

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

__all__ = [
    "setup_logging",
    "run_data_loader",
    "run_evaluation",
    "run_nn",
]