"""
Structured logging for the Neuralese pipeline.

All phases call get_logger(__name__) to get a consistent logger.
Logs go to both stdout and outputs/pipeline.log.
"""

import logging
import sys
from pathlib import Path

_LOG_PATH: Path | None = None
_INITIALISED = False


def _init_logging():
    global _INITIALISED, _LOG_PATH
    if _INITIALISED:
        return

    root = Path(__file__).parent.parent
    log_dir = root / "outputs"
    log_dir.mkdir(exist_ok=True)
    _LOG_PATH = log_dir / "pipeline.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("neuralese")
    root_logger.setLevel(logging.DEBUG)

    if not root_logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root_logger.addHandler(sh)

        fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    _INITIALISED = True


def get_logger(name: str) -> logging.Logger:
    _init_logging()
    if not name.startswith("neuralese"):
        name = f"neuralese.{name.split('.')[-1]}"
    return logging.getLogger(name)
