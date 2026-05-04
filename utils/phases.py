"""
Phase completion flag helpers — prevent accidental re-runs of expensive phases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PHASE1_FLAG, PHASE2_FLAG, PHASE3_FLAG, PHASE4_FLAG
from utils.logging import get_logger

logger = get_logger(__name__)

_FLAGS = {
    1: PHASE1_FLAG,
    2: PHASE2_FLAG,
    3: PHASE3_FLAG,
    4: PHASE4_FLAG,
}


def require_phase(n: int):
    """Raise RuntimeError if phase n has not been completed."""
    flag = _FLAGS.get(n)
    if flag is None:
        return
    if not flag.exists():
        raise RuntimeError(
            f"Phase {n} is not marked complete. "
            f"Run the Phase {n} scripts first, or touch {flag} manually to override."
        )


def mark_complete(n: int, note: str = ""):
    """Touch the completion flag for phase n."""
    flag = _FLAGS.get(n)
    if flag is None:
        logger.warning(f"No flag defined for phase {n}")
        return
    flag.write_text(note or f"phase {n} complete", encoding="utf-8")
    logger.info(f"Phase {n} marked complete → {flag}")


def is_complete(n: int) -> bool:
    flag = _FLAGS.get(n)
    return flag is not None and flag.exists()
