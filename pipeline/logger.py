# pipeline/logger.py

import logging
import os
import sys
import time
from datetime import datetime
from contextlib import contextmanager


# ============================================================
# Logger Initialization
# ============================================================

def init_logger(action: str):
    """
    Creates a logger that writes INFO-level logs to:
        logs/<timestamp>_<action>.log
    and also streams to stdout.

    Returns the configured logger.
    """
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{timestamp}_{action}.log")

    logger = logging.getLogger(action)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset handlers to avoid duplication

    # ---- File handler ----
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # ---- Stream handler (stdout) ----
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    # ---- Formatting ----
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger initialized → {log_path}")

    return logger


# ============================================================
# Timing helper (integer-second resolution)
# ============================================================

@contextmanager
def log_timing(logger, label: str):
    """
    Context manager:
        with log_timing(logger, "TRAIN[abp]"):
            ...
    Outputs:
        Starting TRAIN[abp]
        Finished TRAIN[abp] — 8s
    """
    start = time.time()
    logger.info(f"Starting {label}")
    try:
        yield
    finally:
        elapsed = int(time.time() - start)
        logger.info(f"Finished {label} — {elapsed}s")
