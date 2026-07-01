import logging
import sys
from pathlib import Path


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_timing(start_time, completed, total, now=None):
    import time

    now = time.time() if now is None else now
    elapsed = max(0.0, now - start_time)
    avg = elapsed / completed if completed > 0 else 0.0
    remaining = max(0, total - completed)
    eta = avg * remaining if completed > 0 else 0.0
    return elapsed, avg, eta


def setup_logger(name, log_path=None, rank=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    rank_text = "" if rank is None else f" rank={rank}"
    formatter = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | {name}{rank_text} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
