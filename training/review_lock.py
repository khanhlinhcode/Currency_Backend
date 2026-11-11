"""Cross-process lock for the local review CSV (macOS/Linux)."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def review_file_lock(review_csv):
    lock_path = Path(str(review_csv) + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
