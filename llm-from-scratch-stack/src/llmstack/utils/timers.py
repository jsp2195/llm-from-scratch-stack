"""Timing utilities."""

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.3f}s")
