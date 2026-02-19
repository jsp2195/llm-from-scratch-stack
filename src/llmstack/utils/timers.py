from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    st = time.perf_counter()
    yield lambda: time.perf_counter() - st
