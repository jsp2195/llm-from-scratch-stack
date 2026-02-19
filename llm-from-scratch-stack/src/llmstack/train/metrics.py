"""Training metrics."""

import time


class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, x: float):
        self.total += x
        self.count += 1

    @property
    def value(self):
        return self.total / max(1, self.count)


class TokensPerSecond:
    def __init__(self):
        self.start = time.time()

    def compute(self, tokens: int) -> float:
        dt = max(1e-6, time.time() - self.start)
        return tokens / dt
