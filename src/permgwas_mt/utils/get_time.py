import time
import torch
from collections import defaultdict
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.sections = defaultdict(float)

    def elapsed(self):
        return time.perf_counter() - self.t0

    def add(self, name, dt):
        self.sections[name] += dt

    def log(self, message):
        print(f"{message} Passed time: {self.elapsed():.3f} s")

    def report(self):
        total = self.elapsed()
        print("\nTiming report:")
        for name, t in sorted(self.sections.items(), key=lambda x: -x[1]):
            print(f"{name:25s}: {t:10.3f} s")
        print(f"{'TOTAL':25s}: {total:10.3f} s")


@contextmanager
def timed(timer, name, use_cuda=False):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield
    end = time.perf_counter()

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    timer.add(name, end - start)