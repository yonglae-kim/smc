from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Progress:
    total: int
    label: str = "Progress"
    every: int = 50

    def __post_init__(self):
        self.start = time.time()
        self.last_print = 0

    def _fmt_eta(self, done: int) -> str:
        if done <= 0:
            return "ETA: --"
        elapsed = time.time() - self.start
        rate = done / max(1e-9, elapsed)
        remaining = max(0, self.total - done)
        eta = remaining / max(1e-9, rate)
        m, s = divmod(int(eta), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"ETA: {h}h{m:02d}m"
        return f"ETA: {m}m{s:02d}s"

    def tick(self, done: int, extra: str = ""):
        if self.total <= 0:
            return
        if done == self.total or done - self.last_print >= self.every:
            self.last_print = done
            elapsed = time.time() - self.start
            rate = done / max(1e-9, elapsed)
            pct = (done / self.total) * 100.0
            msg = f"[{self.label}] {done}/{self.total} ({pct:.1f}%) | {rate:.2f}/s | {self._fmt_eta(done)}"
            if extra:
                msg += f" | {extra}"
            print(msg, flush=True)
