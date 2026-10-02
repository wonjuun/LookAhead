"""Learning-rate schedule helpers."""

from __future__ import annotations

import math


def bounded_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    requested = math.ceil(warmup_ratio * total_steps)
    return min(requested, total_steps - 1)
