"""Section 61 -- Monte Carlo robustness.

A single equity curve is one sample from a distribution.  Reshuffling the trade
sequence (and optionally resampling it with replacement) shows how much of the
result was ordering luck, and what drawdown the same edge could plausibly have
produced.  ``risk_of_ruin`` is the share of paths that lose ``ruin_threshold``
of the account.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..core.types import Trade


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class MonteCarloResult:
    runs: int
    ruin_probability: float
    ruin_threshold: float
    final_equity: Dict[str, float]
    max_drawdown: Dict[str, float]
    worst_losing_streak: Dict[str, float]
    profitable_share: float

    def summary(self) -> str:
        fe, dd, st = self.final_equity, self.max_drawdown, self.worst_losing_streak
        return "\n".join([
            f"Monte Carlo runs   : {self.runs}",
            f"Profitable paths   : {self.profitable_share:.1%}",
            f"Final equity  p05/p50/p95 : {fe['p05']:.2f} / {fe['p50']:.2f} / {fe['p95']:.2f}",
            f"Max drawdown  p50/p95/max : {dd['p50']:.2%} / {dd['p95']:.2%} / {dd['max']:.2%}",
            f"Losing streak p50/p95/max : {st['p50']:.0f} / {st['p95']:.0f} / {st['max']:.0f}",
            f"Risk of ruin (-{self.ruin_threshold:.0%}) : {self.ruin_probability:.2%}",
        ])


def monte_carlo(trades: Sequence[Trade], initial_equity: float, runs: int = 2000,
                seed: int = 11, resample: bool = True,
                ruin_threshold: float = 0.5) -> MonteCarloResult:
    """Resample the *percentage* returns so compounding is respected."""
    if not trades:
        return MonteCarloResult(0, 0.0, ruin_threshold, {}, {}, {}, 0.0)

    # Convert each trade to the fraction of equity it made or lost, so a path
    # can be replayed at any account size.
    fractions: List[float] = []
    equity = initial_equity
    for t in trades:
        base = t.equity_after - t.pnl if t.equity_after else equity
        base = base if base > 0 else initial_equity
        fractions.append(t.pnl / base)
        equity = t.equity_after or (equity + t.pnl)

    rng = random.Random(seed)
    finals: List[float] = []
    dds: List[float] = []
    streaks: List[float] = []
    ruins = 0
    n = len(fractions)

    for _ in range(runs):
        path = ([rng.choice(fractions) for _ in range(n)] if resample
                else rng.sample(fractions, n))
        eq = initial_equity
        peak = eq
        max_dd = 0.0
        streak = worst_streak = 0
        ruined = False
        for f in path:
            eq *= (1.0 + f)
            if eq <= 0:
                eq = 0.0
                ruined = True
                break
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)
            if f < 0:
                streak += 1
                worst_streak = max(worst_streak, streak)
            else:
                streak = 0
        if ruined or eq <= initial_equity * (1 - ruin_threshold):
            ruins += 1
        finals.append(eq)
        dds.append(max_dd)
        streaks.append(worst_streak)

    def stats(xs: List[float]) -> Dict[str, float]:
        return {"p05": _percentile(xs, 0.05), "p50": _percentile(xs, 0.50),
                "p95": _percentile(xs, 0.95), "max": max(xs), "min": min(xs)}

    return MonteCarloResult(
        runs=runs,
        ruin_probability=round(ruins / runs, 4),
        ruin_threshold=ruin_threshold,
        final_equity=stats(finals),
        max_drawdown=stats(dds),
        worst_losing_streak=stats(streaks),
        profitable_share=round(sum(1 for f in finals if f > initial_equity) / runs, 4),
    )
