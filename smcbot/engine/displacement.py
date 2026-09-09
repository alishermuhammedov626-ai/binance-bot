"""Section 12 -- displacement scoring.

Explicitly *not* a hard requirement: a weak displacement only removes bonus
points, it never vetoes a setup that is strong everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import DisplacementConfig
from ..core.series import CandleSeries


@dataclass
class Displacement:
    bullish: bool
    body_atr: float
    consecutive: int
    volume_ratio: float
    bonus: float
    label: str          # STRONG | MEDIUM | WEAK

    @property
    def strong(self) -> bool:
        return self.label == "STRONG"


class DisplacementEngine:
    def __init__(self, cfg: DisplacementConfig):
        self.cfg = cfg

    def measure(self, series: CandleSeries, bullish: bool,
                lookback: int = 0) -> Displacement:
        n = lookback or self.cfg.lookback
        window = series.window(n)
        atr = series.atr or 1e-9
        if not window:
            return Displacement(bullish, 0.0, 0, 0.0, 0.0, "WEAK")

        # Net directional travel across the impulse window, in ATR.
        directional = [c for c in window if (c.bullish if bullish else c.bearish)]
        net = ((window[-1].close - window[0].open) if bullish
               else (window[0].open - window[-1].close))
        body_atr = max(net, 0.0) / atr

        consecutive = 0
        for c in reversed(window):
            if (c.bullish if bullish else c.bearish):
                consecutive += 1
            else:
                break

        vol_ma = series.vol_ma() or 1e-9
        vol_ratio = (sum(c.volume for c in directional) / len(directional) / vol_ma
                     if directional else 0.0)

        if body_atr >= self.cfg.strong_body_atr and vol_ratio >= self.cfg.volume_boost * 0.7:
            label, bonus = "STRONG", self.cfg.strong_bonus
        elif body_atr >= self.cfg.strong_body_atr:
            label, bonus = "STRONG", self.cfg.strong_bonus * 0.8
        elif body_atr >= self.cfg.medium_body_atr:
            label, bonus = "MEDIUM", self.cfg.medium_bonus
        else:
            label, bonus = "WEAK", 0.0
        return Displacement(bullish, round(body_atr, 3), consecutive,
                            round(vol_ratio, 3), bonus, label)
