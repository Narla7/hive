"""Price data for the harness.

Two sources, one interface:
  - `SimulatedMarket`: deterministic synthetic intraday prices. No API key, no
    network, byte-identical across runs given a seed. This is the default and it
    is what makes fitness comparable between agents.
  - `FileMarket`: CSV of real bars, for when you have exported data.

Both hand back `Bar` records in ascending time order. Neither knows anything
about orders; the broker owns execution.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class Market:
    """Common interface: get bars for a symbol over a window."""

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        raise NotImplementedError


class SimulatedMarket(Market):
    """Synthetic intraday prices via seeded random walk with regime shifts.

    The regime shifts matter: a market that is pure noise has no edge to find,
    so a fitness-increasing experiment would be measuring nothing. Trends,
    mean-reverting stretches and occasional jumps give the search space
    something real to discover.
    """

    def __init__(
        self,
        seed: int = 7,
        start_price: float = 100.0,
        volatility: float = 0.0015,
        drift: float = 0.0,
        regime_prob: float = 0.02,
        jump_prob: float = 0.004,
        spread_bps: float = 2.0,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift
        self.regime_prob = regime_prob
        self.jump_prob = jump_prob
        self.spread_bps = spread_bps
        self._cache: dict[tuple[str, datetime, datetime], list[Bar]] = {}

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        key = (symbol, start, end)
        if key not in self._cache:
            self._cache[key] = list(self._generate(symbol, start, end))
        return self._cache[key]

    def _generate(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        rng = random.Random(f"{self.seed}:{symbol}:{start.isoformat()}")
        price = self.start_price
        # Per-symbol bias so multiple symbols are not clones of one another.
        symbol_drift = self.drift + rng.uniform(-0.0004, 0.0004)
        regime = rng.choice((-1, 0, 1))
        regime_left = 0
        out: list[Bar] = []

        ts = start
        while ts < end:
            if regime_left <= 0 and rng.random() < self.regime_prob:
                regime = rng.choice((-1, 0, 1))
                regime_left = rng.randint(20, 90)
            if regime_left > 0:
                regime_left -= 1

            # Geometric random walk, plus the current regime's drift, plus
            # mean reversion pulling price back toward the session's anchor.
            anchor = self.start_price * (1 + 0.04 * math.sin(ts.timestamp() / 8.6e6))
            reversion = 0.004 * (anchor - price) / max(price, 1e-9)
            shock = rng.gauss(0.0, self.volatility)
            if rng.random() < self.jump_prob:
                shock += rng.choice((-1, 1)) * self.volatility * 8

            ret = symbol_drift + regime * 0.0006 + reversion + shock
            open_ = price
            close = max(0.01, price * (1.0 + ret))

            # Intrabar extremes must contain both open and close, and the
            # spread has to widen a little on big moves or the simulated
            # broker is quietly giving away free liquidity.
            wick = abs(close - open_) + price * self.volatility * rng.uniform(0.2, 1.0)
            high = max(open_, close) + wick * rng.random()
            low = min(open_, close) - wick * rng.random()
            low = max(low, 0.005)

            volume = int(rng.lognormvariate(9.0, 0.6))
            out.append(Bar(ts, open_, high, low, close, volume))
            price = close
            ts += timedelta(minutes=1)

        return out


class FileMarket(Market):
    """Real bars from CSV: ts,open,high,low,close,volume (ISO timestamps)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._rows: dict[str, list[Bar]] | None = None

    def _load(self) -> dict[str, list[Bar]]:
        if self._rows is not None:
            return self._rows
        rows: dict[str, list[Bar]] = {}
        with self.path.open(newline="") as fh:
            for rec in csv.DictReader(fh):
                sym = rec["symbol"]
                rows.setdefault(sym, []).append(
                    Bar(
                        ts=datetime.fromisoformat(rec["ts"]),
                        open=float(rec["open"]),
                        high=float(rec["high"]),
                        low=float(rec["low"]),
                        close=float(rec["close"]),
                        volume=int(rec.get("volume") or 0),
                    )
                )
        for bars in rows.values():
            bars.sort(key=lambda b: b.ts)
        self._rows = rows
        return rows

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        return [b for b in self._load().get(symbol, []) if start <= b.ts < end]
