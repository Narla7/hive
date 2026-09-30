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

# One US regular session is 390 minutes, 09:30-16:00. Windows are addressed in
# session space so an episode is exactly one trading day.
SESSION_MINUTES = 390


@dataclass(frozen=True, slots=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class WindowUnavailable(RuntimeError):
    """A requested window cannot be served from the data that exists.

    Raised instead of returning a short window. A short window does not fail
    loudly: the episode just runs fewer bars, takes fewer trades, and produces a
    different fitness, which looks exactly like a strategy that stopped working.
    That is the silent-failure class this whole dataset exists to prevent.
    """


class Market:
    """Common interface: get bars for a symbol over a window.

    Sessions are derived from the data itself wherever possible, so windows can
    be addressed in session space instead of wall-clock minutes. That makes gaps
    unrepresentable rather than something to validate away afterwards.
    """

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        raise NotImplementedError

    def sessions(self, symbol: str) -> list[tuple[datetime, datetime]]:
        """Contiguous (start, end) blocks of bars, in order.

        Derived by gap detection: a jump larger than `gap_tolerance` between
        consecutive bars starts a new session. No hardcoded exchange hours, so a
        file with pre-market rows or a half-day still splits correctly.
        """
        raise NotImplementedError

    def span(self, symbol: str) -> tuple[datetime, datetime] | None:
        """First and last bar timestamps, or None if the symbol is absent."""
        rows = self.bars(symbol, datetime.min, datetime.max)
        if not rows:
            return None
        return rows[0].ts, rows[-1].ts

    def n_sessions(self, symbol: str) -> int:
        return len(self.sessions(symbol))


def split_sessions(bars: list[Bar], gap_tolerance: int = 180) -> list[tuple[datetime, datetime]]:
    """Group bars into sessions by gap detection.

    The tolerance is 180s, not 90s, and the difference matters: dropping one
    minute from a series leaves a *two-minute* jump between the neighbours, so a
    90s tolerance splits a session that is merely missing a minute in two. 180s
    absorbs that while any real boundary -- an overnight gap of 17 hours or more
    -- still splits correctly.
    """
    if not bars:
        return []
    out: list[tuple[datetime, datetime]] = []
    start = prev = bars[0].ts
    for b in bars[1:]:
        if (b.ts - prev).total_seconds() > gap_tolerance:
            out.append((start, prev + timedelta(minutes=1)))
            start = b.ts
        prev = b.ts
    out.append((start, prev + timedelta(minutes=1)))
    return out


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
        n_days: int = 320,
        start: datetime = datetime(2026, 1, 5),
        symbol_seed: str = "",
    ) -> None:
        self.seed = seed
        self.n_days = n_days
        self.symbol_seed = symbol_seed
        self.start = start
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

    def sessions(self, symbol: str) -> list[tuple[datetime, datetime]]:
        return [(self._session_start(g), self._session_start(g) + timedelta(minutes=SESSION_MINUTES))
                for g in range(self.n_days)]

    def _session_start(self, day: int) -> datetime:
        """09:30 on the given trading day, skipping weekends."""
        d = self.start + timedelta(days=day)
        while d.weekday() >= 5:      # Sat=5, Sun=6
            d += timedelta(days=1)
        return datetime(d.year, d.month, d.day, 9, 30)

    def _generate(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        rng = random.Random(f"{self.seed}:{self.symbol_seed or symbol}:{start.isoformat()}")
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
    """Real bars from CSV: symbol,ts,open,high,low,close,volume.

    Lines beginning with `#` are treated as a provenance header and skipped, so
    the committed dataset can document itself without breaking the reader.

    OHLC integrity is enforced at load rather than trusted. A bar whose low
    exceeds its open, or whose high is below its close, means the file is wrong
    or misaligned, and silently accepting it corrupts every stop and target
    downstream.
    """

    def __init__(self, path: str | Path, strict: bool = True) -> None:
        self.path = Path(path)
        self.strict = strict
        self.provenance: list[str] = []
        self._rows: dict[str, list[Bar]] | None = None

    def _load(self) -> dict[str, list[Bar]]:
        if self._rows is not None:
            return self._rows
        rows: dict[str, list[Bar]] = {}
        with self.path.open(newline="") as fh:
            reader = csv.DictReader(l for l in fh if not l.startswith("#"))
            for rec in reader:
                sym = rec["symbol"]
                o, h, l, c = (float(rec["open"]), float(rec["high"]),
                              float(rec["low"]), float(rec["close"]))
                vol = int(rec.get("volume") or 0)
                ts = datetime.fromisoformat(rec["ts"])
                if min(o, c) > h or max(o, c) < l:
                    msg = f"{self.path}: OHLC inconsistent at {sym} {rec['ts']}: {o},{h},{l},{c}"
                    if self.strict:
                        raise ValueError(msg)
                if min(o, h, l, c) <= 0:
                    raise ValueError(f"{self.path}: non-positive price at {sym} {rec['ts']}")
                if vol < 0:
                    raise ValueError(f"{self.path}: negative volume at {sym} {rec['ts']}")
                rows.setdefault(sym, []).append(Bar(ts, o, h, l, c, vol))
        for sym, bars in rows.items():
            bars.sort(key=lambda b: b.ts)
            for a, b in zip(bars, bars[1:]):
                if a.ts == b.ts:
                    raise ValueError(f"{self.path}: duplicate timestamp {sym} {a.ts.isoformat()}")
        self._rows = rows
        return rows

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        return [b for b in self._load().get(symbol, []) if start <= b.ts < end]

    def sessions(self, symbol: str) -> list[tuple[datetime, datetime]]:
        return split_sessions(self._load().get(symbol, []))

    @property
    def symbols(self) -> list[str]:
        return sorted(self._load().keys())
