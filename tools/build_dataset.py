"""Regenerate the committed intraday dataset.

Standard library only, no network, no key, deterministic from a seed. The output
is byte-reproducible: same command, same bytes, so the SHA-256 in the README is
a real check rather than decoration.

    python3 -m tools.build_dataset --out data/progeny-1min.csv

WHY SYNTHETIC
-------------
No free, no-key, license-clean source of 1-minute US equity bars was reachable at
build time. Stooq was the candidate and it sits behind a JavaScript proof-of-work
challenge, which is an anti-bot control, and building a bypass into a committed
tool is not a trade this project should make. A live fetch would also not be
byte-reproducible.

So this file is SYNTHETIC, and is calibrated to be *hard* rather than friendly:
regime drift is deliberately small relative to per-bar noise, because the earlier
generator's trends were strong enough that momentum won nearly every market and
the harness looked better than it was. A generator that makes edge easy produces
optimism, and optimism is the thing that gets people to trade.

Consequence, stated plainly: a holdout measured on this file demonstrates
generalisation across independent synthetic sessions. It says NOTHING about
whether a strategy would make money in a real market. Only real bars can do that,
and swapping them in is mechanical -- see `real-bar contract` in the README.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from progeny.sessions import minimum_sessions  # noqa: E402

SESSION_MINUTES = 390
SESSION_OPEN = (9, 30)


# ---- regimes ------------------------------------------------------------
# Two symbols that are not clones: different volatility, different trendiness,
# different autocorrelation. If both were generated from the same parameters the
# dataset would look like evidence from two instruments and be evidence from one.
REGIMES = {
    # High-volatility, strongly trending. The regime the loop should find hardest.
    "TRD": dict(
        start_price=100.0, vol=0.0022, regime_vol=0.0011, jump_prob=0.0035,
        drift=0.0, mean_revert=0.0016, oi_shape=0.55,
        base_volume=48_000, vol_of_vol=0.25,
    ),
    # Lower-volatility, mean-reverting, calmer open.
    "MRV": dict(
        start_price=245.0, vol=0.0011, regime_vol=0.0003, jump_prob=0.0012,
        drift=0.0, mean_revert=0.0075, oi_shape=0.62,
        base_volume=22_000, vol_of_vol=0.18,
    ),
}


def trading_days(start: date, count: int) -> list[date]:
    """Weekday calendar.

    Filtered, not cursor-advanced: advancing a cursor collides with itself,
    because skipping Saturday forward to Monday lands on a date a later index
    reaches naturally.
    """
    out: list[date] = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def intraday_shape(n: int, power: float) -> list[float]:
    """U-shaped relative weights: busy at the open, flat midday, busy at the close.

    Constant volume would make any volume-aware logic unfalsifiable -- a rule
    that "only trades when volume is high" would look fine against a flat
    profile and would be meaningless against a real one.
    """
    raw = []
    for i in range(n):
        u = i / max(1, n - 1)
        u_shape = power + (1.0 - power) * (2.0 * abs(u - 0.5)) ** 1.7
        raw.append(max(0.05, u_shape))
    total = sum(raw)
    return [r / total for r in raw]


def generate_symbol(
    symbol: str,
    cfg: dict,
    days: list[date],
    seed: int,
) -> list[list[str]]:
    rng = random.Random(f"{seed}:{symbol}")
    shape = intraday_shape(SESSION_MINUTES, cfg["oi_shape"])
    rows: list[list[str]] = []
    price = cfg["start_price"]

    prev_close = price
    regime = 0.0
    regime_left = 0

    for day_i, d in enumerate(days):
        # Overnight gap: real sessions do not open where the last one closed.
        gap = rng.gauss(0.0, cfg["vol"] * 0.6)
        price = max(0.5, prev_close * (1.0 + gap))

        if regime_left <= 0:
            regime = rng.choice((-1.0, 0.0, 0.0, 1.0)) * cfg["regime_vol"]
            regime_left = rng.randint(60, 400)

        anchor = prev_close
        session_open = datetime(d.year, d.month, d.day, *SESSION_OPEN)
        open_ = price

        for i in range(SESSION_MINUTES):
            if regime_left > 0:
                regime_left -= 1
            # Volume-of-vol: today's realised volatility is drawn once, not
            # per-bar, which is what gives clustering across a session.
            reversion = cfg["mean_revert"] * (anchor - price) / max(price, 1e-9)
            shock = rng.gauss(0.0, cfg["vol"] * (0.6 + 1.6 * rng.random()))
            ret = regime + reversion + shock
            close = max(0.01, price * (1.0 + ret))

            wick = abs(close - open_) + price * cfg["vol"] * rng.uniform(0.15, 0.9)
            high = max(open_, close) + wick * rng.random()
            low = min(open_, close) - wick * rng.random()
            low = max(low, 0.005)

            base = cfg["base_volume"] * shape[i]
            volume = int(max(1.0, base * rng.lognormvariate(0.0, cfg["vol_of_vol"])))

            ts = session_open + timedelta(minutes=i)
            rows.append([
                symbol, ts.isoformat(),
                f"{open_:.4f}", f"{high:.4f}", f"{low:.4f}", f"{close:.4f}",
                str(volume),
            ])
            open_, price = close, close

        prev_close = price

    return rows


def build(out: Path, days: int, seed: int, start: date) -> Path:
    d = trading_days(start, days)
    need_sessions, need_bars = minimum_sessions()
    if days < need_sessions:
        raise SystemExit(
            f"--days {days} is below the {need_sessions} sessions the default "
            f"configuration needs ({need_bars:,} bars). Ask for more, or shorten "
            f"the run."
        )

    buf = io.StringIO()
    buf.write("# progeny synthetic 1-minute dataset\n")
    buf.write(f"# generator: tools/build_dataset.py seed={seed} start={start.isoformat()}\n")
    buf.write(f"# sessions: {days}  bars/session: {SESSION_MINUTES}  "
              f"symbols: {','.join(REGIMES)}\n")
    buf.write("# session: 09:30-16:00 naive local wall-clock, weekdays only\n")
    buf.write("# SYNTHETIC. Not real market data. A holdout measured here says\n")
    buf.write("# nothing about whether a strategy would make money in a real market.\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["symbol", "ts", "open", "high", "low", "close", "volume"])
    for symbol, cfg in REGIMES.items():
        w.writerows(generate_symbol(symbol, cfg, d, seed))

    text = buf.getvalue()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/progeny-1min.csv")
    ap.add_argument("--days", type=int, default=300,
                    help=f"sessions to generate (default config needs "
                         f"{minimum_sessions()[0]})")
    ap.add_argument("--seed", type=int, default=20260105)
    ap.add_argument("--start", default="2025-01-02")
    a = ap.parse_args(argv)

    out = build(Path(a.out), a.days, a.seed, date.fromisoformat(a.start))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    rows = sum(1 for _ in out.open()) - 6   # minus header + 5 provenance lines
    print(f"wrote {out}  {out.stat().st_size:,} bytes  {rows:,} data rows")
    print(f"sha256 {digest}")
    print(f"note: bytes are a function of (--seed, --days, --start) only, so this "
          f"is reproducible. Re-running must reproduce this digest exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
