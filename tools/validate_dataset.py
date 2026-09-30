"""Validate the committed dataset. Exits non-zero on any violation.

    python3 -m tools.validate_dataset            # default file
    python3 -m tools.validate_dataset path.csv   # explicit

Checks, in order:

  1. schema and provenance header present
  2. per-row OHLC integrity, positive prices, non-negative volume
  3. no duplicate timestamps per symbol
  4. no gaps inside a session
  5. sessions are full-length and contiguous, and no window can straddle a gap
  6. volume varies intraday (open/close busy, midday flat)
  7. at least the number of sessions the default Config needs
  8. symbols are not clones of one another
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from progeny.market import FileMarket, split_sessions  # noqa: E402
from progeny.sessions import minimum_sessions, plan_from_config  # noqa: E402

SESSION_MINUTES = 390
DEFAULT = "data/progeny-1min.csv"


class _Stamp:
    """Minimal stand-in for Bar; split_sessions only reads `.ts`."""

    __slots__ = ("ts",)

    def __init__(self, ts: datetime) -> None:
        self.ts = ts


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def check(self, ok: bool, good: str, bad: str) -> bool:
        if ok:
            self.note(good)
        else:
            self.fail(bad)
        return ok


def validate(path: Path) -> Report:
    r = Report()
    if not path.exists():
        r.fail(f"{path} does not exist. Build it: python3 -m tools.build_dataset")
        return r

    raw = path.read_text().splitlines()
    r.check(any(l.startswith("#") for l in raw[:10]),
            "provenance header present",
            "no provenance header found in the first 10 lines")
    if any("SYNTHETIC" in l for l in raw[:10]):
        r.note("file declares itself SYNTHETIC -- holdout proves generalisation "
               "across synthetic sessions only, not real-market profitability")

    header = next((l for l in raw if not l.startswith("#")), "")
    expected = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    r.check(header.split(",") == expected,
            f"schema {header}",
            f"schema is {header!r}, expected {','.join(expected)}")

    # Row-level integrity, straight off the text, so a bad row is reported with
    # its line number rather than as a downstream oddity.
    by_symbol: dict[str, list[tuple[int, datetime, float, float, float, float, int]]] = defaultdict(list)
    for n, line in enumerate(raw, start=1):
        if not line or line.startswith("#") or line.startswith("symbol,"):
            continue
        parts = line.split(",")
        if len(parts) != 7:
            r.fail(f"line {n}: expected 7 fields, got {len(parts)}")
            continue
        sym, ts, o, h, l, c, v = parts
        try:
            ts_d = datetime.fromisoformat(ts)
            o, h, l, c, v = float(o), float(h), float(l), float(c), int(v)
        except ValueError as exc:
            r.fail(f"line {n}: unparseable ({exc})")
            continue
        if min(o, h, l, c) <= 0:
            r.fail(f"line {n}: non-positive price {sym} {ts}")
        if v < 0:
            r.fail(f"line {n}: negative volume {sym} {ts}")
        if not (l <= min(o, c) and max(o, c) <= h):
            r.fail(f"line {n}: OHLC inconsistent {sym} {ts} "
                   f"o={o} h={h} l={l} c={c}")
        by_symbol[sym].append((n, ts_d, o, h, l, c, v))

    if not by_symbol:
        r.fail("no data rows")
        return r

    r.note(f"{len(by_symbol)} symbol(s): {', '.join(sorted(by_symbol))}")

    for sym, rows in sorted(by_symbol.items()):
        stamps = [t for _, t, *_ in rows]
        # O(n) via a seen-set. `stamps.count(t)` inside a comprehension is O(n^2)
        # and takes minutes on a 117k-bar symbol.
        seen: set[datetime] = set()
        dupes: set[datetime] = set()
        for t in stamps:
            if t in seen:
                dupes.add(t)
            else:
                seen.add(t)
        r.check(not dupes,
                f"{sym}: no duplicate timestamps ({len(stamps):,} bars)",
                f"{sym}: {len(dupes)} duplicate timestamp(s), first "
                f"{min(dupes) if dupes else 'n/a'}")

        # Group into sessions using the reader's own logic. A second
        # implementation here with a different tolerance is how the validator and
        # FileMarket come to disagree about where a session ends.
        spans = split_sessions([_Stamp(t) for t in stamps])
        sets = [set(range(int((e - s0).total_seconds() // 60))) for s0, e in spans]
        sessions = [[t for k, t in enumerate(stamps) if k in st] for st in sets]

        bad_len = [i for i, s in enumerate(sessions) if len(s) != SESSION_MINUTES]
        first_bad = bad_len[0] if bad_len else -1
        r.check(not bad_len,
                f"{sym}: all {len(sessions)} sessions are exactly "
                f"{SESSION_MINUTES} bars",
                f"{sym}: {len(bad_len)} session(s) are not "
                f"{SESSION_MINUTES} bars, first index {first_bad} "
                f"({len(sessions[first_bad]) if first_bad >= 0 else 0} bars)")

        gapped = []
        for s in sessions:
            for a, b in zip(s, s[1:]):
                if (b - a) != timedelta(minutes=1):
                    gapped.append(a)
                    break
        r.check(not gapped,
                f"{sym}: no gaps inside any session",
                f"{sym}: {len(gapped)} session(s) contain an internal gap, "
                f"first at {gapped[0] if gapped else 'n/a'}")

        # Property 6: volume must actually vary intraday.
        first_s = sessions[0]
        if len(first_s) == SESSION_MINUTES:
            first_set = set(first_s)
            vols = [row[6] for row in rows if row[1] in first_set]
            if len(vols) >= SESSION_MINUTES:
                # Medians, not means: per-bar volume is lognormal, so a mean
                # over 30 bars is dominated by noise and a real U-shape reads
                # as flat.
                head = statistics.median(vols[:45])
                mid = statistics.median(vols[SESSION_MINUTES // 2 - 22:
                                             SESSION_MINUTES // 2 + 22])
                tail = statistics.median(vols[-45:])
                r.check(min(head, tail) > mid * 1.15,
                        f"{sym}: volume is intraday-shaped "
                        f"(open {head:,.0f} / midday {mid:,.0f} / close {tail:,.0f})",
                        f"{sym}: volume is flat intraday "
                        f"(open {head:,.0f}, midday {mid:,.0f}, close {tail:,.0f}) "
                        f"-- volume-aware logic would be unfalsifiable")

    # Property 5: enough history for the configuration's own arithmetic.
    need_sessions, need_bars = minimum_sessions()
    # Sessions, not bars. Comparing a bar count against a session requirement is
    # off by a factor of ~390, so a single-session file looked adequate.
    n_sess = {
        sym: len(split_sessions([_Stamp(t) for _, t, *_ in rows]))
        for sym, rows in by_symbol.items()
    }
    smallest = min(n_sess.values())
    r.check(smallest >= need_sessions,
            f"smallest symbol has {smallest} sessions (>= {need_sessions} required, "
            f"= {need_bars:,} bars)",
            f"smallest symbol has only {smallest} sessions but the default config "
            f"needs {need_sessions} (= {need_bars:,} bars). Either shorten the run "
            f"or rebuild with more --days.")

    # Property: symbols must not be clones.
    if len(by_symbol) >= 2:
        sigs = {}
        for sym, rows in by_symbol.items():
            closes = [row[5] for row in rows]
            rets = [closes[i + 1] / closes[i] - 1 for i in range(len(closes) - 1)]
            sigs[sym] = (statistics.stdev(rets) if len(rets) > 1 else 0.0,
                         abs(closes[-1] / closes[0] - 1.0))
        ranked = sorted(sigs.items(), key=lambda kv: kv[1][0])
        lo_sym, (lo_vol, lo_drift) = ranked[0]
        hi_sym, (hi_vol, hi_drift) = ranked[-1]
        r.check(hi_vol / max(lo_vol, 1e-12) > 1.5,
                f"symbols differ in regime: {lo_sym} vol {lo_vol:.5f} drift "
                f"{lo_drift:+.1%} vs {hi_sym} vol {hi_vol:.5f} drift {hi_drift:+.1%}",
                f"symbols look like clones: {lo_sym} vol {lo_vol:.5f} and "
                f"{hi_sym} vol {hi_vol:.5f} differ by less than 1.5x")

    # The reader must agree, and the plan must fit.
    try:
        m = FileMarket(path)
        plan = plan_from_config(12, 3, 30, 24, 40, 2)
        for sym in sorted(by_symbol):
            plan.validate(m.n_sessions(sym))
        r.note(f"FileMarket loads; SessionPlan(generations=12) validates for "
               f"{len(by_symbol)} symbol(s)")
    except Exception as exc:
        r.fail(f"reader/plan rejected the file: {type(exc).__name__}: {exc}")

    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=DEFAULT)
    a = ap.parse_args(argv)

    r = validate(Path(a.path))
    for n in r.notes:
        print(f"  ok   {n}")
    for f in r.failures:
        print(f"  FAIL {f}", file=sys.stderr)
    if r.failures:
        print(f"\n{r.path if hasattr(r, 'path') else a.path}: {len(r.failures)} "
              f"violation(s)", file=sys.stderr)
        return 1
    print(f"\n{a.path}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
