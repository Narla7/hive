"""Dataset and session-planning tests.

These cover the properties the dataset exists to guarantee. The theme throughout:
a silent failure is worse than a loud one. A window that quietly returns fewer
bars produces a plausible fitness number derived from nothing, and nothing in
the output looks wrong.
"""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from progeny.decisioners import RulesDecisioner
from progeny.evolution import (
    Config,
    buy_and_hold,
    holdout,
    run,
    windows_for_gen,
)
from progeny.market import FileMarket, WindowUnavailable, split_sessions
from progeny.sessions import (
    SessionPlanError,
    minimum_sessions,
    plan_from_config,
)
from tools.validate_dataset import validate

TS = datetime(2026, 1, 5, 9, 30)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "progeny-1min.csv"


def _skip_without_data():
    return unittest.skipUnless(DATA.exists(), f"{DATA} not built; run tools.build_dataset")


@_skip_without_data()
class TestShippedDataset(unittest.TestCase):
    def test_file_present_and_substantial(self):
        self.assertTrue(DATA.exists())
        self.assertGreater(DATA.stat().st_size, 1_000_000)

    def test_validates_clean(self):
        r = validate(DATA)
        self.assertEqual(r.failures, [], "dataset violations: " + "; ".join(r.failures))

    def test_declares_itself_synthetic(self):
        # The header must not let a reader mistake these for real bars.
        head = "\n".join(DATA.read_text().splitlines()[:10])
        self.assertIn("SYNTHETIC", head)

    def test_provenance_header_present(self):
        self.assertTrue(DATA.read_text().startswith("#"))

    def test_two_or_more_symbols_from_different_regimes(self):
        m = FileMarket(DATA)
        self.assertGreaterEqual(len(m.symbols), 2)
        # Different regimes, not two clones: realised vol must differ materially.
        import statistics
        vols = {}
        for sym in m.symbols:
            closes = [b.close for b in m.bars(sym, datetime.min, datetime.max)]
            rets = [closes[i + 1] / closes[i] - 1 for i in range(len(closes) - 1)]
            vols[sym] = statistics.stdev(rets)
        self.assertGreater(max(vols.values()) / min(vols.values()), 1.5)

    def test_digest_is_stable_for_the_recorded_seed(self):
        # Documented in the README. If this changes, the data changed.
        head = DATA.read_text().splitlines()[:4]
        seed_line = next(l for l in head if "seed=" in l)
        seed = int(seed_line.split("seed=")[1].split()[0])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "again.csv"
            subprocess.run(
                [sys.executable, "-m", "tools.build_dataset", "--out", str(out),
                 "--days", "300", "--seed", str(seed)],
                cwd=ROOT, check=True, capture_output=True,
            )
            self.assertEqual(
                hashlib.sha256(out.read_bytes()).hexdigest(),
                hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "regenerating with the recorded seed produced different bytes",
            )


@_skip_without_data()
class TestValidatorCatchesViolations(unittest.TestCase):
    """A validator that only ever passes is decoration."""

    HEADER = "# test\nsymbol,ts,open,high,low,close,volume\n"

    def _write(self, tmp: str, body: str) -> Path:
        p = Path(tmp) / "bad.csv"
        p.write_text(self.HEADER + body)
        return p

    def _row(self, ts, o=10.0, h=10.2, l=9.8, c=10.1, v=1000):
        return f"SPY,{ts},{o},{h},{l},{c},{v}\n"

    def test_passes_a_good_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = ""
            day = datetime(2026, 1, 5)
            for d in range(3):
                for i in range(390):
                    rows += self._row((day + timedelta(days=d)
                                       + timedelta(minutes=i)).isoformat())
            r = validate(self._write(tmp, rows))
            # Still short of the default span, so only check what must not fail.
            self.assertFalse(any("OHLC" in f for f in r.failures))

    def test_catches_ohlc_inconsistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = validate(self._write(tmp, self._row("2026-01-05T09:30:00", o=10.0,
                                                    h=9.0, l=9.5, c=9.9)))
            self.assertTrue(any("OHLC inconsistent" in f for f in r.failures))

    def test_catches_duplicate_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = "2026-01-05T09:30:00"
            r = validate(self._write(tmp, self._row(ts) + self._row(ts)))
            self.assertTrue(any("duplicate timestamp" in f for f in r.failures))

    def test_catches_non_positive_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = validate(self._write(tmp, self._row("2026-01-05T09:30:00", o=0.0,
                                                    h=0.2, l=-0.1, c=0.1)))
            self.assertTrue(any("non-positive" in f for f in r.failures))

    def test_catches_internal_gap(self):
        # A full session with one minute missing. Two rows an hour apart is not
        # this test: that is two short sessions, not a gapped one.
        with tempfile.TemporaryDirectory() as tmp:
            rows = "".join(
                self._row((datetime(2026, 1, 5, 9, 30) + timedelta(minutes=i)).isoformat())
                for i in range(390) if i != 200
            )
            r = validate(self._write(tmp, rows))
            self.assertTrue(any("internal gap" in f for f in r.failures),
                            f"expected an internal-gap failure, got {r.failures}")

    def test_catches_short_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = "".join(self._row(f"2026-01-05T09:{30+i:02d}:00")
                           for i in range(10))
            r = validate(self._write(tmp, rows))
            self.assertTrue(any("not 390 bars" in f for f in r.failures))

    def test_catches_short_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = "".join(
                self._row((datetime(2026, 1, 5, 9, 30) + timedelta(minutes=i)).isoformat())
                for i in range(390)
            )
            r = validate(self._write(tmp, rows))
            self.assertTrue(any("sessions but the default config needs" in f
                                for f in r.failures),
                            f"expected a short-dataset failure, got {r.failures}")


class TestSplitSessions(unittest.TestCase):
    def _bars(self, n):
        return [type("B", (), {"ts": TS + timedelta(minutes=i)})()
                for i in range(n)]

    def test_empty(self):
        self.assertEqual(split_sessions([]), [])

    def test_contiguous_is_one_session(self):
        self.assertEqual(len(split_sessions(self._bars(390))), 1)

    def test_gap_splits(self):
        from progeny.market import Bar
        bars = [Bar(TS + timedelta(minutes=i), 1, 1, 1, 1, 1) for i in range(390)]
        bars += [Bar(TS + timedelta(days=1, minutes=i), 1, 1, 1, 1, 1)
                 for i in range(390)]
        self.assertEqual(len(split_sessions(bars)), 2)

    def test_one_missing_minute_does_not_split(self):
        # 90s tolerance: a missing minute inside a session is not a session end.
        from progeny.market import Bar
        stamps = [TS + timedelta(minutes=i) for i in range(390) if i != 200]
        bars = [Bar(t, 1, 1, 1, 1, 1) for t in stamps]
        self.assertEqual(len(split_sessions(bars)), 1)


class TestSessionPlan(unittest.TestCase):
    def test_default_requirement_matches_docs(self):
        sessions, bars = minimum_sessions()
        self.assertEqual(sessions, 259)
        self.assertEqual(bars, 101_010)

    def test_spans_are_disjoint(self):
        p = plan_from_config(12, 3, 30, 24, 40, 2)
        p.validate(1000)

    def test_detects_overlap(self):
        p = plan_from_config(4, 2, 2, 2, 2, 1)
        object.__setattr__(p, "holdout", type(p.holdout)(0, 5))
        with self.assertRaises(SessionPlanError):
            p.validate(1000)

    def test_fails_when_data_too_short(self):
        p = plan_from_config(12, 3, 30, 24, 40, 2)
        with self.assertRaises(SessionPlanError) as ctx:
            p.validate(50)
        self.assertIn("259", str(ctx.exception))

    def test_spacing_must_be_positive(self):
        with self.assertRaises(SessionPlanError):
            plan_from_config(2, 2, 2, 2, 2, 0)

    def test_rejects_zero_spacing_only(self):
        with self.assertRaises(SessionPlanError):
            plan_from_config(2, 2, 2, 2, 2, -1)

    def test_more_generations_needs_more_sessions(self):
        a = plan_from_config(6, 3, 30, 24, 40, 2).needed
        b = plan_from_config(18, 3, 30, 24, 40, 2).needed
        self.assertGreater(b, a)

    def test_session_span_indices(self):
        p = plan_from_config(3, 2, 4, 4, 6, 2)
        self.assertEqual(p.search.indices(0, 2, 4), [0, 2, 4, 6])


@_skip_without_data()
class TestWindowsFromCommittedData(unittest.TestCase):
    def setUp(self):
        self.m = FileMarket(DATA)
        self.sessions = self.m.sessions("TRD")
        self.cfg = Config(generations=6, episodes=3, bars=390, episode_spacing=2,
                          reselect_windows=6, reselect_final_windows=8,
                          holdout_windows=8)
        self.plan = plan_from_config(6, 3, 8, 6, 8, 2)

    def test_consecutive_generations_share_no_window(self):
        seen = []
        for g in range(self.cfg.generations):
            for bars in windows_for_gen(self.cfg, self.m, "TRD", self.sessions,
                                        g, self.plan.search):
                seen.append((g, bars[0].ts))
        by_gen = {}
        for g, ts in seen:
            by_gen.setdefault(g, set()).add(ts)
        gens = sorted(by_gen)
        for a, b in zip(gens, gens[1:]):
            self.assertFalse(by_gen[a] & by_gen[b],
                             f"generations {a} and {b} shared a window")

    def test_every_window_is_exactly_one_session(self):
        for g in range(3):
            for bars in windows_for_gen(self.cfg, self.m, "TRD", self.sessions,
                                        g, self.plan.search):
                self.assertEqual(len(bars), self.cfg.bars)
                # Contiguous: no internal gap, so one session by construction.
                for a, b in zip(bars, bars[1:]):
                    self.assertEqual(b.ts - a.ts, timedelta(minutes=1))

    def test_search_and_holdout_are_disjoint(self):
        search_idx = set()
        for g in range(self.cfg.generations):
            for bars in windows_for_gen(self.cfg, self.m, "TRD", self.sessions,
                                        g, self.plan.search):
                search_idx.add(bars[0].ts)
        hold_idx = set()
        n = len(self.plan.holdout) // self.plan.spacing
        for k in range(n):
            i = self.plan.holdout.lo + k * self.plan.spacing
            hold_idx.add(self.m.bars("TRD", *self.sessions[i])[0].ts)
        self.assertFalse(search_idx & hold_idx)
        self.assertTrue(hold_idx)

    def test_raises_when_span_too_small(self):
        with self.assertRaises(WindowUnavailable):
            windows_for_gen(self.cfg, self.m, "TRD", self.sessions, 999,
                            self.plan.search)

    def test_run_and_holdout_on_committed_data(self):
        reports, best = run(self.cfg, market=self.m,
                            decisioner=RulesDecisioner(), symbol="TRD")
        self.assertTrue(best.valid)
        h = holdout(best, self.cfg, RulesDecisioner(), self.m, "TRD",
                    self.sessions, self.plan.holdout, self.plan.spacing)
        self.assertEqual(h.n_windows, len(self.plan.holdout) // self.plan.spacing)
        self.assertGreater(h.n_windows, 0)

    def test_buy_and_hold_baseline(self):
        bh = buy_and_hold(self.m, "TRD", self.sessions, self.plan.holdout,
                          self.plan.spacing)
        self.assertGreater(bh.windows, 0)
        # A long-only result is uninterpretable without this number.
        self.assertNotEqual(bh.up_windows, -1)

    def test_same_seed_produces_identical_fitness(self):
        cfg = Config(population=4, generations=3, episodes=2, bars=390,
                     episode_spacing=2, reselect_windows=3,
                     reselect_final_windows=4, holdout_windows=3, seed=99)
        outs = []
        for _ in range(2):
            reps, best = run(cfg, market=self.m, decisioner=RulesDecisioner(),
                             symbol="TRD")
            outs.append([r.best.fitness.fitness for r in reps])
        self.assertEqual(outs[0], outs[1],
                         "same seed gave different fitness across two runs")

    def test_unknown_symbol_raises(self):
        with self.assertRaises(WindowUnavailable):
            run(self.cfg, market=self.m, decisioner=RulesDecisioner(),
                symbol="NOT_A_SYMBOL")

    def test_short_dataset_raises_before_the_loop(self):
        with self.assertRaises(SessionPlanError):
            run(Config(generations=40, episodes=3, reselect_windows=24,
                       reselect_final_windows=40, holdout_windows=30),
                market=self.m, decisioner=RulesDecisioner(), symbol="TRD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
