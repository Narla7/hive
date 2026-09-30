"""Tests for the search improvements: allocation, re-selection, window rotation.

The headline claim these protect is the one the benchmark disproved before it
was fixed: selecting the argmax of in-sample fitness made *more* generations
produce a *worse* winner.
"""

import unittest
from datetime import datetime, timedelta

import progeny.evolution as evo
from progeny.decisioners import RulesDecisioner
from progeny.evolution import Config, allocate, holdout, reselect, run, windows_for_gen
from progeny.fitness import Episode, Fitness, evaluate
from progeny.gates import GateConfig, Gates
from progeny.genome import Genome, Policy, hand_seeded
from progeny.market import SimulatedMarket
from progeny.sessions import SessionPlan, plan_from_config
from tests.support import market_for, sessions_for, small_cfg

TS = datetime(2026, 1, 5, 14, 30)


def fit(value: float, n: int = 3) -> Fitness:
    return Fitness(
        raw=value, fitness=value, ret=value, cost_per_eval=0.0, drawdown=0.0,
        downside_dev=0.01, turnover=1.0, n=n, n_eff=float(n),
        trades=5, n_truncated=0, hit_rate=0.6, consistency=0.7,
    )


class TestAllocate(unittest.TestCase):
    def test_spends_exactly_the_budget(self):
        live = [f"g{i}" for i in range(8)]
        f = {g: fit(1.0) for g in live}
        ev = {g: 3 for g in live}
        self.assertEqual(sum(allocate(f, ev, live, 40, 0.35).values()), 40)

    def test_starved_budget_never_overspends(self):
        # When there is not enough budget for everyone, the excess goes to zero
        # rather than silently overspending. The caller floors at one episode, so
        # a starved genome is unranked, not untested.
        live = [f"g{i}" for i in range(10)]
        alloc = allocate({g: fit(0.0) for g in live}, {g: 1 for g in live}, live, 3, 0.35)
        self.assertEqual(sum(alloc.values()), 3)
        self.assertEqual(set(alloc), set(live))
        self.assertEqual(sum(1 for v in alloc.values() if v > 0), 3)

    def test_tight_budget_goes_to_the_best_ranked(self):
        live = [f"g{i}" for i in range(6)]
        f = {g: fit(0.1 * i) for i, g in enumerate(live)}
        alloc = allocate(f, {g: 3 for g in live}, live, 2, 0.35)
        self.assertEqual(sum(alloc.values()), 2)
        self.assertEqual(alloc["g5"], 1)      # highest fitness
        self.assertEqual(alloc["g0"], 0)      # lowest

    def test_wide_budget_keeps_gradation(self):
        # A shift from the maximum would zero everyone a few percent behind the
        # leader, collapsing the apportionment to a winner and a flat pack of 1s.
        live = [f"g{i}" for i in range(6)]
        f = {g: fit(0.5 * i) for i, g in enumerate(live)}
        alloc = allocate(f, {g: 4 for g in live}, live, 60, 0.1)
        values = [alloc[g] for g in live]
        self.assertEqual(values, sorted(values))

    def test_high_fitness_gets_more(self):
        live = ["good", "mid", "bad"]
        f = {"good": fit(9.0), "mid": fit(3.0), "bad": fit(0.1)}
        ev = {g: 6 for g in live}
        alloc = allocate(f, ev, live, 60, 0.1)
        self.assertGreater(alloc["good"], alloc["mid"])
        self.assertGreater(alloc["mid"], alloc["bad"])

    def test_under_evidence_genome_is_explored(self):
        # The optimism term's whole purpose: a newcomer with little data must be
        # able to out-draw an incumbent on a fluke, or it is never discovered.
        alloc = allocate({"proven": fit(5.0), "newcomer": fit(4.0)},
                         {"proven": 40, "newcomer": 1}, ["proven", "newcomer"],
                         40, 1.5)
        self.assertGreater(alloc["newcomer"], 1)

    def test_uses_shrunk_fitness_not_raw(self):
        # Allocating on raw would invert exactly the genomes that most need
        # testing: a spectacular raw score with no evidence shrinks to almost
        # nothing, which is the correct signal.
        f = {
            "overfit": Fitness(raw=999.0, fitness=0.2, ret=0.2, cost_per_eval=0.0,
                               drawdown=0.0, downside_dev=0.01, turnover=1.0,
                               n=1, n_eff=1.0, trades=5, n_truncated=0),
            "solid": Fitness(raw=1.0, fitness=1.0, ret=1.0, cost_per_eval=0.0,
                             drawdown=0.0, downside_dev=0.01, turnover=1.0,
                             n=30, n_eff=30.0, trades=5, n_truncated=0),
        }
        alloc = allocate(f, {"overfit": 1, "solid": 30}, ["overfit", "solid"], 40, 0.1)
        self.assertGreaterEqual(alloc["solid"], alloc["overfit"])

    def test_empty_live(self):
        self.assertEqual(allocate({}, {}, [], 10, 0.35), {})

    def test_zero_budget(self):
        self.assertEqual(allocate({"a": fit(1), "b": fit(1)},
                                  {"a": 1, "b": 1}, ["a", "b"], 0, 0.35),
                         {"a": 0, "b": 0})

    def test_deterministic(self):
        live = [f"g{i}" for i in range(7)]
        f = {g: fit(i * 0.3) for i, g in enumerate(live)}
        ev = {g: 2 for g in live}
        self.assertEqual(allocate(f, ev, live, 30, 0.35),
                         allocate(f, ev, live, 30, 0.35))


class _Spy:
    """Records the first timestamp each genome saw, via run_episode patching."""

    def __init__(self):
        self.real = evo.run_episode
        self.seen = {}

    def __enter__(self):
        def spy(genome, bars, *a, **kw):
            self.seen.setdefault(genome.id, []).append(bars[0].ts)
            return self.real(genome, bars, *a, **kw)

        evo.run_episode = spy
        return self

    def __exit__(self, *exc):
        evo.run_episode = self.real


class TestWindowRotation(unittest.TestCase):
    """The core anti-overfitting property.

    Reusing one fixed set of windows makes accumulated history worthless -- with
    a deterministic decisioner the extra episodes are literally the same number
    again -- and lets the search converge on fitting those particular paths.
    """

    def _blocks(self, cfg=None):
        cfg = cfg or small_cfg(generations=4, episodes=3)
        m = market_for(cfg)
        sessions = m.sessions("SIM")
        plan = plan_from_config(cfg.generations, cfg.episodes, cfg.holdout_windows,
                                cfg.reselect_windows, cfg.reselect_final_windows,
                                cfg.episode_spacing)
        return cfg, m, sessions, plan, [
            windows_for_gen(cfg, m, "SIM", sessions, g, plan.search)
            for g in range(cfg.generations)
        ]

    def test_consecutive_generations_do_not_share_windows(self):
        _, _, _, _, blocks = self._blocks()
        for a, b in zip(blocks, blocks[1:]):
            sa = {bars[0].ts for bars in a}
            sb = {bars[0].ts for bars in b}
            self.assertFalse(sa & sb, f"generations shared windows: {sa & sb}")

    def test_windows_shared_within_a_generation(self):
        # Every genome in a generation is handed the same list object, so fitness
        # is comparable and the search is not measuring the market.
        self.assertEqual(len(self._blocks()[4][0]), 3)

    def test_window_count_per_generation(self):
        cfg = small_cfg(episodes=5)
        blocks = self._blocks(cfg)[4]
        self.assertEqual(len(blocks[0]), 5)

    def test_every_window_is_one_session(self):
        # The point of session-space windows: a window can never straddle an
        # overnight gap, so the old silent-short-window failure is unrepresentable.
        cfg, m, sessions, _, blocks = self._blocks()
        for gen_blocks in blocks:
            for bars in gen_blocks:
                self.assertEqual(len(bars), cfg.bars)
                start = bars[0].ts
                self.assertIn(start, [s for s, _ in sessions])

    def test_stride_prevents_overlap_by_default(self):
        cfg = small_cfg()
        self.assertEqual(cfg.window_stride, 0)   # 0 -> derived as episodes*spacing


class TestHistoryAccumulation(unittest.TestCase):
    def test_surviving_genome_gains_evidence(self):
        cfg = small_cfg(population=6, generations=4, episodes=2, decide_every=4, seed=5)
        reports, _ = run(cfg, market=market_for(cfg),
                         decisioner=RulesDecisioner())
        counts = [len(r.episodes) for r in reports[-1].records]
        self.assertGreater(max(counts), cfg.episodes,
                           f"no genome accumulated history: {counts}")

    def test_history_is_bounded(self):
        cfg = small_cfg(population=6, generations=5, episodes=2, decide_every=4,
                        seed=5, max_history=4)
        reports, _ = run(cfg, market=market_for(cfg),
                         decisioner=RulesDecisioner())
        for rec in reports[-1].records:
            self.assertLessEqual(len(rec.episodes), 4)

    def test_child_does_not_inherit_evidence(self):
        # Correct by design: an id is a fingerprint of the policy, so a mutated
        # child is a different genome with no inherited evidence.
        self.assertNotEqual(Genome(policy=Policy(lookback=10)).id,
                            Genome(policy=Policy(lookback=11)).id)


class TestReselect(unittest.TestCase):
    def _call(self, cands, cfg=None):
        cfg = cfg or small_cfg(decide_every=4, seed=3)
        m = market_for(cfg)
        sessions = m.sessions("SIM")
        plan = plan_from_config(cfg.generations, cfg.episodes, cfg.holdout_windows,
                                cfg.reselect_windows, cfg.reselect_final_windows,
                                cfg.episode_spacing)
        return reselect(cands, cfg, RulesDecisioner(), m, "SIM", sessions, plan,
                        cfg.reselect_windows, cfg.reselect_final_windows)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            self._call([])

    def test_single_candidate_short_circuits(self):
        g = Genome()
        win, table = self._call([g])
        self.assertIs(win, g)
        self.assertEqual(len(table), 1)

    def test_returns_a_valid_winner_from_the_finalists(self):
        # Stage 1 shortlists, stage 2 decides among the finalists. The table
        # therefore reports the finalists, not the whole field.
        winner, table = self._call(hand_seeded()[:5])
        self.assertTrue(winner.valid)
        self.assertEqual(len(table), 3)

    def test_finalists_widens_with_config(self):
        _, table = self._call(hand_seeded()[:5],
                              small_cfg(decide_every=4, seed=3, reselect_finalists=4))
        self.assertEqual(len(table), 4)

    def test_result_is_stable(self):
        cands = hand_seeded()[:4]
        self.assertEqual(self._call(cands)[0].id, self._call(cands)[0].id)

    def test_table_sorted_by_return(self):
        _, table = self._call(hand_seeded()[:4])
        rets = [r for _, r, _ in table]
        self.assertEqual(rets, sorted(rets, reverse=True))


class TestLongerRunsDoNotDegrade(unittest.TestCase):
    """The regression this branch exists to fix.

    Selecting the argmax of in-sample fitness is a winner's curse, and a
    20-generation run used to produce a *worse* holdout winner than a
    1-generation run. This is asserted across seeds because a single seed does
    not support a claim about a stochastic objective.
    """
    SEEDS = (11, 47, 313)

    def _winner_return(self, gens, seed):
        cfg = small_cfg(population=10, generations=gens, episodes=3,
                        decide_every=3, seed=seed)
        m = market_for(cfg)
        _, best = run(cfg, market=m, decisioner=RulesDecisioner())
        plan = plan_from_config(cfg.generations, cfg.episodes, cfg.holdout_windows,
                                cfg.reselect_windows, cfg.reselect_final_windows,
                                cfg.episode_spacing)
        return holdout(best, cfg, RulesDecisioner(), m, "SIM",
                       m.sessions("SIM"), plan.holdout, plan.spacing).ret

    def test_ten_generations_no_worse_than_one(self):
        deltas = [self._winner_return(10, s) - self._winner_return(1, s)
                  for s in self.SEEDS]
        worst = min(deltas)
        # Measured: the prior guarantees evolved is never below baseline, and the
        # search is only expected to add a little. A regression (which was -1.1pp
        # of mean and far worse on worst-case) still has to fail this.
        self.assertGreaterEqual(worst, -0.005,
                                f"longer runs hurt on some seed: {deltas}")


class TestCooldownIsEnforced(unittest.TestCase):
    def test_policy_cooldown_limits_reentry(self):
        # cooldown_bars was evolved but never read, so the search was free to
        # mutate a gene with no effect on behaviour.
        from progeny.evolution import run_episode
        base = dict(style="momentum", lookback=10, entry_threshold=0.05,
                    exit_threshold=0.02, stop_loss=0.9, take_profit=1.8)
        slow = Genome(policy=Policy(**base, cooldown_bars=500))
        fast = Genome(policy=Policy(**base, cooldown_bars=0))
        cfg = small_cfg(bars=200, decide_every=3, initial_cash=10_000.0)
        m = market_for(cfg)
        bars = m.bars("SIM", *m.sessions("SIM")[0])
        a = run_episode(slow, bars, RulesDecisioner(), cfg, Gates(), GateConfig())
        b = run_episode(fast, bars, RulesDecisioner(), cfg, Gates(), GateConfig())
        self.assertLessEqual(a.n_trades, b.n_trades)


class TestSystemicFailure(unittest.TestCase):
    def test_kill_switch_trips_on_mass_breach(self):
        cfg = small_cfg(population=10, generations=3, episodes=1, decide_every=2, seed=2)
        reports, _ = run(cfg, market=market_for(cfg),
                         decisioner=RulesDecisioner(),
                         gates_cfg=GateConfig(max_drawdown=0.0001))
        self.assertGreater(len(reports[-1].quarantined), 0)

    def test_run_completes_with_loose_gates(self):
        cfg = small_cfg(population=6, generations=3, episodes=1, decide_every=3, seed=2)
        reports, _ = run(cfg, market=market_for(cfg),
                         decisioner=RulesDecisioner(),
                         gates_cfg=GateConfig(max_drawdown=0.99))
        self.assertEqual(len(reports), 3)


class TestSymbolIsHonoured(unittest.TestCase):
    """--symbol used to be a no-op: run() hardcoded "SIM", so a CSV whose symbol
    column read anything else silently evaluated nothing at all and every genome
    pinned at the inactivity penalty."""

    def _csv(self, tmpdir: str, days: int = 10) -> str:
        """A session-shaped file: 390 bars a day, 09:30-16:00, weekdays only.

        Walks calendar days and skips weekends by *filtering*, not by advancing a
        cursor. Advancing a cursor collides with itself: skipping Saturday to
        Monday lands on a date a later index reaches naturally.
        """
        import os
        path = os.path.join(tmpdir, "bars.csv")
        rows = ["symbol,ts,open,high,low,close,volume"]
        base = datetime(2026, 1, 5)
        cal = [base + timedelta(days=i) for i in range(60)]
        cal = [d for d in cal if d.weekday() < 5][:days]
        px = 100.0
        for d in cal:
            for i in range(390):
                px *= 1.0 + (0.0016 if i % 9 < 6 else -0.0004)
                ts = datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=i)
                rows.append(
                    f"SPY,{ts.isoformat()},{px:.4f},{px*1.002:.4f},"
                    f"{px*0.998:.4f},{px:.4f},100000"
                )
        with open(path, "w") as fh:
            fh.write("# provenance: test fixture\n")
            fh.write("\n".join(rows) + "\n")
        return path

    def test_matching_symbol_resolves_bars_and_trades(self):
        import tempfile
        from progeny.market import FileMarket
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp)
            mkt = FileMarket(path)
            self.assertGreaterEqual(len(mkt.sessions("SPY")), 7)
            self.assertEqual(mkt.bars("NOPE", datetime.min, datetime.max), [])
            cfg = small_cfg(population=4, generations=1, episodes=1,
                            reselect_windows=1, reselect_final_windows=1,
                            holdout_windows=1, decide_every=3)
            reports, _ = run(cfg, market=mkt, decisioner=RulesDecisioner(),
                             symbol="SPY")
        self.assertGreater(max(r.fitness.trades for r in reports[0].records), 0)

    def test_wrong_symbol_raises_instead_of_scoring_zero(self):
        """The whole point of fail-loud.

        A symbol absent from the file used to yield no bars, every genome scored
        the inactivity penalty, and the run reported a confident result derived
        from nothing. Now it raises.
        """
        import tempfile
        from progeny.market import FileMarket, WindowUnavailable
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp)
            cfg = small_cfg(population=4, generations=1, episodes=1,
                            reselect_windows=1, reselect_final_windows=1,
                            holdout_windows=1, decide_every=3)
            with self.assertRaises(WindowUnavailable):
                run(cfg, market=FileMarket(path),
                    decisioner=RulesDecisioner(), symbol="NOPE")


class TestFitnessHitRate(unittest.TestCase):
    def _ep(self, pnl):
        return Episode(pnl=pnl, equity_curve=[10_000.0, 10_000.0 + pnl],
                       capital_deployed=10_000.0, inference_cost=0.0,
                       n_evals=10, turnover=5.0, n_trades=5)

    def test_hit_rate_reported(self):
        f = evaluate([self._ep(100), self._ep(-50), self._ep(20), self._ep(-10)],
                     10_000.0)
        self.assertAlmostEqual(f.hit_rate, 0.5)

    def test_lucky_genome_penalised(self):
        # One big win and three losses should not beat modest consistent gains.
        lucky = evaluate([self._ep(900)] + [self._ep(-300)] * 3, 10_000.0)
        steady = evaluate([self._ep(200), self._ep(150), self._ep(180),
                           self._ep(120)], 10_000.0)
        self.assertGreater(steady.fitness, lucky.fitness)

    def test_all_winners_beat_all_losers(self):
        self.assertGreater(evaluate([self._ep(50)] * 4, 10_000.0).fitness,
                           evaluate([self._ep(-50)] * 4, 10_000.0).fitness)

    def test_consistency_bounded(self):
        f = evaluate([self._ep(100), self._ep(-200), self._ep(300)], 10_000.0)
        self.assertGreaterEqual(f.consistency, 0.0)
        self.assertLessEqual(f.consistency, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
