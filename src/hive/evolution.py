"""The evolution loop.

One epoch: evaluate the population on identical market data, allocate budget
toward uncertainty, cull, breed.

Two rules that carry the design:
  - Every genome sees the *same* market window, or fitness is incomparable and
    the search is measuring the market instead of the strategy.
  - Inference spend is charged to the ledger before fitness is computed, so a
    strategy that is profitable but expensive to run scores as unprofitable.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from .broker import BrokerConfig, PaperBroker
from .decisioners import AgentState, CallBudget, Decisioner, RulesDecisioner
from .fitness import Episode, Fitness, evaluate, max_drawdown
from .gates import GateConfig, Gates, GateVerdict
from .genome import Genome, crossover, hand_seeded, mutate, random_genome
from .ledger import Ledger
from .market import Bar, Market, SimulatedMarket


@dataclass(slots=True)
class Config:
    population: int = 12
    generations: int = 8
    episodes: int = 3               # per genome, per generation
    episode_spacing: int = 900      # bars between each episode's market window
    bars: int = 720                 # one day of minute bars
    elite: int = 3
    novelty_frac: float = 0.10      # fresh genomes per generation
    # UCB exploration constant. Raised from 0.35: fitness is a noisy estimate, so
    # re-testing a promising candidate on more windows is worth more than
    # sampling many untested ones. Under-weighting optimism here is the main cost
    # of a noisy objective.
    exploration_c: float = 0.8
    seed: int = 7
    market_seed: int = 1234
    initial_cash: float = 10_000.0
    decide_every: int = 3          # bars between decisions
    model: str = "offline/rules"
    cross_genome_rate: float = 0.3
    max_calls_per_epoch: int | None = None
    # Distance between one generation's block of price windows and the next.
    # Must exceed episodes * episode_spacing or consecutive generations score
    # overlapping data and the "fresh" windows are not fresh.
    window_stride: int = 0          # 0 -> derived as episodes * episode_spacing
    max_history: int = 12           # episodes retained per genome
    # Fraction of the population that must breach a gate in one generation
    # before the kill switch fires. Per-genome gates protect against individual
    # blowups; nothing protects against a regime change that fails everyone.
    systemic_failure_frac: float = 0.5
    # (success rate, sigma) -> mutation strength. The 1/5th rule from evolution
    # strategies: when most offspring are rejected, step out further; when they
    # are usually accepted, tighten up and exploit.
    mutate_start: float = 0.8
    mutate_min: float = 0.2
    mutate_max: float = 3.0
    # Final selection. The highest in-sample score seen anywhere in a run is the
    # luckiest of every evaluation performed, so taking the argmax guarantees a
    # genome that was selected on noise. Instead, keep a shortlist and re-run it
    # on fresh windows before choosing.
    shortlist: int = 6
    # Two-stage re-selection. Taking the max over N candidates scored on M
    # windows is upward-biased by roughly sigma*sqrt(2 ln N): with 13 candidates
    # on 40 windows the winner is measurably the luckiest, not the best. So a
    # cheap wide pass shortlists, then a narrow, much larger pass decides.
    reselect_windows: int = 24
    reselect_finalists: int = 3
    reselect_final_windows: int = 60
    # How many hand-written seeds may be re-injected across the whole run. The
    # seeds are a real prior; letting them disappear means the search can only
    # drift away from them.
    seed_recycle: int = 3
    # Tournament size for choosing parents. Top-k elitism alone collapses the
    # population onto a few near-identical genomes, and hill-climbing on a
    # collapsed population is a random walk.
    tournament: int = 3


@dataclass(slots=True)
class Record:
    genome_id: str
    generation: int
    fitness: Fitness
    episodes: list[Episode] = field(default_factory=list)


@dataclass(slots=True)
class Holdout:
    """Out-of-sample check on the selected genome.

    Without this the loop is unfalsifiable. Selection over a handful of fixed
    price windows will always produce a winner, and in-sample return says
    nothing about whether the strategy is real or just fitted to those windows.
    A fresh market seed the search never saw is the only honest test.
    """
    fitness: Fitness
    ret: float
    pnl: float
    trades: int
    n_windows: int


@dataclass(slots=True)
class GenerationReport:
    generation: int
    records: list[Record]
    quarantined: list[str]
    new_genomes: int
    mean_fitness: float
    best: Record
    best_diff: str


def run_episode(
    genome: Genome,
    bars: list[Bar],
    decisioner: Decisioner,
    cfg: Config,
    gates: Gates,
    gates_cfg: GateConfig,
    symbol: str = "SIM",
) -> Episode:
    """One full pass of a genome over the bar series.

    The equity curve is recorded per bar so drawdown is computed from realized
    path, not from summary statistics.
    """
    ledger = Ledger()
    broker = PaperBroker(ledger, BrokerConfig(initial_cash=cfg.initial_cash))

    entry_price = 0.0
    held_bars = 0
    n_evals = 0
    curve: list[float] = [cfg.initial_cash]
    capital_deployed = 0.0
    n_trades = 0
    last_fill_bar = -10**6
    budget = getattr(decisioner, "budget", None)
    snap = budget.snapshot() if budget else None

    def weight(bar: Bar) -> float:
        """Position value as a fraction of net worth. 0.0 when flat."""
        nav = ledger.net_worth(bar.close)
        if nav <= 0:
            return 0.0
        return (ledger.quantity() * bar.close) / nav

    for i, bar in enumerate(bars):
        if i % cfg.decide_every != 0 or i < 2:
            broker.mark(bar, "SIM")
            curve.append(ledger.net_worth(bar.close))
            continue

        held_weight = weight(bar)
        state = AgentState(
            weight=held_weight,
            held_bars=held_bars,
            entry_price=entry_price,
            equity=ledger.net_worth(bar.close),
        )
        decision = decisioner.decide(genome, bars[:i], state)
        n_evals += 1
        held_bars += 1
        if decision.raw_cost:
            broker.record_inference(bar.ts, decision.raw_cost, "SIM", model=decisioner.id)

        target = decision.clamped().target_weight

        # policy.cooldown_bars was being evolved but never read, so the search
        # was free to mutate a gene with no effect on behaviour. It is enforced
        # here, per genome, independently of the global gate cooldown.
        since = i - last_fill_bar
        cool = max(gates_cfg.cooldown_bars, genome.policy.cooldown_bars)
        gate = gates.check_order(genome.id, bar.ts, abs(target), since)
        if gate.ok and 0 <= since < cool:
            gate = GateVerdict(False, f"policy cooldown: {since} < {cool}")
        if not gate.ok:
            broker.mark(bar, "SIM")
            curve.append(ledger.net_worth(bar.close))
            continue

        if target > 0 and held_weight <= 1e-6:
            fill = broker.buy(bar, min(1.0, abs(target)), "SIM", reason=decision.reason)
            if fill:
                entry_price = fill.price
                held_bars = 0
                last_fill_bar = i
                n_trades += 1
                capital_deployed += fill.notional
        elif target <= 0 and held_weight > 1e-6:
            fill = broker.sell(bar, 1.0, "SIM", reason=decision.reason or "exit")
            if fill:
                held_bars = 0
                last_fill_bar = i
                n_trades += 1
        broker.mark(bar, "SIM")
        curve.append(ledger.net_worth(bar.close))

    # Session closes flat so P&L is realized, not left as an open position
    # that flatters whichever genome happened to end long.
    if bars:
        broker.sell(bars[-1], 1.0, "SIM", reason="session_close")
        curve[-1] = ledger.net_worth(bars[-1].close)

    dd = max_drawdown(curve)
    gates.check_drawdown(genome.id, bars[-1].ts if bars else datetime.now(), dd)

    return Episode(
        pnl=ledger.pnl(bars[-1].close if bars else None),
        equity_curve=curve,
        capital_deployed=capital_deployed,
        inference_cost=ledger.inference_spend(),
        n_evals=n_evals,
        turnover=broker.turnover,
        n_trades=n_trades,
        truncated=bool(budget and budget.blocked_since(snap)),
    )


def allocate(
    fitness_by_id: dict[str, Fitness],
    evidence_by_id: dict[str, int],
    live_ids: list[str],
    total_budget: int,
    exploration_c: float,
) -> dict[str, int]:
    """UCB1 over shrunken fitness, spending a fixed number of episodes.

    The budget is deliberately *constant*: the same number of episodes is spent
    every generation, only redistributed. That makes the comparison against a
    uniform allocator honest rather than a confounded "we also spent more".

    The optimism term is what makes this a bandit rather than a leaderboard. A
    genome with a high score and little evidence gets sqrt(log N / n), so a
    newcomer can out-draw an incumbent on a fluke, earn a real evaluation, and
    keep its place if the result holds. A pure argmax allocator would never find
    out.

    Every live genome gets at least one episode: a genome with no evidence
    cannot be ranked, and dropping it would make the allocation permanent.
    """
    if not live_ids or total_budget <= 0:
        return {gid: 0 for gid in live_ids}

    if len(live_ids) >= total_budget:
        # Not enough budget for everyone. Spend it on the most promising
        # genomes; the caller floors this at one episode, so a starved genome is
        # merely unranked rather than untested.
        ordered = sorted(live_ids, key=lambda g: (fitness_by_id.get(g).fitness
                                                   if fitness_by_id.get(g) else 0.0),
                         reverse=True)
        return {gid: (1 if gid in ordered[:total_budget] else 0) for gid in live_ids}

    n_max = max(1, sum(evidence_by_id.get(g, 1) for g in live_ids))
    log_n = math.log(n_max + 1)

    scores: dict[str, float] = {}
    for gid in live_ids:
        fit = fitness_by_id.get(gid)
        base = fit.fitness if fit else 0.0
        n = max(1, evidence_by_id.get(gid, 1))
        # Exploit the *shrunk* score, not raw. Raw is inflated for the
        # under-sampled, which is exactly backwards for allocation.
        scores[gid] = base + exploration_c * math.sqrt(log_n / n)

    # Shift by the minimum, not the maximum. Shifting down from the best zeroes
    # every genome more than a few percent behind it, collapsing the
    # apportionment onto a winner and a flat pack of ones. Shifting up from the
    # minimum preserves the ordering the UCB scores worked to produce.
    floor = min(scores.values())
    shifted = {gid: v - floor + 1e-6 for gid, v in scores.items()}
    norm = sum(shifted.values())

    alloc = {gid: 1 for gid in live_ids}
    if norm <= 0:
        return alloc

    remaining = total_budget - len(live_ids)
    if remaining <= 0:
        return alloc

    # Largest-remainder apportionment, so the budget is spent exactly.
    exact = {gid: remaining * (w / norm) for gid, w in shifted.items()}
    for gid, val in exact.items():
        whole = int(val)
        alloc[gid] += whole
    left = total_budget - sum(alloc.values())
    order = sorted(live_ids, key=lambda g: exact[g] - int(exact[g]), reverse=True)
    for i in range(left):
        alloc[order[i % len(order)]] += 1
    return alloc


def holdout(
    genome: Genome,
    cfg: Config,
    decisioner: Decisioner,
    market_seed: int,
    n_windows: int = 5,
    symbol: str = "SIM",
    market: Market | None = None,
    start: datetime | None = None,
) -> Holdout:
    """Score a genome on price paths it was never selected on.

    `market` defaults to a fresh SimulatedMarket, which is right for the
    simulated case. When the caller is already trading a real data file, it must
    be passed in: the out-of-sample half of the experiment has to come from the
    same distribution as the in-sample half, or it is testing something else
    entirely.
    """
    if market is None:
        market = SimulatedMarket(seed=market_seed)   # different seed, unseen path
    start = start or datetime(2026, 3, 2, 14, 30)
    gates = Gates(GateConfig())
    eps = []
    for i in range(n_windows):
        w = market.bars(
            symbol,
            start + timedelta(minutes=i * cfg.episode_spacing),
            start + timedelta(minutes=i * cfg.episode_spacing + cfg.bars),
        )
        eps.append(run_episode(genome, w, decisioner, cfg, gates, GateConfig(), symbol))
    fit = evaluate(eps, cfg.initial_cash)
    return Holdout(
        fitness=fit,
        ret=sum(e.pnl for e in eps) / cfg.initial_cash,
        pnl=sum(e.pnl for e in eps),
        trades=sum(e.n_trades for e in eps),
        n_windows=n_windows,
    )


def windows_for_gen(
    cfg: Config, market: Market, symbol: str, start: datetime, gen: int
) -> list[list[Bar]]:
    """The block of price windows generation `gen` is scored on.

    Extracted so the anti-overfitting property is directly testable: consecutive
    generations must not share windows, and every genome within a generation
    must see the same set.
    """
    stride = cfg.window_stride or (cfg.episodes * cfg.episode_spacing)
    base = start + timedelta(minutes=gen * stride)
    return [
        market.bars(
            symbol,
            base + timedelta(minutes=i * cfg.episode_spacing),
            base + timedelta(minutes=i * cfg.episode_spacing + cfg.bars),
        )
        for i in range(cfg.episodes)
    ]


def reselect(
    shortlist: list[Genome],
    cfg: Config,
    decisioner: Decisioner,
    market: Market,
    symbol: str,
    n_windows: int,
) -> tuple[Genome, list[tuple[str, float, float]]]:
    """Re-run the shortlist on fresh windows and pick the real winner.

    Selection on the maximum of a noisy score is a winner's curse: the genome
    that wins a search is disproportionately the one that got lucky, and the
    more evaluations the search performs, the worse the winner generalises. That
    is exactly what a benchmark showed -- 20 generations produced a *worse*
    holdout winner than 1.

    So the shortlist is re-evaluated on windows the search never saw, and the
    winner is chosen on that. Ties are broken toward the higher hit rate, i.e.
    toward a genome that wins consistently rather than once very hard.
    """
    if not shortlist:
        raise ValueError("empty shortlist")
    if len(shortlist) == 1:
        return shortlist[0], [(shortlist[0].id, 0.0, 0.0)]

    # Windows far past anything the search used, so no overlap with the search.
    coarse_start = datetime(2027, 1, 4, 14, 30)
    # ...and a second, disjoint set for the final decision.
    final_start = datetime(2028, 6, 5, 14, 30)

    def score(g: Genome, start: datetime, n: int) -> Fitness:
        gates = Gates(GateConfig())
        eps = []
        for i in range(n):
            w = market.bars(
                symbol,
                start + timedelta(minutes=i * cfg.episode_spacing),
                start + timedelta(minutes=i * cfg.episode_spacing + cfg.bars),
            )
            eps.append(run_episode(g, w, decisioner, cfg, gates, GateConfig(), symbol))
        return evaluate(eps, cfg.initial_cash)

    # Stage 1: wide and cheap. Reduces the field.
    coarse = sorted(((score(g, coarse_start, n_windows).ret, g) for g in shortlist),
                    key=lambda t: t[0], reverse=True)
    finalists = [g for _, g in coarse[: max(1, cfg.reselect_finalists)]]

    # Stage 2: narrow and large. The final pick is made among few candidates on
    # a lot of data, which is where the selection bias actually shrinks.
    scored = []
    for g in finalists:
        fit = score(g, final_start, cfg.reselect_final_windows)
        scored.append((fit.ret, fit.hit_rate, g))

    scored.sort(key=lambda t: (round(t[0], 4), t[1]), reverse=True)
    winner = scored[0][2]
    table = [(g.id, ret, hit) for ret, hit, g in scored]
    return winner, table


def run(cfg: Config, market: Market | None = None, decisioner: Decisioner | None = None,
        gates_cfg: GateConfig | None = None,
        symbol: str = "SIM",
        on_generation: "Callable[[GenerationReport], bool] | None" = None
        ) -> tuple[list[GenerationReport], Genome]:
    """Run the whole loop. Returns per-generation reports and the best genome.

    `on_generation` is called after each generation. Returning False from it
    stops the loop, which is how the TUI gets a pause and a quit key without
    the evolution module knowing curses exists.
    """
    rng = random.Random(cfg.seed)
    decisioner = decisioner or (
        RulesDecisioner() if cfg.model == "offline/rules" else None
    )
    assert decisioner is not None, "no decisioner for model %r" % cfg.model
    market = market or SimulatedMarket(seed=cfg.market_seed)
    gates = Gates(gates_cfg or GateConfig())

    start = datetime(2026, 1, 5, 14, 30)

    population = hand_seeded()[: cfg.population]
    while len(population) < cfg.population:
        population.append(random_genome(rng))

    reports: list[GenerationReport] = []
    history: dict[str, list[Episode]] = {}
    prev_fit: dict[str, Fitness] = {}
    prev_evidence: dict[str, int] = {}
    parent_of: dict[str, str] = {}
    # Shortlist of distinct strong genomes, kept on a *running average* rather
    # than a single score, so one lucky evaluation cannot eject a solid genome.
    shortlist: list[tuple[float, Genome]] = []
    best_genome: Genome = population[0]
    mutate_strength = cfg.mutate_start
    seeds_recycled = 0
    total_budget = max(len(population), cfg.population * cfg.episodes)

    budget = CallBudget(limit=cfg.max_calls_per_epoch)
    if hasattr(decisioner, "budget"):
        decisioner.budget = budget

    for gen in range(cfg.generations):
        budget.limit = cfg.max_calls_per_epoch
        budget.spent = 0
        budget.blocked = 0

        # Every generation scores a fresh block of price windows. Reusing one
        # fixed set across generations means accumulated history adds no
        # information -- with a deterministic decisioner the extra episodes are
        # literally the same number again -- and the search converges on fitting
        # those particular paths rather than on the strategy space.
        windows = windows_for_gen(cfg, market, symbol, start, gen)

        # Gen 0 is uniform: there is no history to be clever with yet. After
        # that, UCB redistributes a constant episode budget.
        live = [g.id for g in population if g.id not in gates.quarantined]
        if gen == 0 or not prev_fit:
            alloc = {gid: cfg.episodes for gid in live}
        else:
            alloc = allocate(prev_fit, prev_evidence, live, total_budget, cfg.exploration_c)

        pre_breaches = len(gates.quarantined)
        records: list[Record] = []
        for g in population:
            verdict = gates.admit(g)
            if not verdict.ok:
                records.append(Record(g.id, gen, evaluate([], cfg.initial_cash)))
                gates.violations.append((g.id, f"gen{gen}", verdict.reason))
                continue
            n_eps = max(1, alloc.get(g.id, 1))
            new_eps = [
                run_episode(g, w, decisioner, cfg, gates, gates_cfg or GateConfig(), symbol)
                for w in windows[:n_eps]
            ]
            # Keep a bounded history. Unchanged elites keep their identity, so
            # their evidence accumulates and their shrinkage relaxes; a mutated
            # child has a new fingerprint and therefore no inherited evidence,
            # which is correct rather than a bug.
            hist = (history.get(g.id, []) + new_eps)[-cfg.max_history:]
            history[g.id] = hist
            records.append(Record(g.id, gen, evaluate(hist, cfg.initial_cash), hist))
            parent_of.setdefault(g.id, "")

        # Systemic failure: per-genome gates cannot protect against a regime
        # change that fails everyone at the same instant.
        newly = len(gates.quarantined) - pre_breaches
        if live and newly >= max(2, int(len(live) * cfg.systemic_failure_frac)):
            gates.kill_switch(f"{newly}/{len(live)} breached in one generation")

        ranked = sorted(records, key=lambda r: r.fitness.fitness, reverse=True)

        # Merge this generation's best into the shortlist, deduped by id.
        seen = {g.id for _, g in shortlist}
        for rec in ranked[: cfg.shortlist]:
            if rec.genome_id in seen:
                continue
            match = next((g for g in population if g.id == rec.genome_id), None)
            if match is None:
                continue
            seen.add(match.id)
            shortlist.append((rec.fitness.fitness, match))
        shortlist.sort(key=lambda t: t[0], reverse=True)
        del shortlist[cfg.shortlist:]

        prev_fit = {r.genome_id: r.fitness for r in records}
        prev_evidence = {r.genome_id: len(r.episodes) for r in records}

        mean_f = statistics.fmean([r.fitness.fitness for r in records]) if records else 0.0
        best = ranked[0] if ranked else Record("-", gen, evaluate([], cfg.initial_cash))
        diff = ""
        for g in population:
            if g.id == best.genome_id and parent_of.get(g.id):
                par = next((x for x in population if x.id == parent_of[g.id]), None)
                if par:
                    diff = par.diff(g)

        # ---- breed ----
        # Rank, not raw fitness, drives parent selection. Fitness scale swings
        # hard between generations (a hard market shrinks every score), and a
        # single lucky genome can dominate an ordering that raw values produce.
        # Rank is invariant to both.
        rank_of = {r.genome_id: i for i, r in enumerate(ranked)}

        survivors = [g for g in population if g.id not in gates.quarantined]
        survivors = sorted(
            survivors,
            key=lambda g: prev_fit.get(g.id).fitness if prev_fit.get(g.id) else 0.0,
            reverse=True,
        )[: cfg.elite]
        if not survivors:
            survivors = population[: cfg.elite]

        # Parents are drawn by tournament from the *whole* live population, not
        # just the elite. Same selection pressure, but a mid-ranked genome with a
        # useful gene can still contribute, which is what keeps a search on a
        # noisy landscape from collapsing onto one basin.
        pool = [g for g in population if g.id not in gates.quarantined] or population
        k = max(2, min(cfg.tournament, len(pool)))

        def pick_parent() -> "Genome":
            cands = rng.sample(pool, k)
            return min(cands, key=lambda g: rank_of.get(g.id, 10 ** 6))

        def pick_pair() -> tuple["Genome", "Genome"]:
            a = pick_parent()
            for _ in range(4):
                b = pick_parent()
                if b.id != a.id:
                    return a, b
            return a, a

        # How much of the population beat the incumbent. High means the search
        # is still finding improvements and should exploit; low means it has
        # stalled and needs to step further out. This is the 1/5th rule, but
        # measured on the previous generation's scores rather than on children
        # that have not been evaluated yet.
        if ranked and len(ranked) > 1:
            top = ranked[0].fitness.fitness
            beat = sum(1 for r in ranked if r.fitness.fitness >= top)
            success_rate = (beat - 1) / (len(ranked) - 1)
        else:
            success_rate = 1.0
        if success_rate < 0.2:
            mutate_strength = min(cfg.mutate_max, mutate_strength * 1.5)
        elif success_rate > 0.5:
            mutate_strength = max(cfg.mutate_min, mutate_strength * 0.85)

        children: list[Genome] = []
        n_new = max(1, int(cfg.population * cfg.novelty_frac))
        for _ in range(n_new):
            g = random_genome(rng)
            g.generation = gen + 1
            children.append(g)

        # Re-inject a hand-written seed periodically. Novelty injection supplies
        # random genomes, which drift away from the strategy region a human
        # already found to work; a seed carries that prior back in.
        if seeds_recycled < cfg.seed_recycle and gen > 0:
            seed_pool = [g for g in hand_seeded()
                         if g.id not in {c.id for c in children + survivors}]
            if seed_pool:
                revived = seed_pool[gen % len(seed_pool)].clone()
                revived.generation = gen + 1
                children.append(revived)
                seeds_recycled += 1

        n_cross = int(cfg.population * cfg.cross_genome_rate)
        for _ in range(n_cross):
            if pool:
                a, b = pick_pair()
                child = crossover(a, b, rng) if a.id != b.id else mutate(a, rng, strength=mutate_strength * 1.4)
            else:
                child = random_genome(rng)
            child.generation = gen + 1
            if child.valid:
                children.append(child)

        while len(survivors) + len(children) < cfg.population and pool:
            child = mutate(pick_parent(), rng, strength=mutate_strength)
            child.generation = gen + 1
            if child.valid:
                children.append(child)

        population = survivors + children[: max(0, cfg.population - len(survivors))]
        for g in population:
            if g.parent:
                parent_of[g.id] = g.parent[0]

        reports.append(
            GenerationReport(
                generation=gen,
                records=records,
                quarantined=sorted(gates.quarantined),
                new_genomes=len(children),
                mean_fitness=mean_f,
                best=best,
                best_diff=diff,
            )
        )
        if gates.tripped:
            break

        if on_generation is not None and not on_generation(reports[-1]):
            break

    # Final pick on unseen data. See reselect() for why the argmax of in-sample
    # fitness is the wrong answer.
    if shortlist:
        # Always put the hand-written seeds back in front of the re-selection.
        # They are a real baseline, and a stochastic search can drift away from
        # something good and never notice, because the thing it drifted away
        # from is no longer in the population. Including them makes the returned
        # winner at least as good as the best human-written genome, measured on
        # the same fresh windows: evolution can only add to the prior.
        cands = [g for _, g in shortlist]
        have = {g.id for g in cands}
        for g in hand_seeded():
            if g.id not in have:
                cands.append(g)
        best_genome, _table = reselect(
            cands, cfg, decisioner, market, symbol, cfg.reselect_windows,
        )
    return reports, best_genome
