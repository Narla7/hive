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

from .broker import BrokerConfig, PaperBroker
from .decisioners import AgentState, Decisioner, RulesDecisioner
from .fitness import Episode, Fitness, evaluate, max_drawdown
from .gates import GateConfig, Gates
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
    exploration_c: float = 0.35     # UCB exploration constant
    seed: int = 7
    market_seed: int = 1234
    initial_cash: float = 10_000.0
    decide_every: int = 3          # bars between decisions
    model: str = "offline/rules"
    cross_genome_rate: float = 0.3


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

        gate = gates.check_order(genome.id, bar.ts, abs(target), i - last_fill_bar)
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
    )


def allocate(records: list[Record], exploration_c: float) -> dict[str, int]:
    """UCB over shrunken fitness, weighted by budget share.

    Weighted so a genome with a great score and little evidence still gets
    explored, and so allocation can be refused when nothing deserves it.
    """
    live = [r for r in records if r.genome_id not in ()]
    if not live:
        return {}
    total = sum(r.episodes and len(r.episodes) or 1 for r in live)
    scores: dict[str, float] = {}
    for r in live:
        n = max(1, len(r.episodes))
        # UCB on the shrunk score: exploitation plus an optimism term that
        # scales with how little the genome has been tested.
        optimism = exploration_c * math.sqrt(math.log(total + 1) / n)
        scores[r.genome_id] = r.fitness.raw + optimism
    best = max(scores.values())
    if best <= 0:
        return {r.genome_id: 1 for r in live}
    floor = best * 0.05
    weights = {k: max(0.0, v - floor) for k, v in scores.items()}
    norm = sum(weights.values())
    if norm <= 0:
        return {k: 1 for k in weights}
    return {k: max(1, int(round(v / norm * total))) for k, v in weights.items()}


def holdout(
    genome: Genome,
    cfg: Config,
    decisioner: Decisioner,
    market_seed: int,
    n_windows: int = 5,
    symbol: str = "SIM",
) -> Holdout:
    """Score a genome on price paths it was never selected on."""
    market = SimulatedMarket(seed=market_seed)   # different seed, unseen path
    start = datetime(2026, 3, 2, 14, 30)
    gates = Gates(GateConfig())
    eps = []
    for i in range(n_windows):
        w = market.bars(
            symbol,
            start + timedelta(minutes=i * cfg.episode_spacing),
            start + timedelta(minutes=i * cfg.episode_spacing + cfg.bars),
        )
        eps.append(run_episode(genome, w, decisioner, cfg, gates, GateConfig()))
    fit = evaluate(eps, cfg.initial_cash)
    return Holdout(
        fitness=fit,
        ret=sum(e.pnl for e in eps) / cfg.initial_cash,
        pnl=sum(e.pnl for e in eps),
        trades=sum(e.n_trades for e in eps),
        n_windows=n_windows,
    )


def run(cfg: Config, market: Market | None = None, decisioner: Decisioner | None = None,
        gates_cfg: GateConfig | None = None) -> tuple[list[GenerationReport], Genome]:
    """Run the whole loop. Returns per-generation reports and the best genome."""
    rng = random.Random(cfg.seed)
    decisioner = decisioner or (
        RulesDecisioner() if cfg.model == "offline/rules" else None
    )
    assert decisioner is not None, "no decisioner for model %r" % cfg.model
    market = market or SimulatedMarket(seed=cfg.market_seed)
    gates = Gates(gates_cfg or GateConfig())
    symbol = "SIM"

    start = datetime(2026, 1, 5, 14, 30)
    end = start + timedelta(minutes=cfg.bars)

    population = hand_seeded()[: cfg.population]
    while len(population) < cfg.population:
        population.append(random_genome(rng))

    reports: list[GenerationReport] = []
    all_records: dict[str, Record] = {}
    parent_of: dict[str, str] = {}
    best_genome: Genome = population[0]
    best_score = float("-inf")

    # Each episode gets its own price path, and every genome sees the same set
    # within a generation. Both halves matter:
    #   - same windows across genomes  -> fitness is comparable
    #   - different windows per episode -> episodes are independent samples
    # Running three episodes over one fixed price path with a deterministic
    # decisioner returns three identical results: n_eff is then meaningless and
    # the search can quietly overfit a single price series.
    windows = [
        market.bars(
            symbol,
            start + timedelta(minutes=i * cfg.episode_spacing),
            end + timedelta(minutes=i * cfg.episode_spacing),
        )
        for i in range(cfg.episodes)
    ]

    for gen in range(cfg.generations):
        bars = windows[0]

        records: list[Record] = []
        for g in population:
            verdict = gates.admit(g)
            if not verdict.ok:
                records.append(Record(g.id, gen, evaluate([], cfg.initial_cash)))
                gates.violations.append((g.id, f"gen{gen}", verdict.reason))
                continue
            eps = [
                run_episode(g, w, decisioner, cfg, gates, gates_cfg or GateConfig())
                for w in windows
            ]
            rec = Record(g.id, gen, evaluate(eps, cfg.initial_cash), eps)
            records.append(rec)
            prev = all_records.get(g.id)
            all_records[g.id] = rec
            if prev and g.id in parent_of:
                parent_of[g.id] = parent_of[g.id]
            parent_of.setdefault(g.id, "")

        ranked = sorted(records, key=lambda r: r.fitness.fitness, reverse=True)
        # Track the all-time best, not merely the last one seen to score above
        # zero, so the genome reported at the end is genuinely the best found.
        if ranked and ranked[0].fitness.fitness > best_score:
            match = next((g for g in population if g.id == ranked[0].genome_id), None)
            if match is not None:
                best_genome, best_score = match, ranked[0].fitness.fitness

        mean_f = statistics.fmean([r.fitness.fitness for r in records]) if records else 0.0
        best = ranked[0] if ranked else Record("-", gen, evaluate([], cfg.initial_cash))
        diff = ""
        for g in population:
            if g.id == best.genome_id and parent_of.get(g.id):
                par = next((x for x in population if x.id == parent_of[g.id]), None)
                if par:
                    diff = par.diff(g)

        # ---- breed ----
        survivors = [g for g in population if g.id not in gates.quarantined]
        survivors = sorted(
            survivors,
            key=lambda g: next((r.fitness.fitness for r in records if r.genome_id == g.id), 0.0),
            reverse=True,
        )[: cfg.elite]
        if not survivors:
            survivors = population[: cfg.elite]

        children: list[Genome] = []
        n_new = max(1, int(cfg.population * cfg.novelty_frac))
        for i in range(n_new):
            g = random_genome(rng)
            g.generation = gen + 1
            children.append(g)

        n_cross = int(cfg.population * cfg.cross_genome_rate)
        for _ in range(n_cross):
            if len(survivors) >= 2:
                a, b = rng.sample(survivors, 2)
                child = crossover(a, b, rng)
            elif survivors:
                child = mutate(survivors[0], rng, strength=1.4)
            else:
                child = random_genome(rng)
            child.generation = gen + 1
            if child.valid:
                children.append(child)

        while len(survivors) + len(children) < cfg.population and survivors:
            parent = rng.choice(survivors)
            child = mutate(parent, rng, strength=0.8)
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

    return reports, best_genome
