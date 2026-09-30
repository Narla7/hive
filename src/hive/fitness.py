"""Fitness.

Reads only from the ledger. Pure function of episode results: no clock, no
network, no adapter state. Everything downstream (selection, culling, credit
assignment) is built on this, so it has to be trustworthy above all else.

The important term is the shrinkage factor. A genome with a spectacular score off
three episodes must be *shrunk*, not rewarded — otherwise the loop latches onto
early noise, gives the winner more budget, and mistakes the resulting sample
growth for confirmation. That failure is invisible from the outside: the loop
looks busy and reports confident winners.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Regularizer weights. λ0 cost, λ1 drawdown, λ2 turnover.
LAMBDA_COST = 0.35
LAMBDA_DRAWDOWN = 1.0
# Small on purpose. Trading frictions are already charged in the ledger, so a
# large turnover weight double-counts them and crushes exactly the
# lower-churn strategies that survive costs. This is a tiebreaker, not a cost.
LAMBDA_TURNOVER = 0.0002
# Weight on the cross-window hit rate. A genome that makes money on some price
# paths and loses on others has an edge that is a coin flip, however good its
# average looks. Penalising low hit rate pushes selection toward strategies whose
# edge is *stable* across markets rather than lucky on one.
LAMBDA_HITRATE = 0.4
PRIOR_STRENGTH = 6.0   # k in n_eff/(n_eff + k)
EPS = 1e-4
# Floor on downside deviation, as a fraction of equity. Dividing by a
# near-zero denominator amplifies sampling noise into enormous raw scores, so
# the Sharpe-style ratio is meaningless below this scale.
MIN_DOWNSIDE_DEV = 0.005
MIN_TRADES_PER_EPISODE = 2.0   # a genome below this trades in is not yet proven
INACTIVITY_PENALTY = 1.5      # additive, in raw-score units


@dataclass(slots=True)
class Episode:
    """One full run of one genome over one market window."""
    pnl: float
    equity_curve: list[float]
    capital_deployed: float
    inference_cost: float
    n_evals: int
    turnover: float
    n_trades: int = 0
    truncated: bool = False     # call budget ran out mid-episode
    error: str | None = None


@dataclass(slots=True)
class Fitness:
    raw: float
    fitness: float          # shrunk; this is what selection uses
    ret: float
    cost_per_eval: float
    drawdown: float
    downside_dev: float
    turnover: float
    n: int
    n_eff: float
    trades: int = 0
    n_truncated: int = 0
    hit_rate: float = 0.0     # fraction of scored windows that made money
    consistency: float = 0.0  # 1 - downside_dev / (mean |return|), 0..1

    @property
    def shrinked(self) -> bool:
        return self.fitness < self.raw * 0.999


def max_drawdown(curve: list[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def effective_n(values: list[float]) -> float:
    """Sample size corrected for autocorrelation.

    Episode returns are serially correlated: winning strategies cluster wins and
    broken ones repeat their mistake. Using the raw count badly overstates the
    evidence and defeats the shrinkage term, which is exactly the protection
    being sought here.
    """
    n = len(values)
    if n < 3:
        return float(n)
    mean = statistics.fmean(values)
    var = statistics.pvariance(values)
    if var <= 1e-12:
        return float(n)

    max_lag = min(n - 1, 20)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        num = sum(
            (values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)
        )
        den = (n - lag) * var
        rho = num / den if den else 0.0
        if rho <= 0.0:
            break
        rho_sum += rho
    return max(1.0, n / (1.0 + 2.0 * rho_sum))


def downside_dev(values: list[float], floor: float = MIN_DOWNSIDE_DEV) -> float:
    losses = [v for v in values if v < 0]
    if len(losses) < 2:
        return floor
    return max(floor, statistics.stdev(losses))


def evaluate(episodes: list[Episode], initial_cash: float) -> Fitness:
    """Score a genome across its episode history."""
    # Truncated episodes are excluded, not merely flagged. A half-length episode
    # understates a genome because it had less time to trade, so scoring it
    # would bias selection toward strategies that churn fast.
    n_truncated = sum(1 for e in episodes if e.truncated and e.error is None)
    ok = [e for e in episodes if e.error is None and not e.truncated]
    n = len(ok)
    if n == 0:
        # Still report truncation, otherwise "everything was truncated" is
        # indistinguishable from "nothing ran".
        return Fitness(0.0, 0.0, 0.0, 0.0, 0.0, EPS, 0.0, 0, 0.0, 0, n_truncated)

    # Return on account equity, not on deployed capital. Normalizing by
    # deployed capital deflates every high-turnover strategy and hides the fact
    # that churning costs money. "How much did this make on the account" is the
    # only number that answers the question the project is asking.
    ret = sum(e.pnl for e in ok) / initial_cash
    inference = sum(e.inference_cost for e in ok)
    n_evals = sum(e.n_evals for e in ok) or 1
    cost_per_eval = inference / n_evals
    turnover = sum(e.turnover for e in ok) / initial_cash
    trades = sum(e.n_trades for e in ok)

    # Worst episode drawdown across the run, since one blowup matters more than
    # the average good day.
    dd = max((max_drawdown(e.equity_curve) for e in ok), default=0.0)
    ddev = downside_dev([e.pnl / initial_cash for e in ok])
    n_eff = effective_n([e.pnl / initial_cash for e in ok])

    # Hit rate across independently-scored windows. This is the cheapest
    # available test of whether an edge is real: a genome that wins on half its
    # windows and loses on half is not a strategy, it is a coin.
    wins = sum(1 for e in ok if e.pnl > 0)
    hit_rate = wins / len(ok)

    # Consistency: how much of the return survives its own variance. Near 1 means
    # windows agree with each other; near 0 means they contradict.
    mean_abs = statistics.fmean(abs(e.pnl) / initial_cash for e in ok) if ok else 0.0
    consistency = 0.0 if mean_abs <= 0 else max(0.0, 1.0 - ddev / mean_abs)

    raw = (
        ret
        - LAMBDA_COST * cost_per_eval
        - LAMBDA_DRAWDOWN * dd
        - LAMBDA_TURNOVER * turnover
        - LAMBDA_HITRATE * (1.0 - hit_rate) * abs(ret)
    ) / (ddev + EPS)

    if not math.isfinite(raw):
        raw = 0.0

    # Inactivity penalty. A genome that never trades earns zero return, zero
    # drawdown and zero downside deviation, so it scores exactly 0 — which beats
    # every strategy that actually loses money. Left alone, the search reliably
    # converges on doing nothing, which looks like progress and is not.
    # Not trading is unproven, not optimal, so the penalty is additive: it has
    # to push an inactive genome *below* zero, not just to it.
    activity = trades / max(n, 1)
    deficit = max(0.0, 1.0 - activity / MIN_TRADES_PER_EPISODE)
    raw -= INACTIVITY_PENALTY * deficit

    fitness = raw * n_eff / (n_eff + PRIOR_STRENGTH)
    if not math.isfinite(fitness):
        fitness = 0.0

    return Fitness(
        raw=raw, fitness=fitness, ret=ret, cost_per_eval=cost_per_eval,
        drawdown=dd, downside_dev=ddev, turnover=turnover, n=n, n_eff=n_eff,
        trades=trades, n_truncated=n_truncated,
        hit_rate=hit_rate, consistency=consistency,
    )
