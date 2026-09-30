```
 _    _ _______      ________
| |  | |_   _\ \    / /  ____|
| |__| | | |  \ \  / /| |__
|  __  | | |   \ \/ / |  __|
| |  | |_| |_   \  /  | |____
|_|  |_|_____|   \/   |______|
```

# HIVE

An evolutionary harness for day-trading agents. **Fitness is realized net P&L from a
double-entry ledger.** A population of strategy genomes trades the same price windows,
the ledger records every fill, the profitable ones breed, the rest die, and the
population is scored again. At the end, one market nobody trained on decides whether
the winner is real.

```bash
PYTHONPATH=src python3 -m hive.cli --population 16 --generations 12 --episodes 6
```

No API key. No network. No dependencies. Python 3.11+ and the standard library.

> **Version:** 0.1.0 · **Branch:** `tui` · **Python:** ≥ 3.11 · **Dependencies:** none ·
> **Repo:** https://github.com/Narla7/hive (private) · **Image:** `hive:latest`

---

## Contents

- [What it is](#what-it-is)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Reading the output](#reading-the-output)
- [The holdout](#the-holdout)
- [Module map](#module-map)
- [Nine decisions that were bugs first](#nine-decisions-that-were-bugs-first)
- [Gates are structural](#gates-are-structural)
- [The TUI](#the-tui)
- [The LLM path](#the-llm-path)
- [CLI reference](#cli-reference)
- [Real bar data](#real-bar-data)
- [Docker](#docker)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Not financial advice](#not-financial-advice)

---

## What it is

A **genome** is a trading strategy expressed as data — lookback window, entry and exit
thresholds, stop loss, take profit, max hold bars, position fraction, cooldown, and a
style (`momentum`, `meanrev`, `breakout`). Not model weights, not a prompt blob. Ten
typed, bounded fields:

| Field | Default | Bound | What it does |
| --- | --- | --- | --- |
| `style` | `momentum` | `momentum` \| `meanrev` \| `breakout` | Which signal shape the rules decisioner computes |
| `lookback` | `20` | `[2, 240]` | Bars in the signal window |
| `entry_threshold` | `0.5` | `[0.01, 3.0]` | Signal magnitude required to go long |
| `exit_threshold` | `0.5` | `[0.01, 3.0]` | Signal magnitude below which a position is dropped |
| `stop_loss` | `0.02` | `> 0`, `< take_profit` | Fractional stop, e.g. `0.02` = 2% |
| `take_profit` | `0.04` | `> stop_loss` | Fractional target |
| `max_hold_bars` | `90` | `[1, 5000]` | Forced exit after this many decisions |
| `position_frac` | `0.5` | `[0, 1]` | Fraction of cash deployed on entry |
| `cooldown_bars` | `5` | `[0, 500]` | **Evolved but never read** — see [Known limitations](#known-limitations) |
| `allow_short` | `false` | — | **v0.2 field, unimplemented** — the broker is long-or-flat |

A population of those trades the same windows. Each is scored by what it actually made,
after fees, after slippage, after model spend. Top genomes breed by mutation and
crossover, losers are culled, a floor of fresh random genomes is injected every
generation, repeat. (The actual injection rate is much higher than 10% in practice —
[see why](#novelty-is-the-floor-not-a-fraction).)

**The LLM is not the source of the value.** It is only a decision-maker. The broker
executes and the ledger decides what happened. That split is why the harness runs with
zero API keys and still produces a real experiment, and why swapping in an LLM later
changes nothing about how results are scored — inference spend is posted to the ledger
as a cost *before* fitness is computed, so a strategy that is profitable but expensive
to run scores as unprofitable.

```python
Decision(target_weight=0.6, reason="entry momentum +0.812")   # the model proposes
  → gates.check_order(...)                                     # the harness constrains
  → broker.buy(bar, 0.6, ...)                                  # the broker executes
  → ledger.post(ts, "fill", ...)                               # the ledger decides
  → evaluate(episodes)                                         # fitness reads only this
```

---

## Quick start

Prerequisite check first — there is nothing to install, so this is the whole setup:

```bash
$ python3 --version
Python 3.14.7
```

A 7-second smoke test, small enough to read end to end:

```bash
$ PYTHONPATH=src python3 -m hive.cli --population 8 --generations 4 --episodes 3
model=offline/rules pop=8 gens=4 episodes=3 bars=720 cash=10,000
------------------------------------------------------------------------------
gen  0  mean_fitness   -0.1775  best a8c6f5909f4c raw  12.5578 -> shrunk   4.1859  ret +10.898%  dd 3.4%  cost/eval 0.0000  n_eff 3.0  new 5
gen  1  mean_fitness    2.6726  best d18757f377ae raw  18.5294 -> shrunk   6.1765  ret +13.807%  dd 3.5%  cost/eval 0.0000  n_eff 3.0  new 5
        winner delta: lookback: 60 -> 45
        winner delta: stop_loss: 0.025 -> 0.018
        winner delta: take_profit: 0.08 -> 0.055
        winner delta: max_hold_bars: 90 -> 150
        winner delta: cooldown_bars: 5 -> 10
gen  2  mean_fitness    3.6098  best d18757f377ae raw  18.5294 -> shrunk   6.1765  ret +13.807%  dd 3.5%  cost/eval 0.0000  n_eff 3.0  new 5
        winner delta: lookback: 60 -> 45
        winner delta: entry_threshold: 0.18 -> 0.35
        winner delta: exit_threshold: 0.2 -> 0.12
        winner delta: max_hold_bars: 90 -> 150
gen  3  mean_fitness    3.1042  best d18757f377ae raw  18.5294 -> shrunk   6.1765  ret +13.807%  dd 3.5%  cost/eval 0.0000  n_eff 3.0  new 5
------------------------------------------------------------------------------
mean fitness: -0.1775 -> 3.1042  (first/last third)  [IMPROVING]
holdout (unseen market seed, 5 windows):
  in-sample ret  +13.81%
  holdout ret    +38.15%   pnl +3,814.98  trades 69  shrunk 30.6289
  >> edge survives out-of-sample
best genome: d18757f377ae
  policy: {
    "style": "breakout",
    "lookback": 60,
    "entry_threshold": 0.18,
    "exit_threshold": 0.2,
    "stop_loss": 0.018,
    "take_profit": 0.055,
    "max_hold_bars": 90,
    "position_frac": 0.45,
    "cooldown_bars": 10,
    "allow_short": false
  }
```

Mean fitness moved from **−0.1775 to 3.1042** and the out-of-sample check passed. On
this machine that took `real 0m7.451s`.

The full recommended run (`--population 16 --generations 12 --episodes 6`) took
**1m25.048s** and reached `mean fitness: 14.2898 -> 21.4831  [IMPROVING]` with
`in-sample ret +43.37%` / `holdout ret +24.30%` over 5 unseen windows and 107 trades.
Read [Honest limits on the numbers](#honest-limits-on-the-numbers) before quoting either.

Full-screen dashboard, if you have a terminal:

```bash
PYTHONPATH=src python3 -m hive.cli --tui
```

---

## Requirements

| Requirement | Minimum | Why | Check |
| --- | --- | --- | --- |
| Python | 3.11 | `dataclass(slots=True)`, `X \| Y` type syntax, `datetime.fromisoformat` on offsets | `python3 --version` |
| Standard library | — | That is the entire dependency list (`pyproject.toml`: `dependencies = []`) | — |
| API key | none | Only for `--model <id>`; `offline/rules` is the default | — |
| Network | none | Only for `--model <id>` or `--probe` | — |
| Terminal | 80×24 | Only for `--tui`; refuses to start under 40×8 or without a TTY | `stty size` |
| Docker | any | Only for `docker compose`; the image is `python:3.14-slim` | `docker --version` |

Confirmed: `pyproject.toml` declares `requires-python = ">=3.11"`, `dependencies = []`,
and `[project.scripts] hive = "hive.cli:main"`. The Docker image is
`python:3.14-slim`, stdlib-only, running as a non-root user named `agent`.

---

## Installation

**Recommended — no install.** `PYTHONPATH=src` is the whole story, and it is how the
tests and the Docker image both run:

```bash
PYTHONPATH=src python3 -m hive.cli --help
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

**Console script — if you want `hive` on your PATH:**

```bash
$ python3 -m venv .venv && . .venv/bin/activate
$ pip install -e .
$ hive --help
usage: hive [-h] [--population POPULATION] [--generations GENERATIONS] ...
$ python -c "import hive; print(hive.__version__)"
0.1.0
```

Verified: a clean venv install produces a working `hive` console script that imports
the same package and prints `0.1.0`.

**Docker:**

```bash
docker compose build          # tags hive:latest
docker compose run --rm test
docker compose run --rm evolve
```

**Running it as a library:**

```python
from hive import Config, SimulatedMarket, GateConfig, run   # re-exported from hive/__init__.py
from hive.decisioners import RulesDecisioner                  # not re-exported
from hive.evolution import holdout                           # not re-exported

cfg = Config(population=12, generations=8, episodes=3, bars=720, market_seed=1234)
reports, best = run(cfg, market=SimulatedMarket(seed=1234),
                    decisioner=RulesDecisioner(), gates_cfg=GateConfig(max_drawdown=0.25))
print(reports[-1].mean_fitness, best.id, best.policy.style)

h = holdout(best, cfg, RulesDecisioner(), market_seed=cfg.market_seed + 9999, n_windows=5)
print(f"out-of-sample {h.ret:+.2%} over {h.n_windows} unseen windows, {h.trades} trades")
```

```console
$ PYTHONPATH=src python3 example.py
1.128810425167529 b7a7fa1e6dba breakout
out-of-sample +24.30% over 5 unseen windows, 107 trades
```

`hive/__init__.py` re-exports `BrokerConfig`, `PaperBroker`, `Config`, `run`,
`Episode`, `Fitness`, `evaluate`, `GateConfig`, `Gates`, `Genome`, `Policy`,
`hand_seeded`, `mutate`, `crossover`, `Ledger`, `Bar`, `SimulatedMarket`,
`FileMarket`. `RulesDecisioner`, `LLMDecisioner` and `holdout` are not in that list —
import them from `hive.decisioners` and `hive.evolution`.

---

## How it works

```
                     ┌──────────────────────────────────────────────┐
  --data CSV ───────►│  market.py   Bar[]                           │
  (or --market-seed) │  SimulatedMarket: seeded, regime-switching  │
                     │  FileMarket:    CSV, one Market interface   │
                     └───────────────────┬──────────────────────────┘
                                         │  one window per EPISODE,
                                         │  the SAME set for every genome
                     ┌───────────────────▼──────────────────────────┐
  genome  ──────────►│  decisioners.py   Genome + Bar[] + AgentState│
  (10 fields)        │      → Decision(target_weight, reason, cost) │
                     │  RulesDecisioner (offline) | LLMDecisioner  │
                     └───────────────────┬──────────────────────────┘
                                         │
                     ┌───────────────────▼──────────────────────────┐
                     │  gates.py   check_order(cooldown, position)  │  ← structural,
                     │             admit(genome) before capital     │    never asked
                     │             check_drawdown() after each ep   │    of the model
                     └───────────────────┬──────────────────────────┘
                     ┌───────────────────▼──────────────────────────┐
                     │  broker.py  buy/sell: adverse slippage that  │
                     │             scales with participation in    │
                     │             bar volume, spread, 5bps/side    │
                     └───────────────────┬──────────────────────────┘
                     ┌───────────────────▼──────────────────────────┐
                     │  ledger.py  append-only double-entry,       │
                     │             REJECTS an unbalanced Entry     │
                     │  cash · position · realized_pnl · fees      │
                     │  inference_cost · equity                     │
                     └───────────────────┬──────────────────────────┘
                     ┌───────────────────▼──────────────────────────┐
                     │  fitness.py  ret − λ₀·cost − λ₁·dd           │  ← pure function
                     │              − λ₂·turnover                   │     of Episode[]
                     │              ─────────────────────           │     only
                     │              / (downside_dev + 1e-4)         │
                     │              − 1.5·inactivity_deficit         │
                     │              × n_eff/(n_eff + 6)   ← shrunk  │
                     └───────────────────┬──────────────────────────┘
                     ┌───────────────────▼──────────────────────────┐
                     │  evolution.py  rank → breed (mutate/         │
                     │                crossover) → cull → inject    │
                     │                ≥1 novel → next generation   │
                     │                …then holdout() on a new seed │
                     └──────────────────────────────────────────────┘
```

Everything in the ledger column is what makes the rest auditable. `fitness.py` imports
no clock, no network, and no adapter state; it reads `list[Episode]` and returns a
`Fitness`. Selection, culling and the holdout are all built on that one function, which
is why it is the part with the most regression tests.

### The loop, per generation

1. **Admit.** `gates.admit(genome)` — a genome missing a stop is rejected before it
   ever touches capital and scores `0` with a recorded violation.
2. **Evaluate.** Every genome runs `cfg.episodes` episodes over `windows[i]`.
3. **Mark and close flat.** Every session ends with a forced sell, so P&L is realized
   rather than left as an open position that flatters whichever genome happened to
   end long.
4. **Gate.** `gates.check_drawdown()` on the realized equity curve. A breach quarantines
   the genome and it is excluded from breeding.
5. **Rank** by `fitness.fitness` — the *shrunk* score, not `raw`.
6. **Breed.** Top `cfg.elite` survive. `n_new = max(1, pop × 0.10)` fresh random
   genomes; `int(pop × 0.30)` crossovers; the remainder fill with `mutate(survivor,
   strength=0.8)`.
7. **Report** and call `on_generation(report)`. Returning `False` stops the loop.

Fitness is compared over **thirds**, not endpoints: the first third of generations is
averaged against the last third. A single noisy generation at the end can flatten or
wreck a verdict on a stochastic objective.

### Scoring constants, verbatim from `fitness.py`

```python
LAMBDA_COST          = 0.35      # λ₀, on inference cost per evaluation
LAMBDA_DRAWDOWN      = 1.0       # λ₁
LAMBDA_TURNOVER      = 0.0002    # λ₂ — a tiebreaker, not a cost
PRIOR_STRENGTH       = 6.0       # k in n_eff/(n_eff + k)
EPS                  = 1e-4
MIN_DOWNSIDE_DEV     = 0.005     # 0.5% of equity
MIN_TRADES_PER_EPISODE = 2.0
INACTIVITY_PENALTY   = 1.5       # additive, raw-score units
```

```
raw     = (ret − 0.35·cost_per_eval − 1.0·drawdown − 0.0002·turnover) / (downside_dev + 1e-4)
raw    −= 1.5 · max(0, 1 − trades_per_episode/2)
fitness = raw · n_eff/(n_eff + 6)          ← this is what selection uses
```

`ret` is on **account equity** (`Σ episode pnl / initial_cash`), never on deployed
capital. Normalizing by deployed capital deflates every high-turnover strategy and
hides the fact that churning costs money. "How much did this make on the account" is
the only number that answers the question being asked.

---

## Reading the output

```
gen  6  mean_fitness   22.0201  best 020965b193e6 raw  75.0117 -> shrunk  28.2551  ret +43.931%  dd 3.5%  cost/eval 0.0000  n_eff 3.6  new 13
```

| Field | Meaning | Read it for |
| --- | --- | --- |
| `mean_fitness` | Population mean of the **shrunk** score | Whether the whole population is moving, not just the winner |
| `best <id>` | 12-hex fingerprint of the winning policy | Stable identity — same policy always hashes the same |
| `raw` | Risk-adjusted score before shrinkage | The raw signal |
| `shrunk` | `raw × n_eff/(n_eff+6)` — **selection uses this** | How much the evidence gap is costing |
| `ret` | Realized net P&L on account equity, after fees and slippage | The number a person cares about |
| `dd` | Max drawdown of the realized per-bar equity curve | Not summary stats — the actual path |
| `cost/eval` | Inference spend per decision | `0.0000` offline; the LLM's real spend when a model is driving |
| `n_eff` | Effective sample size, autocorrelation-corrected | Real evidence, which is less than the episode count |
| `new` | Children injected this generation | `13` of 16 is high — see [Novelty is the floor, not a fraction](#novelty-is-the-floor-not-a-fraction) |
| `quarantined` | Genome ids tripped a gate | Printed on its own line under the generation |

**`raw` vs `shrunk` is the single most important column pair.** With the default
`--episodes 3` and `PRIOR_STRENGTH = 6.0`, a genome with `n_eff = 3` is scaled by
`3/9 = 0.333`. A spectacular raw score off three episodes gets shrunk to a third, not
rewarded. Without it the loop latches onto early noise, hands the winner more budget,
and mistakes the resulting sample growth for confirmation — and that failure is
invisible from outside, because the loop looks busy and reports confident winners.

**`n_eff` corrects for serial correlation.** Episode returns cluster: winning
strategies win repeatedly, broken ones repeat the same mistake. `effective_n()` sums
autocorrelations out to lag 20 and returns `n / (1 + 2·Σρ)`, clamped to `[1, n]`. The
raw episode count overstates the evidence and defeats the shrinkage term, which is
exactly the protection being sought.

`--json` emits the same thing machine-readably:

```bash
PYTHONPATH=src python3 -m hive.cli --population 6 --generations 3 --episodes 2 \
    --quiet --no-holdout --json
```
```json
{
  "verdict": "IMPROVING",
  "generations": [
    {
      "generation": 0,
      "mean_fitness": -3.104209647984369,
      "best_id": "8f74e0d29d38",
      "best_raw": 4.298260502858841,
      "best_fitness": 1.0745651257147102,
      "best_ret": 0.03910358994438029,
      "best_drawdown": 0.013135756450167829,
      "n_eff": 2.0,
      "quarantined": []
    },
    {
      "generation": 1,
      "mean_fitness": 0.47742449514501045,
      "best_id": "8f74e0d29d38",
      "best_raw": 4.298260502858841,
      "best_fitness": 1.0745651257147102,
      "best_ret": 0.03910358994438029,
      "best_drawdown": 0.013135756450167829,
      "n_eff": 2.0,
      "quarantined": []
    },
    {
      "generation": 2,
      "mean_fitness": 0.7048767851804941,
      "best_id": "302b5ee6d504",
      "best_raw": 6.561464799336361,
      "best_fitness": 1.6403661998340902,
      "best_ret": 0.04900932281404348,
      "best_drawdown": 0.01233109858123134,
      "n_eff": 2.0,
      "quarantined": []
    }
  ],
  "best_genome": {
    "policy": {
      "style": "momentum",
      "lookback": 84,
      "entry_threshold": 0.4163,
      "exit_threshold": 0.24,
      "stop_loss": 0.03117,
      "take_profit": 0.14396,
      "max_hold_bars": 298,
      "position_frac": 0.1066,
      "cooldown_bars": 13,
      "allow_short": false
    },
    "model": "offline/rules",
    "id": "302b5ee6d504",
    "parent": [],
    "generation": 2,
    "notes": ""
  }
}
```

`--out DIR` writes `best_genome.json` and `report.json` there. Those two files are the
*only* things this repository writes to disk.

---

## The holdout

```
holdout (unseen market seed, 5 windows):
  in-sample ret  +43.37%
  holdout ret    +24.30%   pnl +2,430.29  trades 107  shrunk 15.5295
  >> edge survives out-of-sample
```

Selection over a handful of fixed price windows **always** produces a winner. In-sample
return says nothing about whether the strategy is real or fitted to those windows.
`holdout()` re-runs the selected genome against `SimulatedMarket(seed=market_seed +
9999)` — a seed the search never saw — over 5 windows, and the CLI prints the verdict
in those words:

| Condition | Printed verdict |
| --- | --- |
| `in-sample > 0` and `holdout <= 0` | `>> OVERFIT: profitable in-sample, not out-of-sample` |
| `holdout > 0` | `>> edge survives out-of-sample` |
| `in-sample == 0`, `holdout > 0` | `>> edge survives out-of-sample` — **misleading, see [Known limitations](#known-limitations)** |

Without this the loop is unfalsifiable. That is the whole reason it is a hard part of
the pipeline and not an afterthought.

### Honest limits on the numbers

The default market is synthetic and **deliberately has exploitable structure**:
persistent trend regimes (`regime_prob = 0.02`, 20–90 bar stretches), mean reversion
pulled toward a sinusoidal session anchor, and occasional jumps (`jump_prob = 0.004`,
8× volatility). A pure random walk has no edge to find, so any fitness-increasing
experiment against one would be measuring nothing. The consequence is that the reported
returns are **much** higher than real day trading produces. `+43.37%` in-sample and
`+24.30%` out-of-sample are artifacts of a friendly simulator.

What the default run demonstrates is that **the harness is correct**: the loop
improves, the ledger balances, the shrinkage term bites, the gates fire, and the edge
survives out-of-sample. It does not demonstrate a profitable strategy. Point it at real
data — correctly, per [Real bar data](#real-bar-data) — before concluding anything
about markets.

---

## Module map

| Module | Lines | Role |
| --- | --- | --- |
| `market.py` | 149 | `Bar`, `Market` interface, `SimulatedMarket` (seeded, regime-switching), `FileMarket` (CSV) |
| `broker.py` | 185 | `PaperBroker` — the only place orders execute. Adverse slippage scaling with volume participation, spread, 5 bps/side |
| `ledger.py` | 144 | Append-only double-entry. `post()` **raises `ValueError`** on any entry that does not sum to zero |
| `genome.py` | 214 | `Policy`, `Genome`, `fingerprint()`, `diff()`, `mutate()`, `crossover()`, `random_genome()`, `hand_seeded()` |
| `fitness.py` | 177 | `Episode`, `Fitness`, `evaluate()`, `max_drawdown()`, `effective_n()`, `downside_dev()`. Pure — no clock, no network |
| `gates.py` | 101 | `GateConfig`, `Gates`: `admit()`, `check_order()`, `check_drawdown()`, `check_daily_loss()`, `kill_switch()` |
| `decisioners.py` | 426 | `RulesDecisioner` (offline), `LLMDecisioner` (OpenAI-compatible + Anthropic), `CallBudget`, `probe()`, cost table |
| `evolution.py` | 398 | `Config`, `run_episode()`, `run()`, `holdout()`, `allocate()` (UCB, defined but unwired) |
| `render.py` | 233 | `TuiState`, `build_frame()`. Returns `[(text, colour)]` rows. **Imports no curses** |
| `theme.py` | 238 | Palette, sigil, `sparkline()`, `bar()`, `corrupt()`, `Flicker`, `ritual_line()` |
| `tui.py` | 303 | Curses driver. Worker thread + `Control` flags. The only file that imports `curses` |
| `cli.py` | 266 | `argparse`, report formatting, `--probe`, `--json`, `--out`, `--tui` dispatch |

3,884 lines total: 2,855 in `src/hive/`, 1,029 in `tests/`.

Three interfaces, each deliberately narrow:

| Interface | Implementations | Swapping it changes |
| --- | --- | --- |
| `Market.bars(symbol, start, end)` | `SimulatedMarket`, `FileMarket` | The price data. Nothing else |
| `Decisioner.decide(genome, bars, state)` | `RulesDecisioner`, `LLMDecisioner` | Who proposes. Scoring is unchanged |
| `PaperBroker` | paper only | What fills the ledger. Nothing about the algorithm |

---

## Nine decisions that were bugs first

Each of these is a specific thing that broke, got diagnosed, and got a regression test.
The test names are in the file — search for them.

### 1. Fitness cannot reward inaction

A genome that never trades has zero return, zero drawdown and zero downside deviation,
so it scores exactly `0` — which **beats every strategy that actually loses money**. The
first working version converged on doing nothing and reported it as progress. Mean
fitness sat at `0.0` and looked stable.

The fix is an additive penalty, `INACTIVITY_PENALTY = 1.5`, scaled by how far below
`MIN_TRADES_PER_EPISODE = 2.0` trades/episode the genome falls. Additive, not
multiplicative, because it has to push an inactive genome *below* zero, not merely down
to it. The three-way ordering, straight from the module:

```console
$ PYTHONPATH=src python3 -c '
import hive.fitness as F
ep = lambda pnl, t: F.Episode(pnl=pnl, equity_curve=[10000.0, 10000.0+pnl],
                              capital_deployed=10000.0, inference_cost=0.0,
                              n_evals=10, turnover=5.0, n_trades=t)
for label, pnl, t in (("inactive", 0.0, 0), ("small loser", -50.0, 5), ("profitable", 300.0, 5)):
    f = F.evaluate([ep(pnl, t)]*3, 10_000.0)
    print(f"{label:>12}  raw {f.raw:>8.4f}  fitness {f.fitness:>8.4f}  n_eff {f.n_eff}")
'
    inactive  raw  -1.5001  fitness  -0.5000  n_eff 3.0
 small loser  raw  -3.9216  fitness  -1.3072  n_eff 3.0
 profitable  raw  17.6470  fitness   5.8823  n_eff 3.0
```

Read the ordering carefully, because it is the whole design. Inactivity (`-0.5000`) is
still *above* a small loser (`-1.3072`) — a genome that refuses to trade is unproven
rather than disqualified, and that is intentional. What the penalty changes is the sign:
without it, inactive scores exactly `0.0`, which is the same score as "no evidence at
all", sits at the top of a field of money-losers, and produces a population mean that
looks flat and stable while nothing happens. With it, inaction is unambiguously negative,
so a run whose mean fitness never lifts off zero is visibly a failed run rather than a
quiet one.

> **Naming caveat:** the test `test_inactive_scores_below_a_small_loser` does not assert
> what its name says. Its body is `assertGreater(inactive.fitness, loser.fitness)` — it
> pins inactive *above* the loser, the same relationship shown above. The test that
> actually guards the sign is `test_inactive_genome_scores_below_zero`; the third is
> `test_profitable_scores_above_inactive`.

### 2. Trading frictions are charged once

`LAMBDA_TURNOVER = 0.0002` — two ten-thousandths, deliberately tiny. Fees and slippage
are **already in the ledger**, charged per side by the broker. An earlier version used a
meaningful turnover weight, which double-counted them and crushed exactly the
low-churn strategies that survive costs. It is a tiebreaker, not a cost.

### 3. Downside deviation is floored at 0.5% of equity

Dividing by a near-zero denominator amplifies sampling noise into enormous raw scores.
A genome making −2.7% with almost no variance was scoring `raw = −122` — a magnitude
several times the entire spread of genuine winners in the same generation, and a
meaningless number, because the divisor was sampling noise rather than risk.
`MIN_DOWNSIDE_DEV = 0.005` is the floor, so the denominator is never smaller than half
a percent of equity no matter how quiet the equity curve is.

### 4. Episodes must use different market windows

Running three episodes over **one** fixed price path with a deterministic decisioner
returns three identical results. `n_eff` is then meaningless — three copies of one
number are not three samples — and the search can quietly overfit a single price series
while reporting a healthy `n_eff`. Every episode now gets its own window, and every
genome in a generation sees the same set, so fitness stays comparable:

```python
windows = [market.bars(symbol,
                       start + timedelta(minutes=i * cfg.episode_spacing),   # 900
                       end   + timedelta(minutes=i * cfg.episode_spacing))
           for i in range(cfg.episodes)]
```

With the defaults (`--bars 720`, spacing 900 min, `--episodes 6`) the windows are
`2026-01-05T14:30 → 2026-01-06T02:30`, then `01-06T05:30 → 01-06T17:30`, then
`01-06T20:30 → 01-07T08:30`, and so on. They are spaced wider than they are long on
purpose: an overlapping window would re-use the same bars across episodes and
reintroduce the correlation that `n_eff` is supposed to be measuring.

### 5. Mutation self-repairs reward/risk

`stop_loss` and `take_profit` are jittered independently, so mutation could invert the
relationship and emit an invalid child. A variation operator that produces garbage half
the time burns search budget for nothing. After jitter:

```python
if p.take_profit <= p.stop_loss:
    p.take_profit = round(p.stop_loss * rng.uniform(1.1, 4.0), 5)
```

Crossover inherits each parent's caps rather than averaging them, for the same reason —
averaging a 1% stop with a 20% stop is how a child ends up with no stop at all. Test:
`test_mutation_preserves_validity` (25 mutations × 7 seeds at `strength=2.0`),
`test_crossover_keeps_caps`.

### 6. Gaps are floored, not clamped away

`PRIOR_STRENGTH = 6.0` shrinks toward zero in proportion to how little evidence a
genome has, and `effective_n()` returns at least `1.0` and at most `n`. A genome that
has never been evaluated returns `Fitness(0.0, 0.0, …)` from `evaluate([])` rather than
raising, so a quarantined genome still has a score the sort can handle.

### 7. Float dust is not a rejection

`PaperBroker._can_afford()` uses a *relative* tolerance. An exact comparison rejected
every fill, because `10000.000000000002 > 10000.0`. Test:
`test_buy_respects_cash_with_float_dust`.

### 8. The evolution loop knows nothing about curses

`run()` takes an `on_generation: Callable[[GenerationReport], bool]` callback and
breaks when it returns `False`. That is the *entire* interface between the TUI and the
loop. The TUI, the CLI and the tests all drive the same loop with equal ease, and
`evolution.py` contains no import of `curses`, no thread, and no key handling. The
callback fires **after** the report is built, so a UI sees exactly the data the final
summary prints.

### 9. Truncated episodes are excluded, not flagged

When `--max-calls-per-epoch` runs out mid-episode, the episode is dropped from fitness
entirely:

```python
ok = [e for e in episodes if e.error is None and not e.truncated]
```

A half-length episode understates a genome because it had less time to trade. Scoring it
would bias selection toward strategies that churn fast — the exact opposite of what
survives costs. The count is still reported (`n_truncated`) so "everything was
truncated" stays distinguishable from "nothing ran", and the CLI prints a loud
degradation notice. Tests: `test_truncated_excluded_from_fitness`,
`test_all_truncated_scores_zero`, `test_truncation_counted_reported`.

### Novelty is the floor, not a fraction

`n_new = max(1, int(cfg.population * cfg.novelty_frac))` — the 10% novel injection is a
*minimum* of one fresh genome per generation, so a small population does not stagnate.
But crossover and mutation then pad the rest of the roster, so the actual `new` count in
the report is much higher than 10%: in the `--population 16` run it is `13` of 16 every
generation, because only the top `--elite 3` survive. Selection pressure is therefore
tight, and exploration comes almost entirely from random genomes rather than from
crossover. This is a real property of the current loop, not a bug, but it is worth
knowing before you interpret a winner's lineage.

---

## Gates are structural

Gates are checked by the harness, between the decision and the order. A genome that
breaches a drawdown cap is quarantined; it does not get to argue its way past by
out-earning the limit. **A gate enforced by a model is a suggestion.**

| Gate | Default | Enforced at | Wired into the loop? |
| --- | --- | --- | --- |
| `admit()` — valid policy, adapter allowlist, `position_frac` cap | adapter `{"sim"}` | Before any capital | ✅ every generation |
| `check_order()` — `max_position_frac` | `1.0` | Every bar, before the fill | ✅ live (`--max-drawdown` is the only one on the CLI) |
| `check_order()` — `cooldown_bars` | `0` | Every bar | ⚠️ inert at the default |
| `check_drawdown()` | `0.25` | End of each episode | ✅ `--max-drawdown` |
| `check_daily_loss()` | `0.05` | — | ❌ never called |
| `kill_switch()` | — | — | ❌ never triggered; `gates.tripped` is read but nothing sets it |

Quarantine excludes a genome from **breeding**; it does not stop it being evaluated, so
a quarantined id still appears in `mean_fitness` and in the quarantine list. Verified
with a deliberately tight cap:

```bash
$ PYTHONPATH=src python3 -m hive.cli --population 6 --generations 3 --episodes 2 --max-drawdown 0.01
gen  0  mean_fitness   -3.1042  best 8f74e0d29d38 raw   4.2983 -> shrunk  1.0746  ret +3.910%  dd 1.3%  cost/eval 0.0000  n_eff 2.0  new 3
        quarantined: 35ab66ec8f9a, 8f74e0d29d38, 986e4d30f80d, afcf02b78f30, b7a7fa1e6dba, f170d1056d25
gen  1  mean_fitness   -1.8054  best b7a7fa1e6dba raw   1.7847 -> shrunk  0.4462  ret +3.843%  dd 2.1%  cost/eval 0.0000  n_eff 2.0  new 5
        quarantined: 35ab66ec8f9a, 8f74e0d29d38, 986e4d30f80d, afcf02b78f30, b7a7fa1e6dba, d18757f377ae, f170d1056d25
gen  2  mean_fitness   -0.1938  best 302b5ee6d504 raw   6.5615 -> shrunk  1.6404  ret +4.901%  dd 1.2%  cost/eval 0.0000  n_eff 2.0  new 3
        quarantined: 302b5ee6d504, 35ab66ec8f9a, 8f74e0d29d38, 986e4d30f80d, afcf02b78f30, b7a7fa1e6dba, d18757f377ae, f170d1056d25
```

Every genome breaches at once, because `--max-drawdown 0.01` is 1% and a normal minute
of volatility clears that. That is the systemic case, and it is why a separate
population-wide kill switch exists as an API — it is simply not wired to a trigger yet.

`Ledger.post()` is the same idea one level down. It raises rather than accepting a fill
that does not balance:

```console
$ PYTHONPATH=src python3 -c "
from datetime import datetime
from hive.ledger import Ledger, Posting, CASH, POSITION
led = Ledger()
try:
    led.post(datetime(2026,1,5,14,30), 'fill', 'SIM', [Posting(CASH,-1000.0), Posting(POSITION,990.0)])
except ValueError as e:
    print('ValueError:', e)
"
ValueError: unbalanced entry at 2026-01-05T14:30:00 (fill): off by -10.0000000000
```

A sell posts three lines — cash proceeds, position cost basis, realized P&L — because
without that third posting the entry is unbalanced and the ledger refuses it. Test:
`test_rejects_unbalanced_entry`.

---

## The TUI

```bash
PYTHONPATH=src python3 -m hive.cli --tui
```

Full-screen curses dashboard. Rendered frame at 100×40 with real numbers from the
`--population 16 --generations 12 --episodes 6` run above (colour stripped):

```
════════════════════════════════════════════════════════════════════════════════════════════════════
                                        __  _______    ________
                                       / / / /  _/ |  / / ____/
                                        / /_/ // / | | / / __/
                                       / __  // /  | |/ / /___
                                      /_/ /_/___/  |___/_____/
                                              H I V E
                              THE SWARM // EVOLUTIONARY PROFIT MACHINE

 THE QUEEN  gen 6/12                             │  VITALS
──────────────────────────────────────────────── │ ─────────────────────────────────────────────────
 entity           fitness   return               │  vitality   ███████████████████████ +23.0248
 6d3bf3c97f91        33.383  +43.37%             │  peak       ███████████████████████ +33.3800
 37c2e2f97ed2        32.581  +42.37%             │  trace      ▁▅▇▆▇▇▇▇▇▇▆█
 b7a7fa1e6dba        31.849  +41.55%             │  omen  RISING
 020965b193e6        28.928  +41.09%             │  breath +23.025 over 12 rites
 302b5ee6d504        27.559  +41.91%             │  sealed 1 entities
 d18757f377ae        27.632  +41.62%             │ holdout +24.30% -- it survives
 35c99163139b        12.190  +19.02%             │  WHISPERS
 a8c6f5909f4c         4.186  +10.90%             │ ─────────────────────────────────────────────────
 │  the chamber wakes
 │  it learned your thresholds
 │  a new queen is born
 │  3 things were born
 │  the other market says +24.30%
 │
 │
 │
 │
 │
 │
 │
 │
 │
 │
 │

 offline/rules  x1                                                            IT IS STILL TRADING
                       q quit   space pause   +/- speed   n next   s summary
```

- **`THE QUEEN`** is the population, ranked by shrunk fitness. Green means positive.
- **`VITALS`** carries the two gauges, the sparkline trace of mean fitness, the omen
  (`AWAKENING` / `RISING` / `STAGNANT`, same thirds comparison as the CLI verdict), and
  conditional rows for sealed genomes, truncated episodes, parse failures and the
  holdout result.
- **`WHISPERS`** is a 200-line bounded log of bee/swarm one-liners from
  `theme.RITUAL_LINES`, coloured by whether the generation's mean was positive.

| Key | Effect |
| --- | --- |
| `q` | Quit (and `Q`) |
| `space` | Pause — the loop blocks at the next generation boundary in `Control.wait_if_paused()` |
| `+` / `=` | Speed up, doubling to a maximum of **8.0×** |
| `-` / `_` | Slow down, halving to a minimum of **0.25×** |
| `n` / `N` | Stop waiting — unpause and run to the end |
| `s` / `S` | Re-run the out-of-sample check on the current winner |

Three details that are deliberate:

- **The evolution loop knows nothing about curses.** It calls `on_generation` and checks
  whether the callback wants it to stop. See
  [decision 8](#8-the-evolution-loop-knows-nothing-about-curses).
- **It refuses to claim a completed rite over a partial run.** `completed` is only true
  when `len(reports) >= cfg.generations`. Quit early and the log says
  `the rite was abandoned`; fall short and it says `stopped at N of M rites`; crash and
  it says `the rite broke` and returns exit code 1. An earlier version cheerfully printed
  "the rite is complete" over a population that never finished, and silently skipped the
  holdout.
- **Layout degrades by terminal size.** The full 5-line figlet sigil needs ≥ 24 rows
  (`len(sigil) + 5 <= height - 14`); below that a 3-line compact sigil is used. Under
  40×8 the screen just says `terminal too small`. The renderer is a pure function of
  `(TuiState, width, height)` returning `[(text, colour)]` rows, which is why all 38 TUI
  tests run without a terminal.

The loop runs on a **worker thread** (`threading.Thread(target=worker, daemon=True)`) so
key handling stays responsive during slow LLM generations. Curses is only ever touched
from the main thread. The worker catches `BaseException`, not `Exception` — a
`KeyboardInterrupt` would otherwise skip the handler, run the `finally`, and leave the
result empty while the UI claimed it finished.

---

## The LLM path

Optional. `RulesDecisioner` is the default and the reason the harness can be run and
tested anywhere.

```bash
export HIVE_API_KEY=sk-...
PYTHONPATH=src python3 -m hive.cli --model anthropic/claude-sonnet \
    --base-url https://openrouter.ai/api/v1 --population 8 --generations 5
```

`LLMDecisioner` speaks the OpenAI `chat/completions` shape by default and Anthropic's
`Messages` API with `--provider anthropic`. It sends:

```
POST {base_url}/chat/completions          (or {base_url}/v1/messages for anthropic)
Authorization: Bearer {key}               (or x-api-key + anthropic-version: 2023-06-01)
User-Agent: hive/0.1 (+https://github.com/Narla7/hive)
Content-Type: application/json

{"model": ..., "messages": [system, user], "temperature": 0.0, "seed": 0}
```

The system prompt asks for exactly one JSON object, `{"target_weight": float, "reason":
str}`, no prose. The parser slices from the first `{` to the last `}` and clamps
`target_weight` to `[-1, 1]` — so a chatty model still works, and a model that returns
prose still gets counted as a failure rather than silently holding position.

### Probe first, always

```bash
$ PYTHONPATH=src python3 -m hive.cli --probe --model big-pickle --base-url https://opencode.ai/zen/v1
error: no API key: set HIVE_API_KEY, OPENROUTER_API_KEY or OPENCODE_API_KEY, or pass --api-key. Use --model offline/rules to run with no key at all.
$ echo $?
2
```

With a key set, `--probe` makes **one** call against a 12-bar synthetic series with a
gentle trend, prints the endpoint, status, latency, token counts, estimated cost, the
first 300 characters of the raw reply, and the parsed object — then exits.

This exists because a full run fires thousands of calls. Without it, a bad key or a
model that returns prose is discovered minutes and real money later — *or not at all*,
because an unparseable reply degrades to "hold position" and every genome quietly
behaves identically. The probe makes that failure loud and costs essentially nothing.
Exit code is `0` on a parseable reply, `1` on any HTTP or parse failure, `2` on a
missing key.

### Counting calls, because the full default run is not cheap

A decision happens every `--decide-every` bars (default 3) on a `--bars 720` episode,
minus the first two bars: **239 decisions per episode**, verified from `run_episode`.

| Run | Calls |
| --- | --- |
| `--population 8 --generations 4 --episodes 3` | 8 × 3 × 239 × 4 = **22,944** |
| `--population 16 --generations 12 --episodes 6` | 16 × 6 × 239 × 12 = **275,328** + 1,195 holdout = **276,523** |
| `--population 4 --generations 2 --episodes 2` (the run below) | 4 × 2 × 239 × 2 = **3,824** — the `inference:` line confirms it |

The last row is a real measured run, not arithmetic: its `inference:` counter read
`3824 calls` exactly.

`--max-calls-per-epoch N` is the lever. The budget is **reset every generation**, so
the cap is per-epoch rather than cumulative; when it runs out the decisioner returns
`budget_exhausted`, `run_episode` marks the episode `truncated`, and `evaluate()` drops
it from fitness. Genuine output, from a real run against a stubbed endpoint with
`--max-calls-per-epoch 900`:

```console
$ PYTHONPATH=src python3 -m hive.cli --model stub/flash --base-url https://example.invalid/v1 \
    --population 4 --generations 2 --episodes 2 --no-holdout --max-calls-per-epoch 900
model=stub/flash pop=4 gens=2 episodes=2 bars=720 cash=10,000
------------------------------------------------------------------------------
gen  0  mean_fitness   -2.5914  best b7a7fa1e6dba raw   0.0000 -> shrunk   0.0000  ret +0.000%  dd 0.0%  cost/eval 0.0000  n_eff 0.0  new 2
gen  1  mean_fitness   -2.5914  best f170d1056d25 raw   0.0000 -> shrunk   0.0000  ret +0.000%  dd 0.0%  cost/eval 0.0000  n_eff 0.0  new 2
------------------------------------------------------------------------------
mean fitness: -2.5914 -> -2.5914  (first/last third)  [NOT IMPROVING]

inference: 1800 calls  $1.2276  parse_failures=0  network_errors=0

>> 4 episode(s) hit --max-calls-per-epoch and were
>> excluded from fitness. The verdict above is degraded.
```

Note `n_eff 0.0` and `ret +0.000%` on the winner: every episode it had was truncated, so
it has no evidence at all. The mean is `-2.5914` rather than `0.0` because the *other*
genomes did complete an episode and lost money. Do not read `raw 0.0000` as "neutral" —
read it as "not measured".

`--max-calls-per-epoch` has **no effect on `offline/rules`**, because `RulesDecisioner`
has no `budget` attribute and `run_episode` reads it with `getattr(decisioner,
"budget", None)`.

### Failure counters

Every run reports, when a model is driving, and says so in those words when the
counters are not zero. Genuine output from a real run against a stubbed endpoint where
every ninth reply was prose:

```console
inference: 3824 calls  $2.6080  parse_failures=424  network_errors=0
  >> WARNING: 424 of 4248 decisions were unparseable.
  >> Fitness is NOT measuring strategy quality. Fix the prompt
  >> or the parser before trusting any number above.
```

> **The warning's denominator is wrong.** It reads
> `f"{pf} of {calls + pf} decisions were unparseable"`, but `LLMDecisioner.decide`
> increments `self.calls` *before* attempting the parse — unparseable replies are
> already inside `calls`. So `4248` is really `3824 + 424`, and the true rate is
> `424 / 3824 ≈ 11%`, not the `10%` the line implies. Trust `parse_failures` over the
> fraction.

The other two warnings are `>> WARNING: {ne} network errors. Results are partial.` and
`>> WARNING: zero model calls were made. Check --model.`

These counters exist because a silent parse failure is indistinguishable from a market
with no edge. Both look exactly like a genome that never trades.

### What the probe found about OpenCode Zen

Four things, all of which cost nothing to discover now and thousands of calls to discover
later.

- **Free models are in-app only.** `big-pickle` and every `*-free` model return
  `FreeTierError 403` when called from an external script: *"OpenCode's free tier can
  only be used from within OpenCode."* Not bypassable from outside.
- **Paid models need a balance.** `402 Insufficient account funds` until Zen has one.
  Cheapest usable are `deepseek-v4-flash` and `glm-5.3-flash` — roughly **$0.30 for a
  1,700-call trial**, which is a small run, not the default one. See the call table above
  before sizing a real run.
- **Zen auto-reloads $20** when the balance drops below $5. Disable that if you want a
  hard ceiling.
- **A real `User-Agent` is mandatory.** Cloudflare fronts Zen and rejects Python's
  default `Python-urllib/x.y` with error 1010, *"access based on your browser's
  signature"*. The client identifies honestly rather than impersonating a browser:
  `USER_AGENT = "hive/0.1 (+https://github.com/Narla7/hive)"`.

**OpenCode Go's subscription is rate-capped per rolling window and is unsuitable for
this workload.** It is for dev and CI.

> **Note on the cost table:** `decisioners.PRICING` is explicitly labelled *"Rough
> per-1M-token USD, used only to put a number on inference spend so the fitness function
> can charge for it. Replace with your provider's real rates."* It is a placeholder, it
> does not match the rates recorded in `.env.example`, and fitness depends on it — so a
> model missing from the table falls to `_default = (1.0, 3.0)` and gets charged as if
> it were a large model.

### Environment variables

| Variable | Used for | Notes |
| --- | --- | --- |
| `HIVE_API_KEY` | LLM auth | `MA_API_KEY` still read as a fallback, so pre-rename `.env` files keep working |
| `HIVE_BASE_URL` | LLM endpoint | Defaults to `https://openrouter.ai/api/v1` |
| `OPENROUTER_API_KEY` | LLM auth | Checked after `HIVE_API_KEY` |
| `OPENCODE_API_KEY` | LLM auth | Checked last |

Resolution order in `LLMDecisioner.__init__`: `--api-key` → `HIVE_API_KEY` →
`OPENROUTER_API_KEY` → `OPENCODE_API_KEY`. No key at all raises `ValueError` and the
CLI exits `2`.

---

## CLI reference

`prog="hive"`. Three argument groups. Every default below is from `build_parser()`.

### Loop shape

| Flag | Default | Description |
| --- | --- | --- |
| `--population POPULATION` | `12` | Genomes per generation |
| `--generations GENERATIONS` | `8` | Epochs to run |
| `--episodes EPISODES` | `3` | Episodes per genome per generation |
| `--bars BARS` | `720` | Minutes of data per episode (one US session) |
| `--cash CASH` | `10000.0` | Starting cash, and the denominator for `ret` |
| `--seed SEED` | `7` | RNG seed for mutation, crossover and injection |
| `--market-seed MARKET_SEED` | `1234` | Synthetic market seed. The holdout uses `market_seed + 9999` |
| `--decide-every DECIDE_EVERY` | `3` | Bars between decisions → **239 decisions per 720-bar episode** |
| `--elite ELITE` | `3` | Survivors carried into the next generation |
| `--novelty NOVELTY` | `0.10` | Fresh random genomes per generation, as a fraction of population |
| `--exploration EXPLORATION` | `0.35` | UCB exploration constant `c` (see [Known limitations](#known-limitations)) |
| `--max-drawdown MAX_DRAWDOWN` | `0.25` | Drawdown gate. A breach quarantines the genome |

### Model

| Flag | Default | Description |
| --- | --- | --- |
| `--model MODEL` | `offline/rules` | `'offline/rules'` (no key) or any model id on an OpenAI-compatible endpoint. `rules` and `none` are also accepted |
| `--base-url BASE_URL` | `None` | e.g. `https://opencode.ai/zen/v1`. Falls back to `HIVE_BASE_URL`, then OpenRouter |
| `--api-key API_KEY` | `None` | Or set `HIVE_API_KEY` |
| `--provider {openai,anthropic}` | `openai` | Wire format |
| `--data DATA` | `None` | CSV of real bars instead of simulated |
| `--symbol SYMBOL` | `SIM` | **Currently a no-op** — see [Real bar data](#real-bar-data) |
| `--probe` | off | One health-check call, then exit. Exit `0` / `1` / `2` |
| `--max-calls-per-epoch N` | `None` | Hard cap on model calls per generation. No effect on `offline/rules` |

### Output

| Flag | Default | Description |
| --- | --- | --- |
| `--json` | off | Emit the machine-readable report on stdout |
| `--out OUT` | `None` | Write `best_genome.json` and `report.json` into `OUT` (created if absent) |
| `--quiet` | off | Suppress the human report. `--json` still prints |
| `--no-holdout` | off | Skip the out-of-sample check |
| `--tui` | off | Full-screen curses interface. Requires a real TTY; exits `2` without one |

### Exit codes

| Code | When |
| --- | --- |
| `0` | Run completed (including a `--quiet` run, and a TUI session that finished) |
| `1` | `--probe` failed (HTTP error, or unparseable JSON), or the TUI worker raised |
| `2` | No API key for a non-`offline/rules` model, or `--tui` without a real terminal |

Note that the loop's `IMPROVING` / `NOT IMPROVING` verdict and an `OVERFIT` holdout do
**not** change the exit code. This is an experiment harness, not a CI gate.

### `Config` fields with no CLI flag

`evolution.Config` carries three tuning knobs the CLI never exposes; set them from
Python:

| Field | Default | What it does |
| --- | --- | --- |
| `episode_spacing` | `900` | Bars between episode windows. Must exceed `bars` or episodes overlap |
| `cross_genome_rate` | `0.3` | Fraction of population created by crossover each generation |
| `max_calls_per_epoch` | `None` | Same as the CLI flag |

---

## Real bar data

```bash
PYTHONPATH=src python3 -m hive.cli --data data/spy.csv
```

CSV, one row per bar, ISO-8601 timestamps. Note the `symbol` column value and the
timestamp range — both are load-bearing, and both are easy to get wrong:

```csv
ts,open,high,low,close,volume,symbol
2026-01-05T14:30:00,100.0000,100.0800,99.9200,100.0500,1000000,SIM
2026-01-05T14:31:00,100.0500,100.1400,99.9800,100.1100,1000000,SIM
```

`FileMarket` sorts by timestamp and returns bars in `[start, end)`. Multiple symbols in
one file are fine — it groups by the `symbol` column.

### Two things that will bite you

**1. The `symbol` column must literally read `SIM`.** `evolution.run()` hardcodes
`symbol = "SIM"` and never reads `--symbol`, which is parsed and passed to
`run_tui()` and then dropped. This is a bug, not a design choice. If your file says
`SPY`, `FileMarket` returns zero bars, every genome sits out the episode, every genome
takes the inactivity penalty, and the run reports this:

```
$ PYTHONPATH=src python3 -m hive.cli --data spy.csv --symbol SPY --population 4 --generations 2 --episodes 2
gen  0  mean_fitness   -0.3750  best 35ab66ec8f9a raw  -1.5000 -> shrunk  -0.3750  ret +0.000%  dd 0.0%  cost/eval 0.0000  n_eff 2.0  new 2
gen  1  mean_fitness   -0.3750  best 35ab66ec8f9a raw  -1.5000 -> shrunk  -0.3750  ret +0.000%  dd 0.0%  cost/eval 0.0000  n_eff 2.0  new 2
------------------------------------------------------------------------------
mean fitness: -0.3750 -> -0.3750  (first/last third)  [NOT IMPROVING]
```

`ret +0.000%` with every genome pinned at `raw = -1.5000` is the fingerprint. Rename
the column to `SIM` and the same file produces real trades:

```
gen  0  mean_fitness   -0.6414  best 986e4d30f80d raw   0.7149 -> shrunk   0.1787  ret +4.827%  dd 2.6%  cost/eval 0.0000  n_eff 2.0  new 3
```

**2. Your data must cover the hardcoded session windows.** `run()` uses
`start = datetime(2026, 1, 5, 14, 30)`, and episode `i` covers
`start + i·900 min` for `bars` minutes. With the defaults that is
`2026-01-05T14:30` through roughly `2026-01-08`. A CSV of yesterday's session yields
zero bars for exactly the same reason.

### The holdout does not use your data

`holdout()` constructs `SimulatedMarket(seed=market_seed)` unconditionally — it takes
no market argument. Under `--data` the in-sample half runs on your CSV and the
out-of-sample half runs on a **synthetic** market:

```
holdout (unseen market seed, 5 windows):
  in-sample ret  +4.83%     ← from spy.csv
  holdout ret    +20.88%    ← from a synthetic market that has nothing to do with spy.csv
  >> edge survives out-of-sample
```

That `+20.88%` says nothing about SPY. It is the most important limitation on this
repository and it is listed again in [Known limitations](#known-limitations).

---

## Docker

```bash
docker compose build                     # tags hive:latest
docker compose run --rm test             # 129 tests
docker compose run --rm evolve           # the default 16/12/6 run
```

Override the command for a quick check:

```bash
docker compose run --rm evolve --generations 4 --quiet
```

| Service | Entry point | Default command | Image |
| --- | --- | --- | --- |
| `evolve` | `python -m hive.cli` | `--population 16 --generations 12 --episodes 6` | `hive:latest` |
| `test` | `python -m unittest discover -s tests -t .` | — | `hive:latest` |

Verified: `docker compose run --rm test` reports `Ran 129 tests in 5.342s` / `OK`.

The image is `python:3.14-slim` with `PYTHONPATH=/app/src`, `PYTHONUNBUFFERED=1`,
`PYTHONDONTWRITEBYTECODE=1`, and a non-root user `agent` — nothing here needs
privileges. There is nothing to audit in a supply chain because there are no
dependencies to install.

The TUI needs a real terminal, so run it without `-d`:

```bash
docker compose run --rm evolve --tui
```

`.env.example` documents every endpoint. The compose file passes `HIVE_API_KEY` and
`HIVE_BASE_URL` through, and mounts `./runs:/app/runs` — note that **nothing in the code
actually writes to `runs/`**; `--out DIR` is the only thing that writes to disk. Attach
a data file by uncommenting the `./data:/app/data:ro` mount.

---

## Tests

```bash
$ PYTHONPATH=src python3 -m unittest discover -s tests -t .
.................................................................................................................................
----------------------------------------------------------------------
Ran 129 tests in 5.351s

OK
```

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_harness.py` | 57 | Ledger, broker, fitness, genome, gates, market, decisioner, episode + loop |
| `tests/test_llm_path.py` | 34 | Probe, `CallBudget`, parse/network failure counters, truncation exclusion |
| `tests/test_tui.py` | 38 | Frame shape and terminal-size degradation, `Control` threading, `theme` helpers |

**No test can spend money.** Every network call in `test_llm_path.py` is
`mock.patch("urllib.request.urlopen", …)`, and the file's own docstring says so. The
TUI tests never touch a terminal: `render.py` imports no curses, so `build_frame()` is
a pure function and the layout is asserted headlessly.

The suite carries a regression test for every bug in
[Nine decisions that were bugs first](#nine-decisions-that-were-bugs-first), which is
what keeps them fixed. A representative run:

```
$ PYTHONPATH=src python3 -m unittest discover -s tests -t . -p test_harness.py
.........................................................
----------------------------------------------------------------------
Ran 57 tests in 5.230s

OK
```

---

## Project structure

```
hive/
├── pyproject.toml            name=hive, version 0.1.0, requires-python >=3.11, dependencies=[]
│                             [project.scripts] hive = "hive.cli:main"
├── Dockerfile                python:3.14-slim, non-root user `agent`, ENTRYPOINT python -m hive.cli
├── docker-compose.yml        services `evolve` and `test`, both on image hive:latest
├── .env.example              every endpoint, and the four Zen/Go/Codex/Claude gotchas
├── src/hive/
│   ├── __init__.py           __version__ = "0.1.0"; re-exports the public API
│   ├── market.py             Bar, Market, SimulatedMarket, FileMarket
│   ├── broker.py             BrokerConfig, Fill, PaperBroker
│   ├── ledger.py             Posting, Entry, Ledger — rejects unbalanced entries
│   ├── genome.py             Policy, Genome, mutate, crossover, random_genome, hand_seeded
│   ├── fitness.py            Episode, Fitness, evaluate, max_drawdown, effective_n, downside_dev
│   ├── gates.py              GateConfig, GateVerdict, Gates
│   ├── decisioners.py        Decision, AgentState, RulesDecisioner, LLMDecisioner, CallBudget, probe
│   ├── evolution.py          Config, Record, Holdout, GenerationReport, run_episode, allocate, holdout, run
│   ├── render.py             TuiState, build_frame  (no curses)
│   ├── theme.py              palette, sigil, sparkline, bar, corrupt, Flicker, ritual lines
│   ├── tui.py                Control, run_tui  (the only curses import)
│   └── cli.py                build_parser, main
└── tests/
    ├── test_harness.py       57 tests
    ├── test_llm_path.py      34 tests, every urlopen mocked
    └── test_tui.py           38 tests, no terminal
```

There is no `LICENSE` file and no CI configuration in the repository. The repository is
private.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'hive'` | Ran from the repo root without `PYTHONPATH` | `PYTHONPATH=src python3 -m hive.cli …`, or `pip install -e .` |
| Every genome pinned at `raw = -1.5000`, `ret +0.000%` | CSV `symbol` column is not `SIM`, or the file does not cover the hardcoded 2026-01-05T14:30 window | See [Real bar data](#real-bar-data). `raw = -1.5` is the inactivity penalty with zero bars |
| `error: --tui needs a real terminal` (exit 2) | `--tui` under a pipe, a CI job, or a detached shell | Run it directly, or drop `--tui` for the text report |
| `terminal too small` in the TUI | Fewer than 40 columns or 8 rows | Enlarge, or use the text report |
| `error: no API key: set HIVE_API_KEY, OPENROUTER_API_KEY or OPENCODE_API_KEY…` (exit 2) | `--model <id>` with no key in the environment or on the command line | Export the key, or use `--model offline/rules` |
| `HTTP 403: … FreeTierError` from the probe | OpenCode Zen free model called from outside OpenCode | Use a paid Zen model, or another provider |
| `HTTP 402: Insufficient account funds` | Zen account balance is zero | Top up. `deepseek-v4-flash` / `glm-5.3-flash` are the cheapest |
| `HTTP 403 … error 1010` / "browser's signature" | Cloudflare rejecting the default Python User-Agent | Not a config problem — the client already sends `hive/0.1 (+https://github.com/Narla7/hive)`. If you forked, keep a real `User-Agent` |
| `parse_failures=N` in the report | The model is not returning the requested JSON | The prompt asks for a single JSON object. Check `raw` from `--probe`; a chatty model still parses if it contains braces |
| `>> 4 episode(s) hit --max-calls-per-epoch` | The per-generation call cap was reached | Raise `--max-calls-per-epoch`, shrink `--population`, or accept that the verdict is degraded. Affected genomes show `raw 0.0000` and `n_eff 0.0` |
| `--max-calls-per-epoch` appears to do nothing | Running `offline/rules`, which has no call budget | It only applies to `LLMDecisioner` |
| `[NOT IMPROVING]` with a positive best fitness | A short run, or a hard market. Verdict compares the first third against the last third | More `--generations`, or `--data` |
| `holdout ret` looks great but your data is flat | Under `--data`, the holdout runs on a *synthetic* market | Known bug. See [Known limitations](#known-limitations) |
| Full default run is slow, or the LLM bill is | 239 decisions/episode × pop × episodes × generations = 276,523 calls for `--population 16 --generations 12 --episodes 6` | `--population 8 --generations 4 --episodes 3` is 22,944 calls and finishes in ~7s offline. Set `--max-calls-per-epoch` |
| `run_tui` returns exit 1 and the log says `the rite broke` | The worker thread raised | The exception and traceback are pushed into the WHISPERS panel |

---

## Known limitations

Listed most-consequential first. All of these are in the code, not in a plan.

### The default market is synthetic and deliberately exploitable

Trend regimes, mean reversion and jumps, as described in
[Honest limits on the numbers](#honest-limits-on-the-numbers). Reported returns are much
higher than real day trading produces. The default run demonstrates that **the harness
is correct**, not that a strategy is profitable. Point it at real data before
concluding anything about markets.

### The holdout ignores `--data`

`holdout()` always builds a `SimulatedMarket`. Under `--data`, in-sample is your CSV and
out-of-sample is a synthetic market with no relationship to it. The
`>> edge survives out-of-sample` line under `--data` is not evidence about your
instrument. This is the single most important limitation here.

### `--symbol` is a no-op

Parsed, passed into `run_tui()`, dropped. `evolution.run()` hardcodes `symbol = "SIM"`.
The CSV `symbol` column must read `SIM` or the run silently evaluates nothing. See
[Real bar data](#real-bar-data).

### `>> edge survives out-of-sample` can print on a zero in-sample result

The verdict is `if ins > 0 and hold.ret <= 0: OVERFIT elif hold.ret > 0: survives`. A
run where the in-sample half never traded (`ins == 0.0`, the `--data` failure mode
above) and the synthetic holdout happened to make money prints "edge survives". Check
`in-sample ret` yourself; do not trust the line.

### UCB budget allocation is defined but never called

`evolution.allocate()` implements UCB over shrunken fitness with an optimism term
`c·√(log(total+1)/n)` and a 5% floor. **No code path calls it.** Every genome receives
exactly `cfg.episodes` episodes every generation. `--exploration` is therefore a dead
flag, and the exploration/exploitation trade-off the module docstring describes is not
in effect. The comments around the breeding step are accurate about what happens;
`allocate()` is a prepared tool that has not been connected.

### The population kill switch and the daily-loss gate have no trigger

`Gates.kill_switch()` sets `_tripped`; `run()` reads `gates.tripped` and breaks. Nothing
in the loop ever calls `kill_switch()`. `check_daily_loss()` is never called either, so
`GateConfig.max_daily_loss` never fires. `max_gross_exposure` is declared and never read
by any method. None of the three is settable from the CLI. The gates that are actually
live are `admit()` (which does consult `cfg.adapters`, the `{"sim"}` allowlist),
`check_order()`'s `max_position_frac` cap, and `check_drawdown()`.

### Two genome fields are evolved but never read

`Policy.cooldown_bars` is mutated, diffed, printed, and validated — and neither
decisioner consults it. `Policy.allow_short` is a documented v0.2 field that nothing
reads. They inflate the appearance of the search space and consume mutation budget
without changing behaviour. (`GateConfig.cooldown_bars` is a separate mechanism and
also defaults to inert.)

### Long-only

The broker is long-or-flat. A negative `target_weight` from an LLM is clamped to
`[-1, 1]` and then the sell branch treats any `target <= 0` as an exit, so a model that
asks to go short simply gets flat. `allow_short` is the v0.2 hook.

### Paper trading only

No real money moves anywhere in this repository. `PaperBroker` is the only broker, and
the `DEFAULT_ADAPTERS = {"sim"}` allowlist in `gates.py` means a genome cannot even name
a live adapter — `Gates().admit(genome, adapter="live-money")` is rejected, and there is
a test for exactly that.

### The TUI is the one place curses is allowed

It is the only module that imports `curses`, it requires a real TTY, and it is not
usable over `docker compose run -d` or in CI. Everything else in the interface stack is
a pure function specifically so this stays true.

### The rename, and one compatibility shim

The rename from `money-agent` to `hive` is complete: package `hive`, module `hive.cli`,
console script `hive`, Docker image `hive:latest`, environment variables `HIVE_API_KEY` and
`HIVE_BASE_URL`, and a `User-Agent` of `hive/0.1 (+https://github.com/Narla7/hive)`.

The single exception is deliberate: `MA_API_KEY` and `MA_BASE_URL` are still read as
fallbacks, because a pre-rename `.env` that silently stopped working would be a nasty
surprise for no benefit. Drop them in `LLMDecisioner.__init__` once you have migrated.

### The parse-failure warning miscounts its own denominator

`cli.py` prints `f"{pf} of {calls + pf} decisions were unparseable"`, but
`LLMDecisioner.decide` increments `self.calls` before parsing, so unparseable replies are
already counted inside it. A real run printed `424 of 4248` when 424 of 3824 replies
were unparseable. The warning still fires, which is the important part, but the fraction
it quotes is wrong. `parse_failures` and `calls` are individually correct.

### No `LICENSE`, no CI

The repository has no license file and no `.github/` directory. Every figure in this
README was produced by running the code on this machine; there is no published
benchmark, because there is nothing published to benchmark against.

---

## Not financial advice

Paper trading against a synthetic market. No real money moves anywhere in this
repository, and the reported returns are not a prediction of anything. The default run
demonstrates that the harness works, not that a strategy makes money. If you point it
at real data and get a result you like, that is a much smaller claim than it feels like,
and the honest next step is more out-of-sample windows, not more capital.
