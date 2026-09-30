# money-agent

An evolutionary harness for day-trading agents. **Fitness is realized net P&L from a
double-entry ledger.** Agents trade, the ledger records every fill, the profitable ones
breed, the rest die, and the population is scored again.

```bash
python3 -m money_agent.cli --population 16 --generations 12 --episodes 6
```

No API key, no network, no dependencies. Python 3.11+ and the standard library.

---

## The nutshell

A genome is a trading strategy as data — lookback, thresholds, stop, target, position
size. A population of them trades the same price series. Each one is scored by what it
actually made, after fees, after slippage, after model spend. Top genomes breed, losers
are culled, fresh random genomes are injected, repeat.

The model is *not* the source of the money. It proposes actions; the broker executes them
and the ledger decides what happened. That split is why the whole thing runs and produces
a real experiment with zero API keys, and why adding an LLM later changes nothing about
how results are scored.

## Try it

```bash
# offline, no key
PYTHONPATH=src python3 -m money_agent.cli --population 16 --generations 12 --episodes 6

# with an LLM making the decisions
export MA_API_KEY=sk-...
PYTHONPATH=src python3 -m money_agent.cli --model anthropic/claude-sonnet \
    --base-url https://openrouter.ai/api/v1 --population 8 --generations 5

# real bars instead of simulated
PYTHONPATH=src python3 -m money_agent.cli --data data/spy.csv --symbol SPY

# docker
docker compose run --rm evolve
docker compose run --rm test
```

## Read the output

```
gen  5  mean_fitness 12.5076  best b7a7fa1e6dba raw 88.6673 -> shrunk 50.6670  ret +52.704%  dd 4.4%  n_eff 8.0  new 13
```

- **`raw` vs `shrunk`** — raw is the risk-adjusted score; shrunk is what selection
  actually uses. The gap is the shrinkage term pulling toward zero in proportion to how
  little evidence the genome has. A spectacular raw score off three episodes gets
  *shrunk*, not rewarded. Without it the loop latches onto early noise and mistakes the
  resulting sample growth for confirmation.
- **`n_eff`** — effective sample size after correcting for autocorrelation. Agent P&L is
  serially correlated, so the raw episode count overstates the evidence.
- **`dd`** — max drawdown, from the realized per-bar equity curve, not summary stats.

Then the part that matters most:

```
holdout (unseen market seed, 5 windows):
  in-sample ret  +52.70%
  holdout ret    +31.88%
  >> edge survives out-of-sample
```

Selection over a handful of fixed price windows always produces a winner. The holdout
runs the chosen genome against a market seed the search never saw. **If in-sample is
profitable and holdout is not, the loop has fitted noise** and the CLI says so in those
words.

## Honest caveat about the numbers

The default market is synthetic and **deliberately has exploitable structure** —
persistent trend regimes, mean reversion and occasional jumps. A pure random walk has no
edge to find, so any fitness-improving experiment against one would be measuring
nothing. The consequence is that the reported returns are *much* higher than real day
trading produces. The +30–50% figures are an artifact of a friendly simulator.

What the default run demonstrates is that **the harness is correct**: the loop improves,
the ledger balances, the shrinkage term bites, the gates fire, and the edge survives
out-of-sample. It does not demonstrate a profitable strategy. Point it at real data
(`--data`) before drawing any conclusion about markets.

## Design

| Module | Role |
| --- | --- |
| `market.py` | Price series. Simulated (deterministic, seeded) or CSV. |
| `broker.py` | Executes orders with slippage, spread, fees. Only place orders happen. |
| `ledger.py` | Append-only double-entry. Rejects unbalanced entries. |
| `genome.py` | The evolvable strategy spec, plus mutate/crossover. |
| `fitness.py` | Scoring. Pure function of the ledger. |
| `gates.py` | Hard limits. Enforced by the harness, not requested of the model. |
| `decisioners.py` | `RulesDecisioner` (offline) and `LLMDecisioner` (any endpoint). |
| `evolution.py` | The epoch loop, budget allocation, holdout validation. |

Five decisions worth knowing about, each of which was a bug before it was a decision:

**1. Fitness cannot reward inaction.** A genome that never trades has zero return, zero
drawdown and zero downside deviation, so it scores exactly `0` — which beats every
strategy that actually loses money. The first working version converged on doing nothing
and reported it as progress. There is now an additive inactivity penalty. Not trading is
unproven, not optimal, and has to score below it.

**2. Trading frictions are charged once.** The turnover penalty is deliberately tiny
(`2e-4`) because fees and slippage are already in the ledger. An earlier version charged
them twice and crushed exactly the low-churn strategies that survive costs.

**3. Downside deviation is floored.** Dividing by a near-zero denominator amplifies
sampling noise into enormous raw scores. A genome making −2.7% with almost no variance
was scoring `raw = −122`, which is a meaningless number.

**4. Episodes must be different windows.** Running three episodes over one fixed price
path with a deterministic decisioner returns three identical results, so `n_eff` is
meaningless and the search can quietly overfit a single price series. Every episode now
gets its own window; every genome sees the same set, so fitness stays comparable.

**5. Mutation self-repairs.** `stop_loss` and `take_profit` are jittered independently, so
mutation could invert reward/risk. A variation operator emitting invalid offspring half
the time burns search budget for nothing.

### Gates are structural

Gates are checked by the harness, between the decision and the order. A genome that
breaches a drawdown cap is quarantined; it does not get to argue its way past by
out-earning the limit. A gate enforced by a model is a suggestion.

```bash
--max-drawdown 0.25     # kill-switch the genome
```

There is a separate population-wide kill switch for systemic failure — a regime change
makes every genome breach at once, so per-genome gates protect nothing in that case.

## Probing the model path

```bash
python3 -m money_agent.cli --probe --model <id>
```

One call, then out. A full run fires thousands of calls, so without this a bad
key or a model that returns prose instead of JSON is discovered minutes and real
money later — or not at all, because an unparseable reply degrades to "hold
position" and every genome quietly behaves the same. The probe makes that
failure loud and costs nothing.

Every run also reports, at the end:

```
inference: 1,728 calls  $0.31  parse_failures=0  network_errors=0
```

`parse_failures > 0` means fitness is not measuring strategy quality, and the
CLI says so in those words. `--max-calls-per-epoch` caps model calls per
generation; episodes that get cut off are marked truncated and **excluded** from
fitness, because a half-length episode understates a genome that had less time
to trade, which would bias selection toward strategies that churn fast.

### Two things the probe found about OpenCode Zen

- **Free models are in-app only.** `big-pickle` and every `*-free` model return
  `FreeTierError 403` when called from an external script: *"OpenCode's free
  tier can only be used from within OpenCode."* Not bypassable from outside.
- **Paid models need a balance.** `402 Insufficient account funds` until Zen has
  one. Cheapest usable are `deepseek-v4-flash` ($0.14/$0.28 per 1M) and
  `glm-5.3-flash` ($0.15/$0.50) — roughly $0.30 for a 1,700-call trial.

Also: requests must send a real `User-Agent`. Cloudflare fronts Zen and rejects
Python's default with error 1010, "access based on your browser's signature".

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .   # 57 tests
```

The suite includes regression tests for each bug above, so they stay fixed.

## Extending

- **Real market data** — CSV with `ts,open,high,low,close,volume,symbol`. An Alpaca or
  Polygon adapter drops in behind `Market`; only the `bars()` method matters.
- **Live rails** — implement `PaperBroker`'s interface. The paper broker and a real
  venue are the same interface with different implementations. Nothing about the
  algorithm changes between them, only what fills the ledger. If that ever stops being
  true, the abstraction is wrong — fix it before exposing capital.
- **Shorts** — `Policy.allow_short` is already a field; the broker is long-only.
- **Multi-symbol** — the genome names one adapter; portfolios need cross-genome credit
  assignment (Shapley over the pipeline steps).

## Not financial advice

Paper trading against a synthetic market. No real money moves anywhere in this
repository, and the reported returns are not a prediction of anything.
