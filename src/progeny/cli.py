"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .decisioners import LLMDecisioner, RulesDecisioner
from .evolution import Config, buy_and_hold, holdout, run
from .sessions import plan_from_config
from .gates import GateConfig
from .genome import Genome
from .market import FileMarket, SimulatedMarket


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="progeny",
        description="Evolutionary harness for day-trading agents.",
    )
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--bars", type=int, default=390,
                   help="bars per window; one regular US session is 390")
    p.add_argument("--episode-spacing", type=int, default=2,
                   help="sessions between episode windows. 1 is cheaper but "
                        "adjacent sessions are correlated, so 2 is the honest floor")
    p.add_argument("--holdout-windows", type=int, default=30,
                   help="windows in the reported holdout, from a disjoint range")
    p.add_argument("--reselect-windows", type=int, default=24,
                   help="wide cheap pass that narrows the shortlist")
    p.add_argument("--reselect-final-windows", type=int, default=40,
                   help="narrow pass that picks the winner among the finalists")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--market-seed", type=int, default=1234)
    p.add_argument("--decide-every", type=int, default=3, help="bars between decisions")
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--novelty", type=float, default=0.10)
    p.add_argument("--exploration", type=float, default=0.35)
    p.add_argument("--max-drawdown", type=float, default=0.25)

    g = p.add_argument_group("model")
    g.add_argument(
        "--model", default="offline/rules",
        help="'offline/rules' (no key needed) or any model id on an "
             "OpenAI-compatible endpoint",
    )
    g.add_argument("--base-url", default=None, help="e.g. https://opencode.ai/zen/v1")
    g.add_argument("--api-key", default=None, help="or set PROGENY_API_KEY")
    g.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic"],
    )
    g.add_argument("--data", default=None, help="CSV of real bars instead of simulated")
    g.add_argument("--symbol", default="SIM")
    g.add_argument(
        "--probe", action="store_true",
        help="make one health-check call and exit (no key cost, ~1s)",
    )
    g.add_argument(
        "--max-calls-per-epoch", type=int, default=None,
        help="hard cap on model calls per generation; excess episodes are "
             "marked truncated and excluded from fitness",
    )

    o = p.add_argument_group("output")
    o.add_argument("--json", action="store_true", help="emit machine-readable report")
    o.add_argument("--out", default=None, help="write ledger(s) and best genome here")
    o.add_argument("--quiet", action="store_true")
    o.add_argument("--no-holdout", action="store_true", help="skip out-of-sample check")
    o.add_argument(
        "--tui", action="store_true",
        help="full-screen curses interface (needs a real terminal)",
    )
    return p


def fmt_money(v: float) -> str:
    return f"{v:+,.2f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config(
        population=args.population,
        generations=args.generations,
        episodes=args.episodes,
        bars=args.bars,
        elite=args.elite,
        novelty_frac=args.novelty,
        exploration_c=args.exploration,
        seed=args.seed,
        market_seed=args.market_seed,
        initial_cash=args.cash,
        decide_every=args.decide_every,
        model=args.model,
        episode_spacing=args.episode_spacing,
        holdout_windows=args.holdout_windows,
        reselect_windows=args.reselect_windows,
        reselect_final_windows=args.reselect_final_windows,
        symbol=args.symbol,
    )
    gates_cfg = GateConfig(max_drawdown=args.max_drawdown)

    market = FileMarket(args.data) if args.data else SimulatedMarket(seed=args.market_seed)

    if args.tui:
        if not sys.stdout.isatty():
            print("error: --tui needs a real terminal. "
                  "Run it directly, or drop --tui for the text report.", file=sys.stderr)
            return 2
        from .tui import run_tui
        return run_tui(
            cfg, args.model, args.api_key, args.base_url, args.provider,
            args.data, args.symbol, gates_cfg,
            do_holdout=not args.no_holdout,
        )

    if args.model in ("offline/rules", "rules", "none"):
        decisioner = RulesDecisioner()
    else:
        try:
            decisioner = LLMDecisioner(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                provider=args.provider,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # ---- probe: one call, then out ----
    if args.probe:
        r = decisioner.probe()
        print(f"endpoint   {r.endpoint}")
        print(f"model      {r.model}")
        if r.error and r.status != 200:
            print(f"status     FAILED")
            print(f"error      {r.error}")
            print("\nprobe FAILED — fix this before a real run, which fires thousands of calls.")
            return 1
        print(f"status     {r.status}  ({r.elapsed_ms}ms)")
        print(f"tokens     in={r.in_tokens}  out={r.out_tokens}")
        print(f"cost       ${r.cost:.6f}")
        print(f"raw        {r.raw[:300]!r}")
        if r.parsed is None:
            print(f"parsed     FAIL ({r.error})")
            print("\nprobe FAILED — the model answered, but not with parseable JSON.")
            print("Fitness would silently measure nothing, so this has to be fixed first.")
            return 1
        print(f"parsed     {r.parsed}")
        print(f"\nprobe OK — {r.in_tokens} in / {r.out_tokens} out tokens, ${r.cost:.6f}")
        return 0

    cfg.max_calls_per_epoch = args.max_calls_per_epoch

    if not args.quiet:
        source = "REAL (--data)" if args.data else "SYNTHETIC (--market-seed)"
        print(f"model={args.model} pop={cfg.population} gens={cfg.generations} "
              f"episodes={cfg.episodes} bars={cfg.bars} cash={cfg.initial_cash:,.0f}")
        print(f"data={source}  sessions={len(market.sessions(args.symbol))}")
        if not args.data:
            print("  NOTE: synthetic prices. Returns say nothing about real markets.")
        print("-" * 78)

    reports, best = run(
        cfg, market=market, decisioner=decisioner, gates_cfg=gates_cfg,
        symbol=args.symbol,
    )

    out = []
    for rep in reports:
        line = (
            f"gen {rep.generation:>2}  mean_fitness {rep.mean_fitness:>9.4f}  "
            f"best {rep.best.genome_id} raw {rep.best.fitness.raw:>8.4f} "
            f"-> shrunk {rep.best.fitness.fitness:>8.4f}  "
            f"ret {rep.best.fitness.ret:+.3%}  dd {rep.best.fitness.drawdown:.1%}  "
            f"cost/eval {rep.best.fitness.cost_per_eval:.4f}  "
            f"n_eff {rep.best.fitness.n_eff:.1f}  new {rep.new_genomes}"
        )
        out.append(line)
        if not args.quiet:
            print(line)
            if rep.best_diff:
                for d in rep.best_diff.splitlines():
                    print(f"        winner delta: {d}")
            if rep.quarantined:
                print(f"        quarantined: {', '.join(rep.quarantined)}")

    # Compare thirds, not endpoints. A single noisy generation at the end can
    # flatter or wreck the verdict; averaging a third of the run each side is
    # far more stable on a stochastic objective.
    means = [r.mean_fitness for r in reports]
    if len(means) >= 4:
        k = max(1, len(means) // 3)
        early = sum(means[:k]) / k
        late = sum(means[-k:]) / k
    elif len(means) >= 2:
        early, late = means[0], means[-1]
    else:
        early = late = means[0] if means else 0.0
    improving = late > early
    verdict = "IMPROVING" if improving else "NOT IMPROVING"

    hold = None
    if not args.no_holdout:
        # Same file, same sessions, disjoint range. The holdout used to build a
        # fresh SimulatedMarket, so under --data it was testing a synthetic
        # series unrelated to the CSV.
        sessions = market.sessions(args.symbol)
        plan = plan_from_config(
            cfg.generations, cfg.episodes, cfg.holdout_windows,
            cfg.reselect_windows, cfg.reselect_final_windows, cfg.episode_spacing,
        )
        hold = holdout(best, cfg, decisioner, market, args.symbol, sessions,
                       plan.holdout, plan.spacing)
        bh = buy_and_hold(market, args.symbol, sessions, plan.holdout, plan.spacing)

    if not args.quiet:
        print("-" * 78)
        print(f"mean fitness: {early:.4f} -> {late:.4f}  (first/last third)  [{verdict}]")
        if hold:
            ins = reports[-1].best.fitness.ret if reports else 0.0
            src = "committed bars" if args.data else "synthetic, disjoint sessions"
            print(f"holdout ({src}, {hold.n_windows} windows, disjoint in time):")
            print(f"  in-sample ret      {ins:+.2%}")
            print(f"  holdout ret        {hold.ret:+.2%}   pnl {hold.pnl:+,.2f}  "
                  f"trades {hold.trades}  shrunk {hold.fitness.fitness:.4f}")
            print(f"  buy-and-hold       {bh.ret:+.2%}   "
                  f"({bh.up_windows}/{bh.windows} windows up)")
            if hold.ret > 0 and hold.ret <= bh.ret:
                print("  >> NO EDGE: the strategy did not beat simply owning it.")
            elif ins > 0 and hold.ret <= 0:
                print("  >> OVERFIT: profitable in-sample, not out-of-sample")
            elif hold.ret > bh.ret:
                print(f"  >> beats buy-and-hold by {hold.ret - bh.ret:+.2%} "
                      f"out-of-sample")

        # ---- model health, loud not silent ----
        if hasattr(decisioner, "calls"):
            calls, cost = decisioner.calls, decisioner.total_cost
            pf, ne = decisioner.parse_failures, decisioner.network_errors
            print(f"\ninference: {calls} calls  ${cost:.4f}  "
                  f"parse_failures={pf}  network_errors={ne}")
            if pf:
                print(f"  >> WARNING: {pf} of {calls + pf} decisions were unparseable.")
                print("  >> Fitness is NOT measuring strategy quality. Fix the prompt")
                print("  >> or the parser before trusting any number above.")
            if ne:
                print(f"  >> WARNING: {ne} network errors. Results are partial.")
            if calls == 0 and not (args.no_holdout and not calls):
                print("  >> WARNING: zero model calls were made. Check --model.")

        # ---- truncation, which biases fitness if ignored ----
        dropped = sum(r.best.fitness.n_truncated for r in reports)
        if dropped:
            print(f"\n>> {dropped} episode(s) hit --max-calls-per-epoch and were")
            print(">> excluded from fitness. The verdict above is degraded.")
        print(f"best genome: {best.id}")
        print("  policy:", json.dumps(best.to_dict()["policy"], indent=2).replace("\n", "\n  "))
        if hasattr(decisioner, "total_cost"):
            print(f"  inference spend: ${decisioner.total_cost:.4f} over {decisioner.calls} calls")

    if args.json:
        print(json.dumps({
            "verdict": verdict,
            "generations": [
                {
                    "generation": r.generation,
                    "mean_fitness": r.mean_fitness,
                    "best_id": r.best.genome_id,
                    "best_raw": r.best.fitness.raw,
                    "best_fitness": r.best.fitness.fitness,
                    "best_ret": r.best.fitness.ret,
                    "best_drawdown": r.best.fitness.drawdown,
                    "n_eff": r.best.fitness.n_eff,
                    "quarantined": r.quarantined,
                }
                for r in reports
            ],
            "best_genome": best.to_dict(),
        }, indent=2))

    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "best_genome.json").write_text(json.dumps(best.to_dict(), indent=2))
        (d / "report.json").write_text(json.dumps(
            [r.best.fitness.__dict__ if hasattr(r.best.fitness, "__dict__") else {
                "raw": r.best.fitness.raw, "fitness": r.best.fitness.fitness,
                "ret": r.best.fitness.ret, "drawdown": r.best.fitness.drawdown,
            } for r in reports], indent=2))
        if not args.quiet:
            print(f"wrote {d}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
