"""Curses driver.

Deliberately thin. All layout lives in `render.py`, all styling in `theme.py`,
and the evolution loop knows nothing about any of it — it just calls
`on_generation` and checks whether the callback wants it to stop.

The loop runs on a worker thread so key handling stays responsive during slow
LLM generations. Curses is only ever touched from the main thread.
"""

from __future__ import annotations

import curses
import random
import threading
import time
from datetime import datetime

from . import render, theme
from .broker import BrokerConfig
from .decisioners import LLMDecisioner, RulesDecisioner
from .evolution import Config, holdout as run_holdout, run as run_evolution
from .gates import GateConfig
from .market import FileMarket, SimulatedMarket


class Control:
    """Shared flags between the curses thread and the evolution worker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paused = False
        self.quit = False
        self.skip_to_end = False
        self.speed = 1.0
        self.finished = False

    def wait_if_paused(self) -> bool:
        """Block while paused. Returns False if the user asked to quit."""
        while True:
            with self.lock:
                if self.quit:
                    return False
                if self.skip_to_end or not self.paused:
                    return True
            time.sleep(0.1)


def _roster_from(report) -> list[dict]:
    out = []
    for rec in sorted(report.records, key=lambda r: r.fitness.fitness, reverse=True):
        out.append({
            "name": rec.genome_id,
            "fitness": rec.fitness.fitness,
            "raw": rec.fitness.raw,
            "ret": rec.fitness.ret,
            "trades": rec.fitness.trades,
        })
    return out


def _intro(stdscr, rng: random.Random) -> None:
    """Type the banner in, one line at a time. Skippable with any key."""
    stdscr.nodelay(False)
    stdscr.curs_set(0)
    height, width = stdscr.getmaxyx()
    lines = theme.art_lines(height)

    for i, line in enumerate(lines):
        for y in range(height):
            stdscr.addnstr(y, 0, " " * width, width)
        stdscr.addnstr(
            max(0, height // 2 - len(lines) // 2 + i), 0,
            line, width,
        )
        stdscr.addnstr(
            height - 3, 0,
            theme.center("the rite begins", width), width,
            curses.A_BOLD,
        )
        stdscr.refresh()
        time.sleep(0.045)
        if stdscr.getch() != -1:
            break
    stdscr.nodelay(True)


def _draw(stdscr, state: render.TuiState, flicker: theme.Flicker) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 8 or width < 40:
        stdscr.addnstr(0, 0, "terminal too small", max(1, width - 1))
        stdscr.refresh()
        return

    for y, row in enumerate(render.build_frame(state, width, height)):
        if y >= height:
            break
        x = 0
        for text, colour in row:
            if x >= width:
                break
            attr = curses.color_pair(colour)
            if state.dark:
                attr |= curses.A_DIM
            if colour == theme.C_BLOOD:
                attr |= curses.A_BOLD
            try:
                stdscr.addnstr(y, x, text, max(0, width - x), attr)
            except curses.error:
                pass   # writing to the last cell raises; harmless
            x += len(text)
    stdscr.refresh()


def _handle_keys(stdscr, ctl: Control, state: render.TuiState) -> None:
    ch = stdscr.getch()
    if ch == -1:
        return
    with ctl.lock:
        if ch in (ord("q"), ord("Q")):
            ctl.quit = True
            state.push("you tried to leave", theme.C_WARN)
        elif ch == ord(" "):
            ctl.paused = not ctl.paused
            state.paused = ctl.paused
            state.push(
                "the rite is held" if ctl.paused else "it resumes",
                theme.C_WARN if ctl.paused else theme.C_BLOOD,
            )
        elif ch in (ord("+"), ord("=")):
            ctl.speed = min(8.0, ctl.speed * 2)
            state.speed = ctl.speed
        elif ch in (ord("-"), ord("_")):
            ctl.speed = max(0.25, ctl.speed / 2)
            state.speed = ctl.speed
        elif ch in (ord("n"), ord("N")):
            ctl.skip_to_end = True
            ctl.paused = False
            state.paused = False
            state.push("no more waiting", theme.C_WARN)
        elif ch in (ord("s"), ord("S")):
            # Re-run the out-of-sample check on demand; it is the only number
            # that says whether the winner is real.
            state.holdout_done = False
            state.holdout = ""

    if ch == -1:
        state.dark = flicker.step()


def run_tui(cfg: Config, model: str, api_key: str | None, base_url: str | None,
            provider: str, data: str | None, symbol: str,
            gates_cfg: GateConfig | None = None, do_holdout: bool = True) -> int:
    """Run the harness with the interface up. Blocking until the rite ends."""

    def main(stdscr) -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        theme.init_pairs()
        stdscr.bkgd(" ")

        rng = random.Random(7)
        flicker = theme.Flicker()
        ctl = Control()
        state = render.TuiState(
            total_generations=cfg.generations,
            population=cfg.population,
            model=model,
        )
        state.push("the chamber seals", theme.C_BLOOD)

        decisioner: object
        try:
            decisioner = (
                RulesDecisioner() if model in ("offline/rules", "rules", "none")
                else LLMDecisioner(model, base_url=base_url, api_key=api_key, provider=provider)
            )
        except ValueError as exc:
            stdscr.addnstr(1, 2, str(exc)[:100], 90)
            stdscr.refresh()
            time.sleep(3.0)
            return 2

        market = FileMarket(data) if data else SimulatedMarket(seed=cfg.market_seed)
        gate_cfg = gates_cfg or GateConfig()

        def on_gen(report) -> bool:
            """Called on the worker thread once per generation."""
            if not ctl.wait_if_paused():
                return False
            with ctl.lock:
                state.generation = report.generation
                state.mean_history.append(report.mean_fitness)
                state.best_history.append(report.best.fitness.fitness)
                state.roster = _roster_from(report)
                state.quarantined = len(report.quarantined)
                state.truncated = sum(r.fitness.n_truncated for r in report.records)
                if hasattr(decisioner, "calls"):
                    state.calls = decisioner.calls
                    state.cost = decisioner.total_cost
                    state.parse_failures = decisioner.parse_failures
                prev = report.best.fitness.fitness
                state.push(theme.ritual_line(rng, report.generation),
                           theme.C_BLOOD if report.mean_fitness > 0 else theme.C_DIM)
                if report.quarantined:
                    state.push(theme.fail_line(rng), theme.C_WARN)
                if report.new_genomes:
                    state.push(f"{report.new_genomes} things were born", theme.C_RITUAL)
                del prev
            time.sleep(max(0.0, 0.85 / ctl.speed))
            return not ctl.quit

        result: dict = {}

        def worker() -> None:
            try:
                result["out"] = run_evolution(
                    cfg, market=market, decisioner=decisioner,
                    gates_cfg=gate_cfg, on_generation=on_gen,
                )
            except BaseException as exc:
                # BaseException, not Exception: a KeyboardInterrupt or other
                # non-Exception would otherwise skip this handler, run the
                # finally, and leave result empty while claiming it finished.
                import traceback
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["trace"] = traceback.format_exc()
            finally:
                ctl.finished = True

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Render loop.
        while not ctl.finished:
            _handle_keys(stdscr, ctl, state)
            if ctl.quit:
                break
            _draw(stdscr, state, flicker)
            time.sleep(0.08)
        thread.join(timeout=5.0)

        if ctl.quit:
            state.push("the rite was abandoned", theme.C_WARN)

        # Holdout, then leave the final frame up long enough to read.
        if "out" in result and do_holdout and not state.holdout_done:
            _reports, best = result["out"]
            state.holdout = "consulting the other market..."
            _draw(stdscr, state, flicker)
            h = run_holdout(best, cfg, decisioner,
                            market_seed=cfg.market_seed + 9999, n_windows=5)
            verdict = "it survives" if h.ret > 0 else "it does not survive"
            state.holdout = f"holdout {h.ret:+.2%} -- {verdict}"
            state.holdout_done = True
            state.push(f"the other market says {h.ret:+.2%}",
                       theme.C_GOOD if h.ret > 0 else theme.C_BLOOD)

        # Do not claim a completed rite when the user cut it short, and do not
        # skip the holdout silently. A run that stopped early has no winner, and
        # "the rite is complete" over a partial population is a lie.
        completed = False
        if "error" in result:
            state.push("the rite broke", theme.C_WARN)
            state.push(result["error"][:60], theme.C_WARN)
        elif "out" in result:
            _reports, _best = result["out"]
            done = len(_reports)
            if done >= cfg.generations:
                completed = True
            else:
                state.push(f"stopped at {done} of {cfg.generations} rites",
                           theme.C_WARN)
        elif ctl.quit:
            state.push("the rite was abandoned", theme.C_WARN)
        else:
            state.push("the rite did not finish", theme.C_WARN)

        state.finished = True
        if completed:
            state.push("the rite is complete", theme.C_GOOD)
        _draw(stdscr, state, flicker)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            _handle_keys(stdscr, ctl, state)
            if ctl.quit:
                break
            _draw(stdscr, state, flicker)
            time.sleep(0.1)

        if "error" in result and ctl.quit is False:
            state.push(result["error"], theme.C_WARN)
            _draw(stdscr, state, flicker)
            time.sleep(4.0)
            return 1
        return 0

    return curses.wrapper(main)
