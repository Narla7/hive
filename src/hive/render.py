"""Dashboard rendering.

Every function here returns a list of `(text, colour_pair)` rows. Nothing in
this module imports curses or touches a terminal, so the layout is testable
headlessly and the curses driver stays a thin shim over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import theme


@dataclass
class LogLine:
    text: str
    colour: int = theme.C_DIM


@dataclass
class TuiState:
    """Everything the screen draws. Mutated by the driver, read by the renderer."""
    generation: int = 0
    total_generations: int = 0
    population: int = 0
    paused: bool = False
    finished: bool = False
    speed: float = 1.0
    model: str = "offline/rules"
    mean_history: list[float] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    roster: list[dict] = field(default_factory=list)
    log: list[LogLine] = field(default_factory=list)
    calls: int = 0
    cost: float = 0.0
    parse_failures: int = 0
    quarantined: int = 0
    truncated: int = 0
    holdout: str = ""
    holdout_done: bool = False
    dark: bool = False
    rng_seed: int = 0

    def push(self, text: str, colour: int = theme.C_DIM) -> None:
        self.log.append(LogLine(text, colour))
        del self.log[:-200]

    # Safe current values. The first frame is drawn before generation 0 has
    # finished, so the histories are empty and any direct [-1] is a crash.
    @property
    def mean_now(self) -> float:
        return self.mean_history[-1] if self.mean_history else 0.0

    @property
    def best_now(self) -> float:
        return self.best_history[-1] if self.best_history else 0.0

    @property
    def verdict(self) -> str:
        if len(self.mean_history) < 2:
            return "AWAKENING"
        k = max(1, len(self.mean_history) // 3)
        early = sum(self.mean_history[:k]) / k
        late = sum(self.mean_history[-k:]) / k
        return "RISING" if late > early else "STAGNANT"


# ---- frame assembly ------------------------------------------------------

def build_frame(state: TuiState, width: int, height: int) -> list[list[tuple[str, int]]]:
    """The whole screen as coloured rows, ready to blit."""
    # Every helper returns a list of ROWS; a row is a list of (text, colour)
    # segments. Mixing "one row" and "list of rows" return types is the fastest
    # way to flatten a whole screen into a single line.
    lines: list[list[tuple[str, int]]] = []
    add = lines.append

    lines.extend(_header(state, width, height))
    add([("", theme.C_TEXT)])

    # Body: roster on the left, gauges and log on the right.
    # Footer, key hints, and a blank separator sit below the body.
    body_height = max(4, height - len(lines) - 4)
    left_width = min(52, max(34, width // 2 - 2))
    right_width = max(24, width - left_width - 3)

    left = _roster(state, left_width, body_height)
    log_height = 7 if body_height >= 12 else 3
    right = (
        _gauge_panel(state, right_width, body_height - log_height)
        + _log(state, right_width, log_height)
    )

    for i in range(body_height):
        # Splice each panel row's segments into one frame row. Pushing the row
        # itself as a single element would nest rows inside rows and break blit.
        lrow = left[i] if i < len(left) else [("", theme.C_TEXT)]
        rrow = right[i] if i < len(right) else [("", theme.C_TEXT)]
        add(list(lrow) + [(" │ ", theme.C_BLOOD_DIM)] + list(rrow))

    add([("", theme.C_TEXT)])
    lines.extend(_footer(state, width))
    lines.extend(_keys(state, width))
    return lines


def _header(state: TuiState, width: int, height: int) -> list[list[tuple[str, int]]]:
    dim = theme.C_BLOOD_DIM if state.dark else theme.C_BLOOD
    title = "H I V E"
    sub = "THE SWARM // EVOLUTIONARY PROFIT MACHINE"
    rows: list[list[tuple[str, int]]] = []
    rows.append([(theme.pad("═" * width, width), dim)])
    for line in theme.art_lines(height):
        cropped = theme.corrupt(theme.center(line, width), 0.05 if state.dark else 0.0)
        rows.append([(theme.pad(cropped, width), dim)])
    rows.append([(theme.pad(theme.center(title, width), width), theme.C_BLOOD)])
    rows.append([(theme.pad(theme.center(sub, width), width), theme.C_DIM)])
    return rows


def _roster(state: TuiState, width: int, height: int) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    head = f" THE QUEEN  gen {state.generation}/{state.total_generations} "
    rows.append([(theme.pad(head, width), theme.C_RITUAL)])
    rows.append([(theme.pad("─" * width, width), theme.C_BLOOD_DIM)])

    name_w = max(12, min(20, width - 30))
    hdr = (
        theme.pad(" entity", name_w)
        + theme.pad("fitness", 10)
        + theme.pad("return", 9)
        + "  "
    )
    rows.append([(theme.pad(hdr, width), theme.C_DIM)])

    visible = state.roster[: max(1, height - 4)]
    for entry in visible:
        colour = theme.C_GOOD if entry.get("fitness", 0) > 0 else theme.C_BLOOD_DIM
        name = entry.get("name", "?")[: name_w - 1]
        line = (
            theme.pad(" " + name, name_w)
            + theme.pad(f"{entry.get('fitness', 0.0):>9.3f}", 10)
            + theme.pad(f"{entry.get('ret', 0.0):>+8.2%}", 9)
            + "  "
        )
        rows.append([(theme.pad(line, width), colour)])

    if not visible:
        rows.append([(theme.pad("  nothing has been summoned yet", width), theme.C_DIM)])
    return rows


def _gauge_panel(state: TuiState, width: int, height: int) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    if height <= 0:
        return rows
    rows.append([(theme.pad(" VITALS", width), theme.C_RITUAL)])
    rows.append([(theme.pad("─" * width, width), theme.C_BLOOD_DIM)])

    hist = state.mean_history or [0.0]
    lo, hi = min(hist), max(hist)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0

    metrics = [
        ("vitality", state.mean_now,
         f"{state.mean_now:+.4f}" if state.mean_history else "--", lo, hi),
        ("peak", state.best_now,
         f"{state.best_now:+.4f}" if state.best_history else "--", lo, hi),
    ]
    for label, value, text, mlo, mhi in metrics:
        gauge_w = max(6, width - 26)
        rows.append([
            (theme.pad(f" {label} ", 12), theme.C_DIM),
            (theme.pad(theme.bar(value, gauge_w, mlo, mhi), gauge_w), theme.C_BLOOD),
            (theme.pad(f"{text:>9}", 12), theme.C_TEXT),
        ])

    trend = theme.sparkline(state.mean_history, max(10, width - 6))
    rows.append([
        (theme.pad(" trace ", 12), theme.C_DIM),
        (theme.pad(trend, width - 6), theme.C_GOOD if state.verdict == "RISING" else theme.C_WARN),
    ])

    rows.append([(theme.pad(f" omen  {state.verdict}", width),
                  theme.C_GOOD if state.verdict == "RISING" else theme.C_WARN)])
    rows.append([(theme.pad(f" breath {state.mean_now:+.3f} over "
                            f"{len(state.mean_history)} rites", width), theme.C_DIM)])

    if state.calls or state.cost:
        extra = f" calls {state.calls:,}  ${state.cost:.4f}"
        rows.append([(theme.pad(extra, width), theme.C_DIM)])
    if state.quarantined:
        rows.append([(theme.pad(f" sealed {state.quarantined} entities", width), theme.C_WARN)])
    if state.truncated:
        rows.append([(theme.pad(f" cut short {state.truncated} episodes", width), theme.C_WARN)])
    if state.parse_failures:
        rows.append([(theme.pad(f" CORRUPT {state.parse_failures} utterances", width),
                      theme.C_CORRUPT)])
    if state.holdout:
        rows.append([(theme.pad(state.holdout, width), theme.C_BLOOD)])
    return rows


def _log(state: TuiState, width: int, height: int) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    if height <= 0:
        return rows
    rows.append([(theme.pad(" WHISPERS", width), theme.C_RITUAL)])
    rows.append([(theme.pad("─" * width, width), theme.C_BLOOD_DIM)])
    entries = state.log[-max(1, height - 2):]
    for entry in entries:
        rows.append([(theme.pad(" " + entry.text[: width - 1], width), entry.colour)])
    return rows


def _footer(state: TuiState, width: int) -> list[tuple[str, int]]:
    if state.finished:
        status, colour = " THE RITE IS COMPLETE ", theme.C_GOOD
    elif state.paused:
        status, colour = " HELD BREATH — PAUSED ", theme.C_WARN
    else:
        status, colour = " IT IS STILL TRADING ", theme.C_BLOOD

    left = theme.pad(f" {state.model}  x{state.speed:g} ", max(0, width - len(status) - 2))
    row = [(left, theme.C_DIM), (status, colour)]
    return [row]


def _keys(state: TuiState, width: int) -> list[tuple[str, int]]:
    hints = "q quit   space pause   +/- speed   n next   s summary"
    return [[(theme.pad(theme.center(hints[:width], width), width), theme.C_BLOOD_DIM)]]
