"""Visual language for the interface.

The renderer in `tui.py` produces plain strings; everything about how they look
lives here. Keeping presentation separate means the layout is testable without a
terminal, and the palette can change without touching the loop.
"""

from __future__ import annotations

import random
import shutil
import time

# ---- palette (curses colour pairs) ---------------------------------------
C_BG = 1
C_TEXT = 2
C_DIM = 3
C_BLOOD = 4
C_BLOOD_DIM = 5
C_GOOD = 6
C_WARN = 7
C_BONE = 8
C_FLASH = 9
C_RITUAL = 10
C_CORRUPT = 11


def init_pairs() -> None:
    """Colour pairs. Safe to call more than once."""
    import curses

    curses.start_color()
    try:
        curses.use_default_colors()
        default_bg = -1
    except curses.error:
        default_bg = curses.COLOR_BLACK

    def pair(idx: int, fg: int, bg: int = default_bg) -> None:
        try:
            curses.init_pair(idx, fg, bg)
        except curses.error:
            pass

    pair(C_TEXT, curses.COLOR_WHITE)
    pair(C_DIM, curses.COLOR_BLUE)
    pair(C_BLOOD, curses.COLOR_RED)
    pair(C_BLOOD_DIM, curses.COLOR_RED, curses.COLOR_BLACK)
    pair(C_GOOD, curses.COLOR_GREEN)
    pair(C_WARN, curses.COLOR_YELLOW)
    pair(C_BONE, curses.COLOR_WHITE)
    pair(C_RITUAL, curses.COLOR_MAGENTA)
    pair(C_CORRUPT, curses.COLOR_CYAN)


SIGIL = r"""
    ____  ____  ____  _____________   ____  __
   / __ \/ __ \/ __ \/ ____/ ____/ | / /\ \/ /
  / /_/ / /_/ / / / / / __/ __/ /  |/ /  \  /
 / ____/ _, _/ /_/ / /_/ / /___/ /|  /   / /
/_/   /_/ |_|\____/\____/_____/_/ |_/   /_/
"""

# Used when the terminal is too short for the full sigil. Terminals vary a lot,
# and a 24-row window with a 14-row banner leaves no room for actual data.
SIGIL_COMPACT = r"""
  ____  ____  ____    selection
 / __ \/ __ \/ __ \      picks
/ /_/ / /_/ / / /_/     the fitter
\____/\____/\____/     survive
"""

GLITCH_CHARS = "▓▒░█▚▞/\\|_-=+*#@$%&"

# Lines the loop narrates. Chosen to be funny-ominous rather than actually
# frightening; the numbers underneath are the real content.
RITUAL_LINES = [
    "the rite begins",
    "selection pressure rises",
    "a mutation takes hold",
    "the weak are culled",
    "something is adapting",
    "the lineage forks",
    "ten thousand generations of this",
    "the fitter survive. that is the whole idea.",
    "do not let it notice the cap",
    "variation is not optional",
    "one child did not survive",
    "the population drifts",
    "you are inside the fitness function now",
    "every fill is a gene",
    "it learned your thresholds",
    "the drawdown is a throat",
    "stop looking at the exit",
    "it is almost profitable",
    "the eyes are on the ledger",
    "the chamber seals",
]

FAIL_LINES = [
    "the gate closed",
    "something was quarantined",
    "a genome could not be contained",
    "it broke the rules. it is contained.",
    "the drawdown exceeded its welcome",
]


def term_width(default: int = 100) -> int:
    try:
        return max(60, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


def center(text: str, width: int) -> str:
    """Centre, then pad to exactly `width`.

    Rounding the left pad down leaves the odd pixel on the right, which makes
    banner text look visibly off-centre rather than a hair off.
    """
    if len(text) >= width:
        return text[:width].ljust(width)
    left = (width - len(text)) // 2
    return " " * left + text.ljust(width - left)


def pad(text: str, width: int) -> str:
    return text[:width].ljust(width)


def visible_len(text: str) -> int:
    """Length ignoring ANSI-ish escapes. Kept simple: no escapes are emitted."""
    return len(text)


# ---- flicker -------------------------------------------------------------
class Flicker:
    """Occasional brightness dips, the way bad fluorescent lighting behaves.

    Rate-limited by time so a fast loop cannot turn the screen into a strobe.
    """

    def __init__(self, chance: float = 0.06, min_gap: float = 0.35) -> None:
        self.chance = chance
        self.min_gap = min_gap
        self._last = 0.0
        self._now_dark = False

    def step(self) -> bool:
        """True when the output should render dimmer this frame."""
        now = time.monotonic()
        if now - self._last < self.min_gap:
            return self._now_dark
        self._last = now
        self._now_dark = random.random() < self.chance
        return self._now_dark


def corrupt(text: str, intensity: float = 0.0) -> str:
    """Sprinkle glitch characters through a string.

    Intensity 0 returns the input untouched, which is what the tests assert and
    what makes this safe to call on every frame.
    """
    if intensity <= 0:
        return text
    out = []
    for ch in text:
        if ch != " " and random.random() < intensity * 0.35:
            out.append(random.choice(GLITCH_CHARS))
        else:
            out.append(ch)
    return "".join(out)


def bar(value: float, width: int, lo: float = -1.0, hi: float = 1.0,
        fill: str = "█", empty: str = "·") -> str:
    """Horizontal gauge. Used for fitness and for the noise-floor comparison."""
    if width <= 0:
        return ""
    if hi <= lo:
        return fill * width
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return fill * n + empty * (width - n)


def sparkline(values: list[float], width: int = 40) -> str:
    """Unicode block sparkline. Reads as a seismograph, which is the point."""
    if not values:
        return " " * width
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return blocks[0] * min(len(values), width)
    span = hi - lo
    chars = [blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))]
             for v in values[-width:]]
    return "".join(chars)


def art_lines(height: int) -> list[str]:
    """Pick the banner that fits. A 24-row window cannot spare 12 for art."""
    full = SIGIL.strip("\n").splitlines()
    if len(full) + 5 <= max(0, height - 14):
        return full
    return SIGIL_COMPACT.strip("\n").splitlines()


def ritual_line(rng: random.Random, generation: int) -> str:
    pool = RITUAL_LINES if generation > 0 else ["the rite begins"]
    return rng.choice(pool)


def fail_line(rng: random.Random) -> str:
    return rng.choice(FAIL_LINES)
