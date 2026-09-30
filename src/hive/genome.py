"""The genome: what actually evolves.

A declarative, diffable strategy spec. Not model weights, not a blob of prompt
text. Every field is typed and bounded, which buys three things the loop
depends on:

  1. Admission is decidable. A genome missing a risk cap is rejected before it
     is ever handed capital.
  2. Parent/child comparison is a text diff, so selection pressure is legible.
  3. Runs are reproducible from (genome, seed, bar data) with no hidden state.

Fitness is realized economic performance. Everything else is plumbing.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field, fields
from typing import Literal

Style = Literal["momentum", "meanrev", "breakout"]


@dataclass(slots=True)
class Policy:
    style: Style = "momentum"
    lookback: int = 20
    entry_threshold: float = 0.5
    exit_threshold: float = 0.5
    stop_loss: float = 0.02          # fraction, e.g. 0.02 = 2%
    take_profit: float = 0.04
    max_hold_bars: int = 90
    position_frac: float = 0.5       # fraction of cash deployed on entry
    cooldown_bars: int = 5
    allow_short: bool = False        # v0.1 is long-or-flat; shorts are v0.2

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not 2 <= self.lookback <= 240:
            errs.append(f"lookback {self.lookback} outside [2, 240]")
        if not 0.0 <= self.position_frac <= 1.0:
            errs.append(f"position_frac {self.position_frac} outside [0, 1]")
        if self.stop_loss <= 0:
            errs.append("stop_loss must be > 0 (a genome with no stop has no cap)")
        if self.take_profit <= 0:
            errs.append("take_profit must be > 0")
        if not 1 <= self.max_hold_bars <= 5000:
            errs.append(f"max_hold_bars {self.max_hold_bars} outside [1, 5000]")
        if self.stop_loss >= self.take_profit:
            errs.append("stop_loss >= take_profit: reward/risk is inverted")
        if self.style not in ("momentum", "meanrev", "breakout"):
            errs.append(f"unknown style {self.style}")
        return errs


@dataclass(slots=True)
class Genome:
    policy: Policy = field(default_factory=Policy)
    model: str = "offline/rules"     # decisioner id
    id: str = ""
    parent: list[str] = field(default_factory=list)
    generation: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.fingerprint()

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self.policy), sort_keys=True) + "|" + self.model
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def valid(self) -> bool:
        return not self.policy.validate()

    def validate(self) -> list[str]:
        return self.policy.validate()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["policy"] = asdict(self.policy)
        return d

    def diff(self, other: "Genome") -> str:
        """Human-readable parent/child delta, for inspecting selection pressure."""
        a, b = asdict(self.policy), asdict(other.policy)
        out = []
        for f in fields(Policy):
            name = f.name
            if a[name] != b[name]:
                out.append(f"{name}: {a[name]} -> {b[name]}")
        if self.model != other.model:
            out.append(f"model: {self.model} -> {other.model}")
        return "\n".join(out)

    def clone(self) -> "Genome":
        return Genome(
            policy=Policy(**asdict(self.policy)),
            model=self.model,
            generation=self.generation,
            notes=self.notes,
        )


# ---- variation operators ------------------------------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mutate(genome: Genome, rng: random.Random, strength: float = 1.0) -> Genome:
    """Perturb continuous knobs and, occasionally, a discrete choice."""
    child = genome.clone()
    p = child.policy
    scale = 0.15 * strength

    p.lookback = int(_clip(round(p.lookback + rng.gauss(0, 6 * strength)), 2, 240))
    p.entry_threshold = round(
        _clip(p.entry_threshold + rng.gauss(0, scale), 0.01, 3.0), 4
    )
    p.exit_threshold = round(
        _clip(p.exit_threshold + rng.gauss(0, scale), 0.01, 3.0), 4
    )
    p.stop_loss = round(_clip(p.stop_loss * (1 + rng.gauss(0, 0.2 * strength)), 0.001, 0.5), 5)
    p.take_profit = round(
        _clip(p.take_profit * (1 + rng.gauss(0, 0.2 * strength)), 0.002, 1.0), 5
    )
    p.max_hold_bars = int(
        _clip(round(p.max_hold_bars * (1 + rng.gauss(0, 0.25 * strength))), 1, 5000)
    )
    p.position_frac = round(
        _clip(p.position_frac + rng.gauss(0, 0.15 * strength), 0.01, 1.0), 4
    )
    p.cooldown_bars = int(_clip(round(p.cooldown_bars + rng.gauss(0, 4 * strength)), 0, 500))

    if rng.random() < 0.15 * strength:
        p.style = rng.choice(["momentum", "meanrev", "breakout"])  # type: ignore[assignment]

    # Self-repair. stop_loss and take_profit are jittered independently, so
    # mutation can invert the reward/risk relationship. A variation operator that
    # emits invalid offspring half the time burns search budget for nothing.
    if p.take_profit <= p.stop_loss:
        p.take_profit = round(p.stop_loss * rng.uniform(1.1, 4.0), 5)

    child.parent = [genome.id]
    return child


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Recombine policy fields. Keeps each parent's caps rather than averaging
    them, because averaging a 1% stop with a 20% stop is how a child ends up
    with no stop at all."""
    child = Genome()
    pa, pb = asdict(a.policy), asdict(b.policy)
    for f in fields(Policy):
        name = f.name
        setattr(child.policy, name, pa[name] if rng.random() < 0.5 else pb[name])
    child.model = a.model if rng.random() < 0.5 else b.model
    child.parent = [a.id, b.id]
    return child


def random_genome(rng: random.Random) -> Genome:
    style = rng.choice(["momentum", "meanrev", "breakout"])
    stop = rng.uniform(0.005, 0.05)
    return Genome(
        policy=Policy(
            style=style,  # type: ignore[arg-type]
            lookback=rng.randint(3, 120),
            entry_threshold=round(rng.uniform(0.05, 2.0), 4),
            exit_threshold=round(rng.uniform(0.05, 2.0), 4),
            stop_loss=round(stop, 5),
            take_profit=round(stop * rng.uniform(1.2, 6.0), 5),
            max_hold_bars=rng.randint(10, 400),
            position_frac=round(rng.uniform(0.05, 1.0), 4),
            cooldown_bars=rng.randint(0, 60),
        )
    )


def hand_seeded() -> list[Genome]:
    """Deliberately diverse starting population.

    Random seeds waste early epochs and make results hard to read. Starting with
    a spread of recognizable strategies means any fitness gain is attributable
    to a real behavioural change rather than noise.
    """
    return [
        Genome(policy=Policy(style="momentum", lookback=10, entry_threshold=0.3,
                            exit_threshold=0.2, stop_loss=0.015, take_profit=0.04,
                            position_frac=0.6), notes="fast momentum"),
        Genome(policy=Policy(style="meanrev", lookback=40, entry_threshold=0.8,
                            exit_threshold=0.4, stop_loss=0.02, take_profit=0.05,
                            position_frac=0.4), notes="slow mean reversion"),
        Genome(policy=Policy(style="breakout", lookback=60, entry_threshold=0.35,
                             exit_threshold=0.2, stop_loss=0.025, take_profit=0.08,
                             position_frac=0.5), notes="breakout"),
        Genome(policy=Policy(style="momentum", lookback=120, entry_threshold=0.45,
                             exit_threshold=0.25, stop_loss=0.03, take_profit=0.10,
                             position_frac=0.3), notes="patient trend"),
        Genome(policy=Policy(style="meanrev", lookback=8, entry_threshold=0.4,
                             exit_threshold=0.25, stop_loss=0.008, take_profit=0.015,
                             position_frac=0.9, cooldown_bars=2), notes="scalper"),
        Genome(policy=Policy(style="momentum", lookback=25, entry_threshold=0.25,
                             exit_threshold=0.15, stop_loss=0.012, take_profit=0.03,
                             position_frac=0.7, max_hold_bars=45), notes="day trader"),
        Genome(policy=Policy(style="momentum", lookback=45, entry_threshold=0.18,
                             exit_threshold=0.12, stop_loss=0.018, take_profit=0.055,
                             position_frac=0.45, max_hold_bars=150,
                             cooldown_bars=10), notes="patient momentum"),
    ]
