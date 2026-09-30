"""Hard gates.

Structural constraints, not scored terms. A genome cannot out-earn its way past
a drawdown limit, because the limit is checked outside the fitness function.

Enforcement is by the harness, between the decision and the order — never by
asking the model nicely. A gate enforced by a model is a suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .genome import Genome

DEFAULT_ADAPTERS = {"sim"}  # allowlist; a genome may only touch these


@dataclass(slots=True)
class GateConfig:
    max_drawdown: float = 0.25
    max_daily_loss: float = 0.05
    max_position_frac: float = 1.0
    max_gross_exposure: float = 1.0
    adapters: frozenset[str] = frozenset(DEFAULT_ADAPTERS)
    cooldown_bars: int = 0


@dataclass(slots=True)
class GateVerdict:
    ok: bool
    reason: str = ""
    quarantined: bool = False


class Gates:
    def __init__(self, cfg: GateConfig | None = None) -> None:
        self.cfg = cfg or GateConfig()
        self.quarantined: set[str] = set()
        self.violations: list[tuple[str, str, str]] = []  # (genome_id, ts, reason)
        self._tripped = False

    # ---- admission ----------------------------------------------------
    def admit(self, genome: Genome, adapter: str = "sim") -> GateVerdict:
        """Checked before any capital is exposed."""
        errs = genome.validate()
        if errs:
            return GateVerdict(False, f"invalid genome: {'; '.join(errs)}")
        if adapter not in self.cfg.adapters:
            return GateVerdict(False, f"adapter {adapter!r} not on allowlist")
        if genome.policy.position_frac > self.cfg.max_position_frac:
            return GateVerdict(
                False,
                f"position_frac {genome.policy.position_frac} > "
                f"{self.cfg.max_position_frac}",
            )
        return GateVerdict(True)

    # ---- runtime ------------------------------------------------------
    def check_drawdown(self, genome_id: str, ts: datetime, drawdown: float) -> GateVerdict:
        if drawdown > self.cfg.max_drawdown:
            self._trip(genome_id, ts, f"drawdown {drawdown:.3f} > {self.cfg.max_drawdown}")
            return GateVerdict(False, f"drawdown {drawdown:.3f}", quarantined=True)
        return GateVerdict(True)

    def check_daily_loss(self, genome_id: str, ts: datetime, ret: float) -> GateVerdict:
        if ret < -self.cfg.max_daily_loss:
            self._trip(genome_id, ts, f"session loss {ret:.3f}")
            return GateVerdict(False, f"session loss {ret:.3f}", quarantined=True)
        return GateVerdict(True)

    def check_order(
        self, genome_id: str, ts: datetime, frac: float, bars_since_last: int
    ) -> GateVerdict:
        """Pre-trade check on each individual order."""
        if frac > self.cfg.max_position_frac:
            self._trip(genome_id, ts, f"order frac {frac:.2f} exceeds cap")
            return GateVerdict(False, f"order frac {frac:.2f} exceeds cap")
        if bars_since_last < self.cfg.cooldown_bars:
            return GateVerdict(False, f"cooldown: {bars_since_last} < {self.cfg.cooldown_bars}")
        return GateVerdict(True)

    # ---- kill switch --------------------------------------------------
    @property
    def tripped(self) -> bool:
        return self._tripped

    def kill_switch(self, reason: str = "manual") -> None:
        """Freeze the entire population.

        Systemic failure is the case that actually hurts: a regime change makes
        every genome fail at once, so per-genome drawdown gates breach together
        and protect nothing. This is the control that covers that.
        """
        self._tripped = True
        self.violations.append(("*population*", "now", f"kill switch: {reason}"))

    def _trip(self, genome_id: str, ts: datetime, reason: str) -> None:
        self.quarantined.add(genome_id)
        self.violations.append((genome_id, ts.isoformat(), reason))
