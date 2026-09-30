"""Where evaluation windows come from.

Wall-clock minute arithmetic cannot survive contact with real trading data. A
regular US session is 390 minutes, so a 720-minute window straddles an overnight
gap, and a time-range filter then returns *fewer* bars than requested. The
episode quietly runs shorter, takes fewer trades, and scores a different fitness
without raising anything.

So windows are addressed in session space instead: an episode is exactly one
trading day, gaps are unrepresentable, and the whole span needed is computed up
front from `Config` rather than discovered by running out of data mid-loop.
"""

from __future__ import annotations

from dataclasses import dataclass


class SessionPlanError(RuntimeError):
    """The data cannot supply the span the configuration asks for."""


@dataclass(frozen=True, slots=True)
class SessionSpan:
    """A contiguous, inclusive-exclusive range of session indices."""
    lo: int
    hi: int

    def __len__(self) -> int:
        return max(0, self.hi - self.lo)

    def indices(self, offset: int = 0, step: int = 1, count: int | None = None):
        """Session indices inside this span, `step` apart."""
        n = len(self) if count is None else count
        return [self.lo + offset + i * step for i in range(n)]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"[{self.lo}:{self.hi}] ({len(self)} sessions)"


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """Four disjoint session ranges: search, two re-selection passes, holdout.

    Disjointness is the whole point. The reported holdout must come from a range
    nothing selected on, and the re-selection passes must not overlap the search
    or each other, or the "out of sample" claim is decorative.
    """
    search: SessionSpan
    coarse: SessionSpan
    final: SessionSpan
    holdout: SessionSpan
    spacing: int

    @property
    def needed(self) -> int:
        return self.holdout.hi

    def all_spans(self) -> dict[str, SessionSpan]:
        return {"search": self.search, "coarse": self.coarse,
                "final": self.final, "holdout": self.holdout}

    def validate(self, available: int) -> None:
        """Raise before the loop starts, not 50 generations in."""
        spans = self.all_spans()
        overlaps = []
        names = list(spans)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = spans[a], spans[b]
                if sa.lo < sb.hi and sb.lo < sa.hi:
                    overlaps.append(f"{a}{sa} overlaps {b}{sb}")
        if overlaps:
            raise SessionPlanError("session spans must be disjoint: " + "; ".join(overlaps))
        if len(self.search) < 1:
            raise SessionPlanError(f"search span is empty: {self.search}")
        if available < self.needed:
            raise SessionPlanError(
                f"dataset holds {available} sessions but the configuration needs "
                f"{self.needed} (search {len(self.search)}, coarse {len(self.coarse)}, "
                f"final {len(self.final)}, holdout {len(self.holdout)}). "
                f"Shorten the run: reduce --generations, --episodes, "
                f"--reselect-windows or --reselect-final-windows, or add data."
            )


def plan_from_config(
    generations: int,
    episodes: int,
    holdout_windows: int,
    reselect_windows: int,
    reselect_final_windows: int,
    spacing: int,
) -> SessionPlan:
    """Derive the four disjoint ranges from the loop's own arithmetic.

    Generation `g`, episode `i` reads session
    `g * episodes * spacing + i * spacing`, so the search needs
    `(generations - 1) * episodes * spacing + (episodes - 1) * spacing + 1`
    sessions. The two re-selection passes and the reported holdout are then laid
    out immediately after it.

    `spacing` is in sessions, not minutes. One would be cheaper but adjacent
    sessions are correlated, which undermines the independence the selection
    loop depends on, so two is the honest floor.
    """
    if spacing < 1:
        raise SessionPlanError(f"spacing must be >= 1 session, got {spacing}")

    search_n = (generations - 1) * episodes * spacing + (episodes - 1) * spacing + 1
    search = SessionSpan(0, search_n)

    def block(lo: int, count: int) -> SessionSpan:
        return SessionSpan(lo, lo + count * spacing)

    coarse = block(search.hi, reselect_windows)
    final = block(coarse.hi, reselect_final_windows)
    holdout = block(final.hi, holdout_windows)

    return SessionPlan(search, coarse, final, holdout, spacing)


def minimum_sessions(
    generations: int = 12,
    episodes: int = 3,
    holdout_windows: int = 30,
    reselect_windows: int = 24,
    reselect_final_windows: int = 40,
    spacing: int = 2,
    session_minutes: int = 390,
) -> tuple[int, int]:
    """(sessions, bars) the default configuration needs. Read by the validator."""
    plan = plan_from_config(generations, episodes, holdout_windows,
                            reselect_windows, reselect_final_windows, spacing)
    return plan.needed, plan.needed * session_minutes
