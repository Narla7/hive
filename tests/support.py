"""Shared helpers for the test suite.

The loop now addresses windows in session space, so a market large enough for a
given Config has to be built explicitly. Doing that in one place keeps every
test honest about what span it actually needs.
"""

from progeny.evolution import Config
from progeny.market import SimulatedMarket
from progeny.sessions import plan_from_config


def sessions_for(cfg: Config) -> int:
    """How many sessions this configuration needs."""
    return plan_from_config(
        cfg.generations, cfg.episodes, cfg.holdout_windows,
        cfg.reselect_windows, cfg.reselect_final_windows, cfg.episode_spacing,
    ).needed


def market_for(cfg: Config, seed: int = 1234, symbol: str = "SIM") -> SimulatedMarket:
    """A simulated market sized to satisfy cfg, with a few sessions to spare."""
    return SimulatedMarket(seed=seed, n_days=sessions_for(cfg) + 5, symbol_seed=symbol)


def small_cfg(**kw) -> Config:
    """A config whose whole span fits in a handful of sessions.

    Fast tests should still exercise the real code path, including the
    two-stage re-selection, rather than stubbing it out.
    """
    base = dict(
        population=4, generations=3, episodes=2, bars=390, episode_spacing=2,
        reselect_windows=3, reselect_final_windows=4, holdout_windows=3,
        window_stride=0,
    )
    base.update(kw)
    return Config(**base)
