"""Paper broker.

Simulates fills against bar data with realistic frictions. This is the only
place orders are executed, and every effect it has on wealth is posted to the
ledger. Long-only, long-or-flat: shorts are a v0.2 change and the state here is
shaped so they slot in without a rewrite.

Deliberately pessimistic. Slippage scales with order size, the spread is paid on
every entry and exit, and there is a fee per side. A simulator that assumes free
liquidity will report a Sharpe no real venue ever reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .ledger import CASH, FEES, INFERENCE, POSITION, REALIZED, Ledger, Posting
from .market import Bar


@dataclass(slots=True)
class BrokerConfig:
    initial_cash: float = 10_000.0
    fee_bps: float = 5.0          # per side, bps of notional
    slippage_bps: float = 3.0     # base, adverse
    impact_bps: float = 40.0      # scales with size as fraction of cash
    max_participation: float = 0.10  # cap on notional as fraction of bar volume
    allow_fractional: bool = True


@dataclass(slots=True)
class Fill:
    ts: datetime
    symbol: str
    qty: float
    price: float          # executed price, slippage included
    notional: float
    fee: float
    slippage_cost: float
    reason: str


class PaperBroker:
    def __init__(self, ledger: Ledger, config: BrokerConfig | None = None) -> None:
        self.ledger = ledger
        self.config = config or BrokerConfig()
        self.avg_cost = 0.0
        self.turnover = 0.0
        self.fills: list[Fill] = []
        self.rejected: list[tuple[datetime, str, str]] = []
        ledger.seed_cash(self.config.initial_cash, datetime(1970, 1, 1))

    # ---- pricing -----------------------------------------------------
    def _slippage(self, notional: float, bar: Bar) -> float:
        cfg = self.config
        participation = notional / max(bar.volume * bar.close, 1e-9)
        participation = min(participation, cfg.max_participation)
        bps = cfg.slippage_bps + cfg.impact_bps * participation
        return bps / 10_000.0

    def _can_afford(self, notional: float, fee: float) -> bool:
        # Relative tolerance: exact comparison rejects legitimate fills on float
        # dust alone (10000.000000000002 > 10000.0).
        cash = self.ledger.balance(CASH)
        return cash >= notional + fee - max(1e-6, abs(cash) * 1e-12)

    # ---- trading -----------------------------------------------------
    def buy(
        self,
        bar: Bar,
        frac: float,
        symbol: str,
        reason: str = "entry",
    ) -> Fill | None:
        """Buy a fraction of available cash, respecting costs and cash on hand."""
        if frac <= 0:
            return None
        slip = self._slippage(self.config.initial_cash * frac, bar)
        px = bar.close * (1.0 + slip)
        budget = self.ledger.balance(CASH)
        # Solve qty*px*(1+fee) <= budget
        gross = budget / (1.0 + self.config.fee_bps / 10_000.0)
        qty = gross / px
        if not self.config.allow_fractional:
            qty = float(int(qty))
        if qty <= 0:
            self.rejected.append((bar.ts, symbol, "insufficient cash"))
            return None

        notional = qty * px
        fee = notional * self.config.fee_bps / 10_000.0
        cost = notional + fee
        if not self._can_afford(notional, fee):
            self.rejected.append((bar.ts, symbol, "insufficient cash"))
            return None

        self.ledger.post(
            bar.ts, "fill", symbol,
            [Posting(CASH, -notional), Posting(POSITION, notional)],
            qty=qty, sign=1.0, price=px, reason=reason, side="buy",
        )
        self.ledger.post(
            bar.ts, "fee", symbol,
            [Posting(CASH, -fee), Posting(FEES, fee)],
            fee=fee, side="buy",
        )

        prev_qty = self.ledger.quantity() - qty
        self.avg_cost = (
            (self.avg_cost * prev_qty + notional) / self.ledger.quantity()
            if self.ledger.quantity() > 0
            else notional / qty
        )
        slip_cost = notional * slip
        self.turnover += notional
        fill = Fill(bar.ts, symbol, qty, px, notional, fee, slip_cost, reason)
        self.fills.append(fill)
        return fill

    def sell(
        self,
        bar: Bar,
        frac: float,
        symbol: str,
        reason: str = "exit",
    ) -> Fill | None:
        """Sell a fraction of the long position."""
        held = self.ledger.quantity()
        if held <= 0 or frac <= 0:
            return None
        qty = held * min(frac, 1.0)
        if not self.config.allow_fractional:
            qty = float(int(qty))
        if qty <= 0:
            return None

        slip = self._slippage(qty * bar.close, bar)
        px = bar.close * (1.0 - slip)
        notional = qty * px
        fee = notional * self.config.fee_bps / 10_000.0

        cost_basis = self.avg_cost * qty
        gross = notional - cost_basis

        # Sale realizes a gain, so the difference between proceeds and cost
        # basis is credited to the earnings account. Without that third posting
        # the entry is unbalanced and the ledger rejects it.
        self.ledger.post(
            bar.ts, "fill", symbol,
            [
                Posting(CASH, notional),
                Posting(POSITION, -cost_basis),
                Posting(REALIZED, -gross),
            ],
            qty=qty, sign=-1.0, price=px, reason=reason, side="sell", pnl=gross,
        )
        self.ledger.post(
            bar.ts, "fee", symbol,
            [Posting(CASH, -fee), Posting(FEES, fee)],
            fee=fee, side="sell",
        )

        self.turnover += notional
        fill = Fill(bar.ts, symbol, -qty, px, notional, fee, notional * slip, reason)
        self.fills.append(fill)
        return fill

    # ---- non-trading costs -------------------------------------------
    def record_inference(self, ts: datetime, amount: float, symbol: str = "-", **meta) -> None:
        """Model/provider spend. Counted as a cost, because it is one."""
        if amount <= 0:
            return
        self.ledger.post(
            ts, "inference", symbol,
            [Posting(CASH, -amount), Posting(INFERENCE, amount)],
            **meta,
        )

    def mark(self, bar: Bar, symbol: str) -> None:
        """Record a mark-to-market observation for drawdown tracking."""
        self.ledger.post(
            bar.ts, "mark", symbol, [],
            equity=self.ledger.net_worth(bar.close), price=bar.close,
        )
