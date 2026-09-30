"""Append-only double-entry ledger.

Every economic event the system observes lands here, balanced. The fitness
function reads only from the ledger, never from live broker state, so the
scoring path is auditable and replayable.

Accounts
--------
cash        assets, positive when held
position    position cost basis, positive when long
realized    earnings, positive when profitable
fees        costs, negative
inference   costs, negative
equity      external capital in/out, so deposits balance
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

CASH = "cash"
POSITION = "position"
REALIZED = "realized_pnl"   # gains credited here, so negative when profitable
FEES = "fees"
INFERENCE = "inference_cost"
EQUITY = "equity"

EntryKind = Literal["fill", "fee", "mark", "inference", "deposit"]


@dataclass(slots=True)
class Posting:
    account: str
    amount: float  # signed, in quote currency


@dataclass(slots=True)
class Entry:
    ts: datetime
    kind: EntryKind
    symbol: str
    postings: list[Posting]
    meta: dict = field(default_factory=dict)

    @property
    def balanced(self) -> bool:
        return abs(sum(p.amount for p in self.postings)) < 1e-9


class Ledger:
    def __init__(self, initial_cash: float = 0.0) -> None:
        self.entries: list[Entry] = []
        self.initial_cash = initial_cash

    def post(
        self,
        ts: datetime,
        kind: EntryKind,
        symbol: str,
        postings: list[Posting],
        **meta: object,
    ) -> Entry:
        entry = Entry(ts, kind, symbol, postings, dict(meta))
        if not entry.balanced:
            drift = sum(p.amount for p in entry.postings)
            raise ValueError(
                f"unbalanced entry at {ts.isoformat()} ({kind}): off by {drift:.10f}"
            )
        self.entries.append(entry)
        return entry

    def seed_cash(self, amount: float, ts: datetime, symbol: str = "-") -> None:
        self.initial_cash = amount
        self.post(ts, "deposit", symbol, [Posting(CASH, amount), Posting(EQUITY, -amount)])

    # ---- balances ----------------------------------------------------
    def balance(self, account: str) -> float:
        return sum(
            p.amount for e in self.entries for p in e.postings if p.account == account
        )

    def quantity(self) -> float:
        """Signed share count. Positive is long."""
        return sum(
            float(e.meta.get("qty", 0.0)) * float(e.meta.get("sign", 1.0))
            for e in self.entries
            if e.kind == "fill"
        )

    def net_worth(self, mark_price: float | None = None) -> float:
        """Cash plus marked position value, or cash plus cost basis if unmarked."""
        return self.balance(CASH) + self.position_value(mark_price)

    def position_value(self, mark_price: float | None) -> float:
        if mark_price is None:
            return self.balance(POSITION)
        return self.quantity() * mark_price

    def pnl(self, mark_price: float | None = None) -> float:
        """Net P&L after fees and inference spend, versus starting cash."""
        return self.net_worth(mark_price) - self.initial_cash

    def realized_pnl(self) -> float:
        """Realized P&L from closed trades.

        Gains are credited to the earnings account, so reads as the negated
        balance. Gross of fees, which are tracked separately.
        """
        return -self.balance(REALIZED)

    def total_costs(self) -> float:
        """Cash consumed by trading frictions and inference.

        Cost accounts are posted as positive magnitudes (debit to expense, credit
        to cash), so they are read directly, not negated.
        """
        return self.fees_paid() + self.inference_spend()

    def inference_spend(self) -> float:
        return max(0.0, self.balance(INFERENCE))

    def fees_paid(self) -> float:
        return max(0.0, self.balance(FEES))

    # ---- persistence -------------------------------------------------
    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w") as fh:
            for e in self.entries:
                fh.write(
                    json.dumps(
                        {
                            "ts": e.ts.isoformat(),
                            "kind": e.kind,
                            "symbol": e.symbol,
                            "postings": [asdict(p) for p in e.postings],
                            "meta": e.meta,
                        }
                    )
                    + "\n"
                )
