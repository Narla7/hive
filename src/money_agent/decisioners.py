"""Decisioners: what actually decides to buy or sell.

Two implementations of one interface.

  RulesDecisioner   deterministic, no network, no key. The default, and the
                    reason the harness can be run and tested anywhere. Fitness
                    is computed from the ledger, not the model, so a run with
                    zero API keys is a real experiment.

  LLMDecisioner     any OpenAI-compatible chat endpoint (OpenRouter, OpenCode
                    Zen/Go, Claude Code, Codex, Ollama, vLLM) or Anthropic's
                    Messages API. Reports token cost back to the broker so
                    inference spend enters the fitness function.

The LLM is a *proposer*, not an authority. Its output is a target weight which
then passes the gates, so an unconstrained model cannot exceed a cap by wanting
to.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .genome import Genome
from .market import Bar

# Rough per-1M-token USD, used only to put a number on inference spend so the
# fitness function can charge for it. Replace with your provider's real rates.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.6-luna": (1.0, 8.0),
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "gemini-flash": (0.30, 2.50),
    "glm-5.3-flash": (0.20, 0.80),
    "kimi-k3": (0.60, 2.50),
    "deepseek-v4-flash": (0.10, 0.40),
    "_default": (1.0, 3.0),
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, PRICING["_default"])
    return (in_tok * pin + out_tok * pout) / 1_000_000


@dataclass(slots=True)
class Decision:
    target_weight: float   # in [-1, 1]; >0 long, <0 short, 0 flat
    reason: str = ""
    raw_cost: float = 0.0

    def clamped(self) -> "Decision":
        return Decision(max(-1.0, min(1.0, self.target_weight)), self.reason, self.raw_cost)


@dataclass(slots=True)
class AgentState:
    """Runtime position state, passed explicitly.

    Deliberately not stored on the genome: a genome is an identity, and runtime
    state leaking into it changes its fingerprint, which breaks parent/child
    diffing and parentage.
    """
    weight: float = 0.0     # position value as a fraction of net worth
    held_bars: int = 0
    entry_price: float = 0.0
    equity: float = 0.0


class Decisioner(Protocol):
    id: str

    def decide(self, genome: Genome, bars: list[Bar], state: AgentState) -> Decision: ...


# ---- offline rules ------------------------------------------------------

class RulesDecisioner:
    """Classic technical strategy expressed in genome terms.

    Three styles, so the initial population has genuinely different behaviour
    to select between rather than cosmetic parameter differences.
    """

    def __init__(self, id: str = "offline/rules") -> None:
        self.id = id

    def _signal(self, genome: Genome, bars: list[Bar]) -> float:
        p = genome.policy
        if len(bars) < p.lookback + 1:
            return 0.0
        window = bars[-(p.lookback + 1):-1]
        closes = [b.close for b in window]
        last = closes[-1]
        sma = sum(closes) / len(closes)
        high = max(b.high for b in window)
        low = min(b.low for b in window)
        span = max(high - low, 1e-9)

        if p.style == "momentum":
            return (last - sma) / span
        if p.style == "meanrev":
            return -((last - sma) / span) * 2.0
        # breakout
        return (last - low) / span - 0.5

    def decide(self, genome: Genome, bars: list[Bar], state: AgentState) -> Decision:
        p = genome.policy
        sig = self._signal(genome, bars)
        last_close = bars[-1].close if bars else 0.0

        if state.weight > 0:
            entry_price = state.entry_price or last_close
            change = (last_close - entry_price) / max(entry_price, 1e-9)
            if change <= -p.stop_loss:
                return Decision(0.0, f"stop_loss {change:+.2%}")
            if change >= p.take_profit:
                return Decision(0.0, f"take_profit {change:+.2%}")
            if state.held_bars >= p.max_hold_bars:
                return Decision(0.0, "max_hold")
            if abs(sig) < p.exit_threshold:
                return Decision(0.0, f"exit_signal {sig:+.3f}")

        if sig >= p.entry_threshold:
            return Decision(p.position_frac, f"entry {p.style} {sig:+.3f}")
        return Decision(0.0, f"flat {sig:+.3f}")


# ---- LLM ----------------------------------------------------------------

SYSTEM = """You are a disciplined intraday trading agent. Given recent price bars and your strategy parameters, return ONLY a JSON object:
{"target_weight": <float -1..1>, "reason": "<short>"}
target_weight is the fraction of capital to be long (>0), short (<0), or flat (0). Be conservative. No prose outside the JSON."""


class LLMDecisioner:
    """Any OpenAI-compatible endpoint. Reports cost to the broker."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai",
        max_bars: int = 40,
        timeout: float = 30.0,
    ) -> None:
        self.id = f"llm/{model}"
        self.model = model
        self.provider = provider
        self.max_bars = max_bars
        self.timeout = timeout
        self.base_url = (base_url or os.environ.get("MA_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("MA_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        self.total_cost = 0.0
        self.calls = 0
        if not self.api_key:
            raise ValueError(
                "no API key: set MA_API_KEY or pass --api-key. "
                "Use --model offline/rules to run with no key at all."
            )

    def _prompt(self, genome: Genome, bars: list[Bar], state: AgentState) -> str:
        p = genome.policy
        recent = bars[-self.max_bars:]
        lines = [
            f"symbol={bars[-1].ts:%Y-%m-%d} time-of-session unknown",
            f"style={p.style} lookback={p.lookback} entry_threshold={p.entry_threshold}",
            f"exit_threshold={p.exit_threshold} stop_loss={p.stop_loss:.3f} take_profit={p.take_profit:.3f}",
            f"current_position_weight={state.weight:.2f} held_bars={state.held_bars}",
            "bars (ts,open,high,low,close):",
        ]
        for b in recent:
            lines.append(f"{b.ts:%H:%M} {b.open:.2f} {b.high:.2f} {b.low:.2f} {b.close:.2f}")
        return "\n".join(lines)

    def decide(self, genome: Genome, bars: list[Bar], state: AgentState) -> Decision:
        if not bars:
            return Decision(0.0, "no data")
        try:
            out, in_tok, out_tok = self._call(self._prompt(genome, bars, state))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            return Decision(state.weight, f"llm_error: {type(exc).__name__}", 0.0)

        cost = estimate_cost(self.model, in_tok, out_tok)
        self.total_cost += cost
        self.calls += 1
        try:
            payload = json.loads(out[out.index("{"):out.rindex("}") + 1])
            weight = float(payload.get("target_weight", 0.0))
            reason = str(payload.get("reason", ""))[:120]
        except (ValueError, json.JSONDecodeError):
            return Decision(state.weight, "unparseable", cost)
        return Decision(weight, reason, cost).clamped()

    def _call(self, user: str) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            return self._call_anthropic(user)
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "seed": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"],
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )

    def _call_anthropic(self, user: str) -> tuple[str, int, int]:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 256,
            "temperature": 0.0,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages", data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage", {})
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        ), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def build_decisioner(model: str, **kw) -> Decisioner:
    """`offline/rules` needs nothing; anything else goes over the network."""
    if model in ("offline/rules", "rules", "none"):
        return RulesDecisioner()
    return LLMDecisioner(model, **kw)
