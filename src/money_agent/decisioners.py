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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
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


# Cloudflare (in front of Zen) rejects the default `Python-urllib/x.y` agent
# with error 1010, "access based on your browser's signature". Identify honestly
# rather than impersonating a browser.
USER_AGENT = "money-agent/0.1 (+https://github.com/Narla7/money-agent)"


@dataclass
class CallBudget:
    """Hard cap on model calls, shared across every genome in one epoch.

    Free tiers and rate-capped subscriptions both fail by truncating, not by
    erroring. A generation that gets cut off mid-way otherwise produces a
    plausible-looking result built on partial data, which is worse than a crash.
    """
    limit: int | None = None
    spent: int = 0
    blocked: int = 0

    def take(self) -> bool:
        if self.limit is not None and self.spent >= self.limit:
            self.blocked += 1
            return False
        self.spent += 1
        return True

    def snapshot(self) -> tuple[int, int]:
        return (self.spent, self.blocked)

    def blocked_since(self, snap: tuple[int, int]) -> int:
        return self.blocked - snap[1]


def _probe_bars(n: int = 12) -> list[Bar]:
    """A tiny synthetic series for the health check. Small on purpose: the probe
    should be fast and effectively free, not a benchmark."""
    base = datetime(2026, 1, 5, 14, 30)
    out = []
    px = 100.0
    for i in range(n):
        drift = 0.0012 * i          # a gentle trend, so a model has something to see
        px = px * (1 + drift)
        out.append(Bar(
            base + timedelta(minutes=i),
            open=px, high=px * 1.001, low=px * 0.999, close=px, volume=1_000_000,
        ))
    return out


@dataclass(slots=True)
class ProbeResult:
    """Outcome of a single health-check call."""
    ok: bool
    endpoint: str
    model: str
    status: int | None = None
    elapsed_ms: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cost: float = 0.0
    raw: str = ""
    parsed: dict | None = None
    error: str = ""


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
        budget: CallBudget | None = None,
    ) -> None:
        self.id = f"llm/{model}"
        self.model = model
        self.provider = provider
        self.max_bars = max_bars
        self.timeout = timeout
        self.budget = budget or CallBudget()
        self.base_url = (base_url or os.environ.get("MA_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")

        self.api_key = (
            api_key
            or os.environ.get("MA_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENCODE_API_KEY")
            or ""
        )
        self.total_cost = 0.0
        self.calls = 0
        # Silent counters. An unparseable reply currently degrades to "hold
        # position", so without these a model that never returns JSON looks
        # exactly like a market with no edge.
        self.parse_failures = 0
        self.network_errors = 0
        self.last_raw = ""
        if not self.api_key:
            raise ValueError(
                "no API key: set MA_API_KEY, OPENROUTER_API_KEY or OPENCODE_API_KEY, "
                "or pass --api-key. Use --model offline/rules to run with no key at all."
            )

    @property
    def endpoint(self) -> str:
        return (
            f"{self.base_url}/v1/messages" if self.provider == "anthropic"
            else f"{self.base_url}/chat/completions"
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
        if not self.budget.take():
            # Budget exhausted. Hold and let the caller mark the episode
            # truncated, rather than pretending a decision was made.
            return Decision(state.weight, "budget_exhausted", 0.0)
        try:
            out, in_tok, out_tok = self._call(self._prompt(genome, bars, state))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            self.network_errors += 1
            return Decision(state.weight, f"llm_error: {type(exc).__name__}", 0.0)

        self.last_raw = out
        cost = estimate_cost(self.model, in_tok, out_tok)
        self.total_cost += cost
        self.calls += 1
        try:
            payload = json.loads(out[out.index("{"):out.rindex("}") + 1])
            weight = float(payload.get("target_weight", 0.0))
            reason = str(payload.get("reason", ""))[:120]
        except (ValueError, json.JSONDecodeError):
            self.parse_failures += 1
            return Decision(state.weight, "unparseable", cost)
        return Decision(weight, reason, cost).clamped()

    def probe(self, genome: Genome | None = None) -> ProbeResult:
        """One health-check call: does the key, URL, and JSON path work?

        A full run fires thousands of calls. Without this, a wrong key or a
        model that returns prose is discovered minutes and real money later --
        or not at all, because a parse failure degrades silently.
        """
        genome = genome or Genome()
        bars = _probe_bars()
        url = self.endpoint
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": self._prompt(genome, bars, AgentState())},
            ],
            "temperature": 0.0,
            "seed": 0,
        }).encode()
        headers = (
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
             "Content-Type": "application/json", "User-Agent": USER_AGENT}
            if self.provider == "anthropic"
            else {"Authorization": f"Bearer {self.api_key}",
                  "Content-Type": "application/json", "User-Agent": USER_AGENT}
        )
        res = ProbeResult(ok=False, endpoint=url, model=self.model)
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            res.status = 200
        except urllib.error.HTTPError as exc:
            res.status = exc.code
            detail = exc.read().decode("utf-8", "replace")[:400]
            res.error = f"HTTP {exc.code}: {detail}"
            res.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return res
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return res

        res.elapsed_ms = int((time.monotonic() - t0) * 1000)
        if self.provider == "anthropic":
            res.raw = "".join(
                b.get("text", "") for b in payload.get("content", [])
                if b.get("type") == "text"
            )
            usage = payload.get("usage", {})
            res.in_tokens = int(usage.get("input_tokens", 0))
            res.out_tokens = int(usage.get("output_tokens", 0))
        else:
            try:
                res.raw = payload["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                res.error = f"unexpected response shape: {str(payload)[:300]}"
                return res
            usage = payload.get("usage", {})
            res.in_tokens = int(usage.get("prompt_tokens", 0))
            res.out_tokens = int(usage.get("completion_tokens", 0))

        res.cost = estimate_cost(self.model, res.in_tokens, res.out_tokens)
        try:
            res.parsed = json.loads(res.raw[res.raw.index("{"):res.raw.rindex("}") + 1])
            res.ok = True
        except (ValueError, json.JSONDecodeError):
            res.error = "response was not parseable JSON"
        return res

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
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
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
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage", {})
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        ), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def build_decider(model: str, **kw) -> Decisioner:
    """`offline/rules` needs nothing; anything else goes over the network."""
    if model in ("offline/rules", "rules", "none"):
        return RulesDecisioner()
    return LLMDecisioner(model, **kw)
