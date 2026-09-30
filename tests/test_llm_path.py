"""Tests for the LLM path: probe, budget, and failure counters.

Every network call is mocked. No test in this file can spend money.
"""

import io
import json
import unittest
import urllib.error
from unittest import mock

from progeny.decisioners import (
    AgentState,
    CallBudget,
    Decision,
    LLMDecisioner,
    RulesDecisioner,
    _probe_bars,
    build_decider,
)
from progeny.evolution import Config, run
from tests.support import market_for, small_cfg
from progeny.fitness import Episode, evaluate
from progeny.genome import Genome

GOOD = json.dumps({"target_weight": 0.42, "reason": "momentum entry"})
PROSE = "I think the market looks bullish so maybe go long, not sure though."
NO_BRACES = "42"
BAD_JSON = "{target_weight: 0.4,}"


def _resp(payload, status=200):
    """Fake urlopen context manager returning a JSON body."""
    body = json.dumps(payload).encode()

    class R:
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return R()


def chat_payload(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 40},
    }


def anthropic_payload(text):
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1200, "output_tokens": 40},
    }


# Mocked HTTPErrors hold an open file object. Collecting and closing them in
# tearDownModule keeps ResourceWarning noise out of the run without hiding real
# warnings.
_OPEN_ERRORS: list = []


def http_error(code, body="{}"):
    def _raise(*a, **k):
        err = urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body.encode()))
        _OPEN_ERRORS.append(err)
        raise err

    return _raise


def tearDownModule():
    for err in _OPEN_ERRORS:
        try:
            err.close()
        except Exception:
            pass
    _OPEN_ERRORS.clear()


def make(model="big-pickle", **kw):
    kw.setdefault("base_url", "https://opencode.ai/zen/v1")
    kw.setdefault("api_key", "test-key")
    return LLMDecisioner(model, **kw)


class TestProbeBars(unittest.TestCase):
    def test_probe_series_is_small_and_ordered(self):
        b = _probe_bars()
        self.assertLessEqual(len(b), 20)
        self.assertEqual([x.ts for x in b], sorted(x.ts for x in b))

    def test_probe_series_has_visible_trend(self):
        b = _probe_bars()
        self.assertGreater(b[-1].close, b[0].close)


class TestProbe(unittest.TestCase):
    def test_probe_success(self):
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(GOOD))):
            r = d.probe()
        self.assertTrue(r.ok)
        self.assertEqual(r.status, 200)
        self.assertEqual(r.parsed["target_weight"], 0.42)
        self.assertEqual(r.in_tokens, 1200)
        self.assertGreaterEqual(r.elapsed_ms, 0)
        self.assertEqual(r.endpoint, "https://opencode.ai/zen/v1/chat/completions")

    def test_probe_uses_no_api_calls_when_it_fails(self):
        d = make()
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401, '{"e":"bad key"}')):
            r = d.probe()
        self.assertFalse(r.ok)
        self.assertEqual(r.status, 401)
        self.assertIn("401", r.error)
        self.assertEqual(d.calls, 0)   # probe must not pollute the counters

    def test_probe_reports_wrong_endpoint(self):
        d = make()
        with mock.patch("urllib.request.urlopen", side_effect=http_error(404, "not found")):
            r = d.probe()
        self.assertFalse(r.ok)
        self.assertEqual(r.status, 404)

    def test_probe_reports_network_error(self):
        d = make()
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
            r = d.probe()
        self.assertFalse(r.ok)
        self.assertIn("URLError", r.error)

    def test_probe_flags_prose_response(self):
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(PROSE))):
            r = d.probe()
        self.assertFalse(r.ok)
        self.assertIsNone(r.parsed)
        self.assertIn("not parseable", r.error)
        self.assertEqual(r.raw, PROSE)   # raw is kept so the user can see it

    def test_probe_flags_unexpected_shape(self):
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp({"nonsense": 1})):
            r = d.probe()
        self.assertFalse(r.ok)
        self.assertIn("unexpected response shape", r.error)

    def test_probe_anthropic_shape(self):
        d = make(model="claude-sonnet-5", provider="anthropic")
        with mock.patch("urllib.request.urlopen", return_value=_resp(anthropic_payload(GOOD))):
            r = d.probe()
        self.assertTrue(r.ok)
        self.assertEqual(r.in_tokens, 1200)
        self.assertEqual(r.out_tokens, 40)
        self.assertTrue(r.endpoint.endswith("/v1/messages"))

    def test_probe_anthropic_sends_correct_headers(self):
        d = make(model="claude-sonnet-5", provider="anthropic")
        seen = {}

        def grab(req, **kw):
            seen.update({h.lower(): v for h, v in req.headers.items()})
            return _resp(anthropic_payload(GOOD))

        with mock.patch("urllib.request.urlopen", grab):
            d.probe()
        self.assertIn("x-api-key", seen)
        self.assertIn("anthropic-version", seen)

    def test_probe_works_for_every_model_in_the_zen_catalog_shape(self):
        # big-pickle is chat/completions; the endpoint is derived, not hardcoded
        # per model, so any id works as long as the provider matches.
        for model in ("big-pickle", "glm-5.3-flash", "deepseek-v4-flash", "kimi-k3"):
            with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(GOOD))):
                self.assertTrue(make(model).probe().ok, model)


class TestDecisionParsing(unittest.TestCase):
    def _decide(self, content, d=None):
        d = d or make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(content))):
            return d.decide(Genome(), _probe_bars(), AgentState()), d

    def test_parses_valid_json(self):
        dec, d = self._decide(GOOD)
        self.assertAlmostEqual(dec.target_weight, 0.42)
        self.assertEqual(d.parse_failures, 0)
        self.assertEqual(d.calls, 1)

    def test_prose_holds_position_and_counts_failure(self):
        dec, d = self._decide(PROSE)
        self.assertEqual(dec.target_weight, 0.0)  # held at state.weight
        self.assertEqual(dec.reason, "unparseable")
        self.assertEqual(d.parse_failures, 1)

    def test_prose_does_not_silently_win(self):
        # The regression this counters: a model that never returns JSON looks
        # exactly like a market with no edge unless the failure is counted.
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(PROSE))):
            for _ in range(5):
                d.decide(Genome(), _probe_bars(), AgentState())
        self.assertEqual(d.parse_failures, 5)
        self.assertGreater(d.parse_failures / max(d.calls, 1), 0.5)

    def test_extracts_json_from_surrounding_prose(self):
        dec, _ = self._decide(f"Here you go: {GOOD} hope that helps")
        self.assertAlmostEqual(dec.target_weight, 0.42)

    def test_malformed_json_counted(self):
        _, d = self._decide(BAD_JSON)
        self.assertEqual(d.parse_failures, 1)

    def test_no_braces_counted(self):
        _, d = self._decide(NO_BRACES)
        self.assertEqual(d.parse_failures, 1)

    def test_weight_is_clamped(self):
        dec, _ = self._decide(json.dumps({"target_weight": 99}))
        self.assertEqual(dec.target_weight, 1.0)

    def test_negative_weight_clamped(self):
        dec, _ = self._decide(json.dumps({"target_weight": -99}))
        self.assertEqual(dec.target_weight, -1.0)

    def test_missing_field_defaults_flat(self):
        dec, d = self._decide(json.dumps({"reason": "no idea"}))
        self.assertEqual(dec.target_weight, 0.0)
        self.assertEqual(d.parse_failures, 0)

    def test_network_error_counted(self):
        d = make()
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
            dec = d.decide(Genome(), _probe_bars(), AgentState())
        self.assertEqual(d.network_errors, 1)
        self.assertIn("llm_error", dec.reason)

    def test_cost_accumulates(self):
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(GOOD))):
            d.decide(Genome(), _probe_bars(), AgentState())
            d.decide(Genome(), _probe_bars(), AgentState())
        self.assertEqual(d.calls, 2)
        self.assertGreater(d.total_cost, 0.0)

    def test_last_raw_retained(self):
        d = make()
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(GOOD))):
            d.decide(Genome(), _probe_bars(), AgentState())
        self.assertEqual(d.last_raw, GOOD)


class TestCallBudget(unittest.TestCase):
    def test_unlimited_by_default(self):
        b = CallBudget()
        self.assertTrue(all(b.take() for _ in range(1000)))

    def test_caps_at_limit(self):
        b = CallBudget(limit=3)
        self.assertEqual([b.take() for _ in range(5)], [True, True, True, False, False])
        self.assertEqual(b.spent, 3)
        self.assertEqual(b.blocked, 2)

    def test_snapshot_detects_truncation(self):
        b = CallBudget(limit=1)
        snap = b.snapshot()
        b.take()
        self.assertEqual(b.blocked_since(snap), 0)
        b.take()
        self.assertEqual(b.blocked_since(snap), 1)

    def test_budget_blocks_decide_without_network(self):
        b = CallBudget(limit=0)
        d = make(budget=b)
        with mock.patch("urllib.request.urlopen") as m:
            dec = d.decide(Genome(), _probe_bars(), AgentState())
        m.assert_not_called()
        self.assertEqual(dec.reason, "budget_exhausted")
        self.assertEqual(b.blocked, 1)


class TestTruncatedEpisodes(unittest.TestCase):
    def _ep(self, pnl, truncated=False):
        return Episode(
            pnl=pnl, equity_curve=[10_000.0, 10_000.0 + pnl], capital_deployed=10_000.0,
            inference_cost=0.0, n_evals=10, turnover=5.0, n_trades=5,
            truncated=truncated,
        )

    def test_truncated_excluded_from_fitness(self):
        f = evaluate([self._ep(500.0), self._ep(500.0), self._ep(-900.0, truncated=True)], 10_000.0)
        self.assertEqual(f.n, 2)                 # the truncated one is not counted
        self.assertEqual(f.n_truncated, 1)
        self.assertGreater(f.ret, 0.0)           # not dragged negative by the partial run

    def test_all_truncated_scores_zero(self):
        f = evaluate([self._ep(500.0, truncated=True)], 10_000.0)
        self.assertEqual(f.fitness, 0.0)
        self.assertEqual(f.n, 0)
        self.assertEqual(f.n_truncated, 1)

    def test_truncation_counted_reported(self):
        f = evaluate([self._ep(10.0), self._ep(10.0, truncated=True)], 10_000.0)
        self.assertEqual(f.n_truncated, 1)


class TestBuildDecisioner(unittest.TestCase):
    def test_offline_needs_no_key(self):
        import os
        for k in ("PROGENY_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY"):
            os.environ.pop(k, None)
        self.assertIsInstance(build_decider("offline/rules"), RulesDecisioner)
        for alias in ("rules", "none"):
            self.assertIsInstance(build_decider(alias), RulesDecisioner)

    def test_requires_key_for_llm(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                build_decider("big-pickle")


class TestBudgetThroughLoop(unittest.TestCase):
    """End-to-end: a tiny LLM cap must truncate, not silently half-run."""
    def test_cap_truncates_and_excludes(self):
        d = make()
        cfg = small_cfg(population=3, generations=2, episodes=2, decide_every=3,
                        max_calls_per_epoch=5)
        pages = {"n": 0}

        def fake(req, **kw):
            pages["n"] += 1
            return _resp(chat_payload(GOOD))

        with mock.patch("urllib.request.urlopen", fake):
            reports, _ = run(cfg, decisioner=d)

        self.assertLess(pages["n"], 3 * 2 * 2 * 20)   # cap held
        dropped = sum(r.best.fitness.n_truncated for r in reports)
        self.assertGreater(dropped, 0)                # and it was detected

    def test_no_cap_means_no_truncation(self):
        d = make()
        cfg = small_cfg(population=2, generations=1, episodes=1, decide_every=3)
        with mock.patch("urllib.request.urlopen", return_value=_resp(chat_payload(GOOD))):
            reports, _ = run(cfg, decisioner=d, market=market_for(cfg))
        self.assertEqual(sum(r.best.fitness.n_truncated for r in reports), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
