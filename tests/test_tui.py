"""Tests for the interface layer.

None of these need a terminal: the renderer is pure string layout and the
control flags are plain state. That separation is the point, and these tests
are what keep it.
"""

import random
import threading
import unittest

from money_agent import render, theme
from money_agent.render import TuiState, build_frame
from money_agent.tui import Control, _roster_from


def rows_to_text(frame):
    return ["".join(t for t, _ in row) for row in frame]


def assert_wellformed(case, frame, width):
    """Every frame row must be a list of (str, int) segments, each fitting width."""
    for y, row in enumerate(frame):
        case.assertIsInstance(row, list, f"row {y} is not a list")
        for seg in row:
            case.assertIsInstance(seg, tuple, f"row {y} segment {seg!r} is not a tuple")
            case.assertEqual(len(seg), 2, f"row {y} segment {seg!r} is not a 2-tuple")
            text, colour = seg
            case.assertIsInstance(text, str, f"row {y} text {text!r} not str")
            case.assertIsInstance(colour, int, f"row {y} colour {colour!r} not int")
            case.assertLessEqual(
                len(text), width,
                f"row {y} segment overflows width: {len(text)} > {width} ({text!r})",
            )


def populated(width=100, height=40):
    s = TuiState(total_generations=10, population=12, generation=4, model="big-pickle")
    s.mean_history = [-0.6, 1.2, 2.4, 3.1, 3.4]
    s.best_history = [0.4, 3.1, 5.2, 6.4, 7.0]
    s.roster = [
        {"name": "a8c6f5909f4c", "fitness": 4.18, "raw": 8.9, "ret": 0.109, "trades": 12},
        {"name": "b7a7fa1e6dba", "fitness": -0.5, "raw": -0.9, "ret": -0.02, "trades": 3},
    ]
    s.calls, s.cost = 1728, 0.31
    s.push("the chamber seals", theme.C_BLOOD)
    s.push("it learned your thresholds", theme.C_RITUAL)
    return s


class TestFrameShape(unittest.TestCase):
    def test_frame_rows_are_segments(self):
        f = build_frame(populated(), 100, 40)
        assert_wellformed(self, f, 100)

    def test_fits_requested_height(self):
        for h in (24, 30, 40, 55):
            with self.subTest(height=h):
                f = build_frame(populated(), 100, h)
                self.assertLessEqual(len(f), h + 2)

    def test_fits_narrow_terminal(self):
        for w in (46, 60, 80, 120):
            with self.subTest(width=w):
                assert_wellformed(self, build_frame(populated(), w, 40), w)

    def test_survives_absurd_size(self):
        # Should not raise, whatever the terminal claims.
        for w, h in ((20, 8), (200, 100), (40, 5)):
            with self.subTest(size=(w, h)):
                build_frame(populated(), w, h)

    def test_empty_state_renders(self):
        assert_wellformed(self, build_frame(TuiState(), 100, 40), 100)

    def test_empty_state_no_crash(self):
        # Regression: the first frame is drawn before generation 0 finishes, so
        # the histories are empty and any direct [-1] raised IndexError.
        text = rows_to_text(build_frame(TuiState(), 100, 34))
        self.assertIn("nothing has been summoned yet", "\n".join(text))


class TestFrameContent(unittest.TestCase):
    def test_shows_sigil_and_title(self):
        text = "\n".join(rows_to_text(build_frame(populated(), 100, 40)))
        self.assertIn("M O N E Y   A G E N T", text)
        self.assertIn("RITUAL DIRECTORY", text)
        self.assertIn("VITALS", text)
        self.assertIn("WHISPERS", text)

    def test_shows_roster_entries(self):
        text = "\n".join(rows_to_text(build_frame(populated(), 100, 40)))
        self.assertIn("a8c6f5909f4c", text)
        self.assertIn("b7a7fa1e6dba", text)

    def test_shows_key_hints(self):
        text = "\n".join(rows_to_text(build_frame(populated(), 100, 40)))
        self.assertIn("q quit", text)
        self.assertIn("space pause", text)

    def test_losing_genome_is_not_green(self):
        s = populated()
        f = build_frame(s, 100, 40)
        found = []
        for row in f:
            # Only the roster column, left of the │ divider. The right panel has
            # its own green (the trace sparkline) and would mask the result.
            roster_col = []
            for text, colour in row:
                if text.strip() == "│":
                    break
                roster_col.append((text, colour))
            line = "".join(t for t, _ in roster_col)
            if "b7a7fa1e6dba" in line:
                found = [c for _, c in roster_col if _]
        self.assertNotIn(theme.C_GOOD, found)
        self.assertTrue(found)

    def test_warnings_surface_when_set(self):
        s = populated()
        s.parse_failures = 412
        s.quarantined = 3
        s.truncated = 2
        text = "\n".join(rows_to_text(build_frame(s, 100, 40)))
        self.assertIn("CORRUPT", text)
        self.assertIn("sealed 3", text)
        self.assertIn("cut short 2", text)

    def test_parse_failures_not_shown_when_zero(self):
        s = populated()
        s.parse_failures = 0
        self.assertNotIn("CORRUPT", "\n".join(rows_to_text(build_frame(s, 100, 40))))

    def test_pause_state_shown(self):
        s = populated()
        s.paused = True
        self.assertIn("PAUSED", "\n".join(rows_to_text(build_frame(s, 100, 40))))

    def test_model_and_speed_shown(self):
        s = populated()
        s.speed = 4.0
        text = "\n".join(rows_to_text(build_frame(s, 100, 40)))
        self.assertIn("big-pickle", text)
        self.assertIn("x4", text)


class TestVerdict(unittest.TestCase):
    def test_awakening_before_any_generation(self):
        self.assertEqual(TuiState().verdict, "AWAKENING")

    def test_rising(self):
        s = TuiState(mean_history=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        self.assertEqual(s.verdict, "RISING")

    def test_stagnant(self):
        s = TuiState(mean_history=[5.0, 4.9, 5.1, 4.8, 5.0, 4.9])
        self.assertEqual(s.verdict, "STAGNANT")


class TestStateSafety(unittest.TestCase):
    def test_accessors_empty(self):
        s = TuiState()
        self.assertEqual(s.mean_now, 0.0)
        self.assertEqual(s.best_now, 0.0)

    def test_log_bounded(self):
        s = TuiState()
        for i in range(500):
            s.push(f"line {i}")
        self.assertLessEqual(len(s.log), 200)

    def test_push_keeps_newest(self):
        s = TuiState()
        s.push("oldest")
        for i in range(300):
            s.push(str(i))
        self.assertNotEqual(s.log[0].text, "oldest")


class TestControl(unittest.TestCase):
    def test_runs_unpaused(self):
        self.assertTrue(Control().wait_if_paused())

    def test_blocks_while_paused(self):
        # wait_if_paused blocks by design, so prove it on a thread rather than
        # calling it inline and hanging the suite.
        ctl = Control()
        ctl.paused = True
        done = []

        def worker():
            done.append(ctl.wait_if_paused())

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=0.3)
        self.assertTrue(t.is_alive(), "should still be blocked while paused")
        self.assertEqual(done, [])

        ctl.paused = False
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(done, [True])

    def test_quit_releases(self):
        ctl = Control()
        ctl.paused = True
        ctl.quit = True
        self.assertFalse(ctl.wait_if_paused())

    def test_skip_releases_pause(self):
        ctl = Control()
        ctl.paused = True
        ctl.skip_to_end = True
        self.assertTrue(ctl.wait_if_paused())


class TestRoster(unittest.TestCase):
    class FakeRec:
        def __init__(self, gid, fit):
            self.genome_id = gid
            class F:
                fitness = fit
                ret = 0.01
                raw = fit
                trades = 3
            self.fitness = F()

    def test_sorted_by_fitness(self):
        class R:
            generation = 0
            records = [TestRoster.FakeRec("low", -2.0), TestRoster.FakeRec("high", 5.0)]
            best = TestRoster.FakeRec("high", 5.0)
            mean_fitness = 1.0
            quarantined = []
            new_genomes = 0
            best_diff = ""
        out = _roster_from(R())
        self.assertEqual(out[0]["name"], "high")
        self.assertEqual(out[1]["name"], "low")

    def test_handles_empty(self):
        class R:
            records = []
        self.assertEqual(_roster_from(R()), [])


class TestTheme(unittest.TestCase):
    def test_bar_bounds(self):
        self.assertEqual(theme.bar(-5.0, 10, -1, 1), "·" * 10)
        self.assertEqual(theme.bar(5.0, 10, -1, 1), "█" * 10)
        self.assertEqual(len(theme.bar(0.0, 10, -1, 1)), 10)

    def test_bar_zero_width(self):
        self.assertEqual(theme.bar(0.5, 0), "")

    def test_sparkline_flat(self):
        self.assertEqual(len(theme.sparkline([1.0] * 5, 10)), 5)

    def test_sparkline_empty(self):
        self.assertEqual(theme.sparkline([], 10).strip(), "")

    def test_sparkline_respects_width(self):
        self.assertLessEqual(len(theme.sparkline(list(range(50)), 10)), 10)

    def test_corrupt_is_noop_at_zero(self):
        self.assertEqual(theme.corrupt("hello", 0.0), "hello")

    def test_corrupt_keeps_spaces(self):
        random.seed(0)
        out = theme.corrupt("a b c", 1.0)
        self.assertIn(" ", out)
        self.assertEqual(len(out), 5)

    def test_center_and_pad(self):
        self.assertEqual(theme.center("ab", 6), "  ab  ")
        self.assertEqual(theme.pad("ab", 4), "ab  ")
        self.assertEqual(theme.pad("abcdef", 3), "abc")

    def test_flicker_rate_limited(self):
        f = theme.Flicker(chance=1.0, min_gap=99.0)
        first = f.step()
        self.assertEqual(f.step(), first)  # gap not elapsed, so unchanged

    def test_art_fits_short_terminal(self):
        self.assertLessEqual(len(theme.art_lines(24)), 4)
        self.assertGreater(len(theme.art_lines(60)), 4)

    def test_ritual_lines_exist(self):
        rng = random.Random(1)
        self.assertTrue(all(isinstance(theme.ritual_line(rng, i), str) for i in range(20)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
