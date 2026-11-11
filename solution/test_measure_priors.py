"""Tests for the prior measurement, on hand-written events only.

    python solution/test_measure_priors.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_priors as mp  # noqa: E402


def event(team, kind, start, end=None, period=1):
    return {"team": team, "type": kind, "period": period, "start": float(start),
            "end": float(start if end is None else end)}


# Home kicks off, passes, loses it; away recover, pass and put it out at 30 s. Home throw-in at
# 40 s, one pass, period ends at 50 s.
GAME = [
    event("Home", "SET PIECE", 0),
    event("Home", "PASS", 0, 5),
    event("Home", "BALL LOST", 10, 12),
    event("Away", "RECOVERY", 12),
    event("Away", "PASS", 13, 20),
    event("Away", "BALL OUT", 25, 30),
    event("Home", "CARD", 33),
    event("Home", "SET PIECE", 40),
    event("Home", "PASS", 40, 50),
]


class Stretches(unittest.TestCase):
    def test_ball_is_dead_from_the_last_event_to_the_restart(self):
        stretches, stoppages = mp.alive_stretches(GAME)
        self.assertEqual([(s["start"], s["stop"], s["ended_by"]) for s in stretches],
                         [(0.0, 30.0, "stoppage"), (40.0, 50.0, "period")])
        self.assertEqual(stoppages, [10.0])

    def test_periods_are_separate(self):
        second_half = [event("Away", "SET PIECE", 100, period=2),
                       event("Away", "PASS", 100, 110, period=2)]
        stretches, stoppages = mp.alive_stretches(GAME + second_half)
        self.assertEqual(len(stretches), 3)
        self.assertEqual(stoppages, [10.0])  # halftime is not a stoppage


class Possessions(unittest.TestCase):
    def test_a_controlled_recovery_changes_possession(self):
        stretch = mp.alive_stretches(GAME)[0][0]
        self.assertEqual(mp.possessions(stretch),
                         [("Home", 0.0, 12.0), ("Away", 12.0, 30.0)])

    def test_an_uncontrolled_touch_does_not(self):
        events = [event("Home", "SET PIECE", 0),
                  event("Home", "PASS", 0, 5),
                  event("Away", "RECOVERY", 6),
                  event("Away", "BALL LOST", 6.5, 7),
                  event("Home", "RECOVERY", 7),
                  event("Home", "PASS", 8, 20)]
        stretch = mp.alive_stretches(events)[0][0]
        self.assertEqual(mp.possessions(stretch), [("Home", 0.0, 20.0)])
        self.assertEqual(len(mp.possessions(stretch, controlled_only=False)), 3)

    def test_keeping_the_ball_long_enough_counts_as_control(self):
        events = [event("Home", "SET PIECE", 0),
                  event("Away", "RECOVERY", 5),
                  event("Away", "BALL LOST", 9, 10)]
        stretch = mp.alive_stretches(events)[0][0]
        self.assertEqual([team for team, _, _ in mp.possessions(stretch)], ["Home", "Away"])

    def test_challenges_never_change_possession(self):
        events = [event("Home", "SET PIECE", 0),
                  event("Away", "CHALLENGE", 3),
                  event("Home", "PASS", 4, 10)]
        stretch = mp.alive_stretches(events)[0][0]
        self.assertEqual(mp.possessions(stretch), [("Home", 0.0, 10.0)])


class Priors(unittest.TestCase):
    def test_numbers_for_the_toy_game(self):
        possession, stoppage, share = mp.priors(mp.measure(GAME))
        self.assertAlmostEqual(possession, (12 + 18 + 10) / 3.0)
        self.assertAlmostEqual(stoppage, 10.0)
        # One turnover and one stoppage; the possession cut off by the whistle counts as neither.
        self.assertAlmostEqual(share, 0.5)


if __name__ == '__main__':
    unittest.main()
