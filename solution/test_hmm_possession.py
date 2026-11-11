"""Tests on synthetic data only, so they never touch the licensed match data.

    python solution/test_hmm_possession.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import hmm_possession as hp  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]


def no_opinion(gap_seconds):
    """A missing ball says nothing. Stands in for the real evidence function."""
    return np.zeros((len(gap_seconds), 3))


def evidence_for(states, strength=1.0):
    """Log evidence favouring the given state on each frame."""
    out = np.zeros((len(states), 3))
    out[np.arange(len(states)), states] = strength
    return out


class GapLengths(unittest.TestCase):
    def test_each_frame_gets_the_length_of_its_own_run(self):
        missing = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
        self.assertEqual(hp.gap_lengths(missing).tolist(), [0, 2, 2, 0, 3, 3, 3])


class MissingBall(unittest.TestCase):
    @staticmethod
    def decode_with_gap(gap_seconds):
        """Decode home possession with one missing-ball gap in the middle."""
        distances = np.tile([2.0, 20.0], (2000, 1))
        gap_frames = int(gap_seconds * FPS)
        distances[1000:1000 + gap_frames] = np.nan
        return hp.viterbi(hp.log_emissions(distances, FPS), hp.log_transitions(FPS))

    def test_evidence_never_favours_a_team(self):
        evidence = hp.missing_ball_log_evidence(np.array([0.5, 3.0, 30.0]))
        self.assertEqual(evidence.shape, (3, 3))
        self.assertTrue(np.array_equal(evidence[:, hp.H], evidence[:, hp.A]))

    def test_evidence_for_a_stoppage_grows_then_stops_growing(self):
        dead = hp.missing_ball_log_evidence(np.array([1.0, 4.0, 8.0, 60.0, 600.0]))[:, hp.D]
        self.assertEqual(dead[0], 0.0)
        self.assertTrue(dead[0] < dead[1] < dead[2] < dead[3])
        self.assertEqual(dead[3], dead[4])

    def test_short_occlusion_does_not_split_a_possession(self):
        self.assertTrue(np.all(self.decode_with_gap(3) == hp.H))

    def test_long_absence_becomes_a_stoppage(self):
        path = self.decode_with_gap(15)
        self.assertTrue(np.all(path[1050:1300] == hp.D))
        self.assertEqual(path[0], hp.H)
        self.assertEqual(path[-1], hp.H)


def set_piece(start, stop, team="N/A", code="SET PIECES"):
    return {"code": code, "start_frame": start, "end_frame": stop, "tags": {"Team": team}}


class SmartTagging(unittest.TestCase):
    def test_home_is_matched_by_name_and_away_by_elimination(self):
        codes = hp.tag_team_codes(["Barcelona", "Atletic Club", "N/A", None, " Barcelona "],
                                  "FC Barcelona")
        self.assertEqual(codes, {"Barcelona": hp.H, "Atletic Club": hp.A})

    def test_ambiguous_names_raise_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            hp.tag_team_codes(["Barcelona", "Atletic Club"], "Real Madrid CF")
        with self.assertRaises(ValueError):
            hp.tag_team_codes(["Barcelona"], "FC Barcelona")

    def test_window_leans_dead_then_towards_the_tagged_team(self):
        evidence = hp.restart_log_evidence([set_piece(10, 20, "Barcelona")],
                                           {"Barcelona": hp.H}, 30)
        self.assertTrue(np.all(evidence[10:15, hp.D] > 0))
        self.assertTrue(np.all(evidence[15:20, hp.H] > 0))
        self.assertTrue(np.all(evidence[:10] == 0) and np.all(evidence[20:] == 0))
        self.assertTrue(np.all(evidence[:, hp.A] == 0))

    def test_untagged_side_and_other_codes(self):
        evidence = hp.restart_log_evidence(
            [set_piece(0, 10), set_piece(10, 20, "Barcelona", code="BUILD UP")],
            {"Barcelona": hp.H}, 20)
        self.assertTrue(np.all(evidence[0:5, hp.D] > 0))
        self.assertTrue(np.all(evidence[5:, :] == 0))

    def test_windows_are_clipped_to_the_video(self):
        evidence = hp.restart_log_evidence([set_piece(-5, 40, "Barcelona")],
                                           {"Barcelona": hp.H}, 30)
        self.assertEqual(evidence.shape, (30, 3))

    def test_restart_ends_an_ambiguous_stretch_with_the_right_team(self):
        # The ball is missing for 20 s, then a home restart: the decoder should pass through D
        # and come out with H, even though tracking says nothing either way.
        distances = np.full((1500, 2), np.nan)
        distances[1000:] = [10.0, 10.0]
        extra = hp.restart_log_evidence([set_piece(940, 1060, "Barcelona")],
                                        {"Barcelona": hp.H}, 1500)
        log_emit = hp.log_emissions(distances, FPS) + extra
        path = hp.viterbi(log_emit, hp.log_transitions(FPS))
        self.assertEqual(path[900], hp.D)
        self.assertEqual(path[-1], hp.H)


class Viterbi(unittest.TestCase):
    def test_one_frame_of_noise_does_not_end_a_possession(self):
        states = np.full(200, hp.H, dtype=np.int8)
        states[100] = hp.A
        path = hp.viterbi(evidence_for(states), hp.log_transitions(FPS))
        self.assertTrue(np.all(path == hp.H))

    def test_sustained_evidence_changes_possession_near_the_true_frame(self):
        states = np.concatenate((np.full(300, hp.H), np.full(300, hp.A))).astype(np.int8)
        path = hp.viterbi(evidence_for(states), hp.log_transitions(FPS))
        self.assertEqual(path[0], hp.H)
        self.assertEqual(path[-1], hp.A)
        self.assertLessEqual(abs(int(np.flatnonzero(path == hp.A)[0]) - 300), 1)

    def test_empty_input(self):
        self.assertEqual(len(hp.viterbi(np.zeros((0, 3)), hp.log_transitions(FPS))), 0)


class Transitions(unittest.TestCase):
    def test_rows_are_probability_distributions(self):
        self.assertTrue(np.allclose(np.exp(hp.log_transitions(FPS)).sum(axis=1), 1.0))


class Parsing(unittest.TestCase):
    def test_distances_land_on_the_right_frame_in_metres(self):
        lines = ["1:0.5,0.5,1;0.6,0.5,1;NaN,NaN,NaN:0.5,0.5,0\n",
                 "2:0.5,0.5,1;0.6,0.5,1;NaN,NaN,NaN:NaN,NaN,NaN\n",
                 "3:NaN,NaN,NaN;0.6,0.5,1;NaN,NaN,NaN:0.6,0.5,0\n"]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "atd.txt")
            with open(path, "w") as handle:
                handle.writelines(lines)
            distances = hp.nearest_by_team(path, np.array([0, 1, 1]), 3, (100.0, 50.0))

        self.assertTrue(np.allclose(distances[0], [0.0, 10.0]))
        self.assertTrue(np.all(np.isnan(distances[1])))
        self.assertEqual(distances[2, hp.H], np.inf)
        self.assertAlmostEqual(distances[2, hp.A], 0.0)


class Predict(unittest.TestCase):
    def test_output_is_a_valid_submission_with_x_outside_the_halves(self):
        n_frames = GAMES[GAME]["n_video_frames"]
        distances = np.tile([2.0, 20.0], (n_frames, 1))
        labels = hp.predict(distances, missing_evidence=no_opinion)

        table = {"frame": np.arange(n_frames), "label": labels}
        import pandas as pd
        self.assertIsNone(pm.validate_submission(pd.DataFrame(table), n_frames))

        (_, first, last), _ = half_video_blocks(GAME)
        self.assertEqual(labels[first - 1], "X")
        self.assertEqual(labels[first], "H")
        self.assertEqual(labels[last], "H")
        self.assertEqual(labels[last + 1], "X")


if __name__ == '__main__':
    unittest.main()
