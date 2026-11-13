"""Tests for broadcast_ball.py, on synthetic inputs only: no match data needed.

    python -m unittest discover -s solution
"""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import broadcast_ball as bb  # noqa: E402
import hmm_possession as hp  # noqa: E402

PITCH = (100.0, 50.0)
FPS = 25


def wandering_ball(n_frames, seed=0):
    """A smooth random ball track inside the pitch, normalised."""
    steps = np.random.default_rng(seed).normal(0.0, 0.01, (n_frames, 2))
    return 0.5 + 0.4 * np.tanh(np.cumsum(steps, axis=0))


def keep_everything(scene_error_m, n_compared, run_s):
    return np.ones(len(scene_error_m), dtype=bool)


class Parsing(unittest.TestCase):
    def test_tracks_land_on_the_right_frame_and_slot(self):
        lines = ["1:0.1,0.2,1;NaN,NaN,NaN:0.5,0.5,0\n",
                 "3:NaN,NaN,NaN;0.3,0.4,1:NaN,NaN,NaN\n"]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "atd.txt")
            with open(path, "w") as handle:
                handle.writelines(lines)
            ball, players = bb.read_tracks(path, 3, 2)

        self.assertTrue(np.allclose(ball[0], [0.5, 0.5]))
        self.assertTrue(np.isnan(ball[1:]).all())
        self.assertTrue(np.allclose(players[0, 0], [0.1, 0.2]))
        self.assertTrue(np.allclose(players[2, 1], [0.3, 0.4]))
        self.assertTrue(np.isnan(players[1]).all())

    def test_valid_column_is_read_as_booleans(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "homography.csv")
            with open(path, "w") as handle:
                handle.write("frame,h00,reprojection_error,valid\n0,1.5,0.2,1\n1,NaN,NaN,0\n")
            self.assertEqual(bb.read_valid(path).tolist(), [True, False])


class Offset(unittest.TestCase):
    def test_recovers_a_known_shift_despite_gaps_and_noise(self):
        reference = wandering_ball(4000)
        other = reference[700:3200] + np.random.default_rng(1).normal(0.0, 0.003, (2500, 2))
        other[400:900] = np.nan      # a replay: the other recording sees nothing
        reference = reference.copy()
        reference[1500:1800] = np.nan  # and the reference loses the ball elsewhere

        offset, agreement = bb.estimate_offset(reference, other, PITCH)
        self.assertEqual(offset, 700)
        self.assertGreater(agreement, 0.9)

    def test_a_negative_shift(self):
        reference = wandering_ball(3000)
        other = np.concatenate((wandering_ball(500, seed=5), reference))
        self.assertEqual(bb.estimate_offset(reference, other, PITCH)[0], -500)

    def test_unrelated_tracks_are_reported_as_disagreeing(self):
        _, agreement = bb.estimate_offset(wandering_ball(3000, 2), wandering_ball(3000, 3), PITCH)
        self.assertLess(agreement, bb.MIN_AGREEMENT_SHARE)

    def test_paired_rows_stay_inside_the_other_recording(self):
        rows, other_rows = bb.paired_rows(10, 8, 14, 3)
        self.assertEqual(rows.tolist(), [10, 11, 12])
        self.assertEqual(other_rows.tolist(), [0, 1, 2])


class SceneError(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.players = rng.uniform(0.1, 0.9, (5, 8, 2))
        self.team = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_the_same_scene_in_another_slot_order_has_no_error(self):
        order = np.array([3, 1, 0, 2, 6, 7, 5, 4])
        error, compared = bb.scene_error(self.players, self.team, self.players[:, order],
                                         self.team[order], PITCH)
        self.assertTrue(np.allclose(error, 0.0))
        self.assertTrue((compared == 8).all())

    def test_a_different_moment_is_metres_away(self):
        other = np.random.default_rng(9).uniform(0.1, 0.9, (5, 8, 2))
        error, _ = bb.scene_error(self.players, self.team, other, self.team, PITCH)
        self.assertTrue((error > 2.0).all())

    def test_players_are_only_compared_with_their_own_team(self):
        swapped = 1 - self.team
        error, _ = bb.scene_error(self.players, self.team, self.players, swapped, PITCH)
        self.assertTrue((error > 2.0).all())

    def test_too_few_players_gives_no_answer(self):
        other = np.full((5, 8, 2), np.nan)
        other[:, :3] = self.players[:, :3]
        error, compared = bb.scene_error(self.players, self.team, other, self.team, PITCH)
        self.assertTrue(np.isnan(error).all())
        self.assertTrue((compared == 3).all())


class Policy(unittest.TestCase):
    def accept(self, scene_error_m, run_s, n_compared=16):
        return bool(bb.accept_broadcast_ball(np.array([scene_error_m]), np.array([n_compared]),
                                             np.array([run_s]))[0])

    def test_the_same_moment_held_for_a_while_is_kept(self):
        self.assertTrue(self.accept(0.3, 4.0))

    def test_a_replay_is_rejected_however_long_it_lasts(self):
        self.assertFalse(self.accept(9.0, 30.0))

    def test_a_brief_sighting_is_rejected_even_on_a_matching_scene(self):
        self.assertFalse(self.accept(0.3, 0.2))

    def test_a_scene_that_could_not_be_compared_is_rejected(self):
        self.assertFalse(self.accept(np.nan, 4.0, n_compared=2))


class Fill(unittest.TestCase):
    def setUp(self):
        n_frames, shift = 3000, 400
        rng = np.random.default_rng(7)
        self.truth = wandering_ball(n_frames)
        self.players = rng.uniform(0.1, 0.9, (n_frames, 8, 2))
        self.team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.ball = self.truth.copy()
        self.ball[1000:1200] = np.nan
        self.other_ball = self.truth[shift:].copy()
        self.other_players = self.players[shift:].copy()
        self.valid = np.ones(n_frames - shift, dtype=bool)
        self.blocks = [(500, 2500)]

    def fill(self, accept=keep_everything):
        return bb.fill_from_broadcast(self.ball, self.players, self.team, self.other_ball,
                                      self.other_players, self.team, self.valid, self.blocks,
                                      PITCH, FPS, accept)

    def test_a_gap_is_filled_with_the_other_recording_at_the_right_frame(self):
        fill, report = self.fill()
        self.assertTrue(np.allclose(fill[1000:1200], self.truth[1000:1200]))
        self.assertEqual(report[0]["offset"], 400)
        self.assertEqual(report[0]["kept"], 200)

    def test_frames_that_have_a_ball_are_never_overridden(self):
        fill, _ = self.fill()
        self.assertTrue(np.isnan(fill[~np.isnan(self.ball[:, 0])]).all())

    def test_frames_without_a_pitch_transform_are_not_used(self):
        self.valid[1050 - 400:1100 - 400] = False
        fill, report = self.fill()
        self.assertTrue(np.isnan(fill[1050:1100]).all())
        self.assertEqual(report[0]["kept"], 150)

    def test_the_policy_sees_scene_error_and_run_length_and_decides(self):
        self.other_players[1100 - 400:1200 - 400] = np.random.default_rng(8).uniform(
            0.1, 0.9, (100, 8, 2))  # the other recording cuts to a replay
        seen = {}

        def policy(scene_error_m, n_compared, run_s):
            seen["run_s"] = run_s
            return scene_error_m < 1.0

        fill, _ = self.fill(policy)
        self.assertTrue(np.allclose(seen["run_s"], 200 / FPS))
        self.assertFalse(np.isnan(fill[1000:1100]).any())
        self.assertTrue(np.isnan(fill[1100:1200]).all())

    def test_tracks_that_cannot_be_aligned_raise(self):
        self.other_ball = wandering_ball(len(self.other_ball), seed=11)
        with self.assertRaises(ValueError):
            self.fill()


class FilledDistances(unittest.TestCase):
    def test_a_filled_ball_is_measured_like_a_seen_one(self):
        lines = ["1:0.5,0.5,1;0.6,0.5,1:NaN,NaN,NaN\n",
                 "2:0.5,0.5,1;0.6,0.5,1:NaN,NaN,NaN\n",
                 "3:0.5,0.5,1;0.6,0.5,1:0.6,0.5,0\n"]
        fill = np.array([[0.5, 0.5], [np.nan, np.nan], [0.0, 0.0]])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "atd.txt")
            with open(path, "w") as handle:
                handle.writelines(lines)
            distances = hp.nearest_by_team(path, np.array([0, 1]), 3, PITCH, fill)

        self.assertTrue(np.allclose(distances[0], [0.0, 10.0]))  # filled
        self.assertTrue(np.isnan(distances[1]).all())            # nothing from either source
        self.assertTrue(np.allclose(distances[2], [10.0, 0.0]))  # its own ball wins over the fill


if __name__ == '__main__':
    unittest.main()
