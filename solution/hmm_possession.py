"""Possession as a sticky three-state model, decoded with Viterbi.

    python solution/hmm_possession.py
    python evaluation/validate_submission.py evaluation/results/solution_hmm.csv

The reference baseline decides every frame and then smooths. This decides the whole half at once:
each frame gives evidence for H, A and D, and every change of state pays a transition cost, so a
possession only ends when the evidence for ending it outweighs that cost. spellF1 scores whole
possessions, which is exactly what the transition cost protects.

numpy only, and deterministic: the same data bundle always produces the same file, as RULES.md
requires.
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import (FIELD_SEP, GROUP_SEP, MISSING, RESULTS_DIR,  # noqa: E402
                      team_names, team_of_each_track, write_submission)
import load_smart_tagging  # noqa: E402
import broadcast_ball  # noqa: E402

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Model priors, measured on other games and never on the challenge match (RULES.md): 938
# possessions and 241 stoppages pooled over the three Metrica sample-data games. Regenerate with
# `python solution/measure_priors.py --fetch`, which prints exactly these three numbers.
MEAN_POSSESSION_S = 12.7
MEAN_STOPPAGE_S = 21.1
TURNOVER_SHARE = 0.74  # share of possessions that end with the other team, not a stoppage

# Distance only says who has the ball while somebody can actually play it. Outside
# CONTROL_RADIUS_M of every player the ball is in flight or loose, neither team gets evidence, and
# possession stays with the last team -- the rule CHALLENGE.md gives for loose and contested
# balls. Without this a long pass travelling past opponents hands them the ball on the way.
#
# Both values chosen on other games with solution/tune_on_sample_games.py: every sample game
# scores better with the radius than without, and 2 m with an 8 m scale sits in the middle of a
# stable region (1.5 to 3 m) and holds up when noise is added to the distances.
CONTROL_RADIUS_M = 2.0  # None switches the radius off
TIE_BREAK = 1e-6  # far below any real evidence, far above floating-point rounding
DISTANCE_SCALE_M = 8.0  # metres of distance advantage worth one unit of log evidence
MAX_DISTANCE_M = 30.0  # beyond this a player is simply "far", however far
VISIBLE_BALL_DEAD_COST = 0.5  # a visible ball is mild evidence that play is alive

# How a missing ball counts towards a stoppage: nothing up to GAP_NEUTRAL_S, then a linear ramp to
# GAP_MAX_DEAD_EVIDENCE per frame at GAP_FULL_S. With the priors above, a gap has to reach about
# 5 s before it can split a possession. Judgement calls, subject to the same rule as the priors.
GAP_NEUTRAL_S = 2.0
GAP_FULL_S = 10.0
GAP_MAX_DEAD_EVIDENCE = 0.3

# Smart Tagging set pieces. Each tagged window contains a restart somewhere inside it, so its first
# half leans towards D and its second half towards the tagged team. Per frame, and deliberately
# gentle: over a ~5 s window this is about 3 units, a nudge that leaves the exact frame to the
# tracking rather than an override. The tagging is automatic and has its own errors (DATA.md).
SET_PIECE_CODE = "set pieces"
RESTART_DEAD_EVIDENCE = 0.05
RESTART_TEAM_EVIDENCE = 0.05
NO_TEAM = "n/a"
# Words that say "football club" rather than which club, ignored when matching names across files.
GENERIC_NAME_WORDS = {"fc", "cf", "club", "de", "sc", "ac", "cd", "ud", "rcd", "the"}


def nearest_by_team(raw_path, team_of_track, n_frames, pitch, ball_fill=None):
    """(n_frames, 2) metres from the ball to the nearest home and away track.

    NaN on both columns where the ball is not visible, inf where a team has no visible track.
    ball_fill, if given, is (n_frames, 2) normalised ball positions from another source, used
    only on frames where this file has no ball; NaN there means no position from it either.
    """
    length, width = pitch
    out = np.full((n_frames, 2), np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            if parts[2][0] != MISSING:
                ball_x, ball_y = [float(v) for v in parts[2].split(",")[:2]]
            elif ball_fill is not None and not np.isnan(ball_fill[int(parts[0]) - 1, 0]):
                ball_x, ball_y = ball_fill[int(parts[0]) - 1]
            else:
                continue
            nearest = [np.inf, np.inf]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = [float(v) for v in group.split(",")[:2]]
                distance = np.hypot((x - ball_x) * length, (y - ball_y) * width)
                team = team_of_track[slot]
                nearest[team] = min(nearest[team], distance)

            out[int(parts[0]) - 1] = nearest  # frame_id counts from 1, submissions from 0
    return out


def gap_lengths(missing):
    """For each True in a boolean array, the length of the run of True containing it. 0 elsewhere."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    lengths = np.zeros(len(missing), dtype=np.int64)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lengths[start:stop] = stop - start
    return lengths


def missing_ball_log_evidence(gap_seconds):
    """Log evidence for (H, A, D) on frames where the ball is not visible.

    gap_seconds is a 1-D array holding, for each such frame, the full length in seconds of the
    missing-ball gap it belongs to. Return an array of shape (len(gap_seconds), 3).
    """
    # H and A stay at zero: a missing ball favours neither team.
    ramp = np.clip((gap_seconds - GAP_NEUTRAL_S) / (GAP_FULL_S - GAP_NEUTRAL_S), 0.0, 1.0)
    out = np.zeros((len(gap_seconds), 3))
    out[:, D] = ramp * GAP_MAX_DEAD_EVIDENCE
    return out


def log_emissions(distances, fps, missing_evidence=missing_ball_log_evidence):
    """(n, 3) log evidence for H, A and D on every frame."""
    seen = ~np.isnan(distances[:, 0])
    out = np.empty((len(distances), 3))

    capped = np.minimum(distances[seen], MAX_DISTANCE_M)
    team = -capped / DISTANCE_SCALE_M
    if CONTROL_RADIUS_M is not None:
        team[capped.min(axis=1) > CONTROL_RADIUS_M] = 0.0  # nobody can play it: no opinion
    out[seen, H] = team[:, H]
    out[seen, A] = team[:, A]
    out[seen, D] = team.max(axis=1) - VISIBLE_BALL_DEAD_COST

    missing = ~seen
    out[missing] = missing_evidence(gap_lengths(missing)[missing] / float(fps))
    return out


def _name_words(name):
    return set(re.findall(r"[a-z]+", (name or "").lower())) - GENERIC_NAME_WORDS


def tag_team_codes(tag_teams, home_name):
    """{Smart Tagging team name: H or A}, anchored on the home club name from the ATD metadata.

    The two files spell clubs differently, so only the home side is matched by name and the away
    side is whichever other team remains. Anything ambiguous raises rather than guessing a side.
    """
    stripped = {(t or "").strip() for t in tag_teams}
    teams = sorted(t for t in stripped if t and t.lower() != NO_TEAM)
    home = [t for t in teams if _name_words(t) & _name_words(home_name)]
    if len(teams) != 2 or len(home) != 1:
        raise ValueError("cannot map Smart Tagging teams {} to home {!r}".format(teams, home_name))
    away = [t for t in teams if t != home[0]][0]
    return {home[0]: H, away: A}


def restart_log_evidence(events, team_codes, n_frames):
    """(n_frames, 3) extra log evidence from Smart Tagging set pieces. Zero everywhere else."""
    out = np.zeros((n_frames, 3))
    for event in events:
        if (event["code"] or "").strip().lower() != SET_PIECE_CODE:
            continue
        start, stop = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if stop <= start:
            continue
        middle = (start + stop) // 2
        out[start:middle, D] += RESTART_DEAD_EVIDENCE
        team = team_codes.get((event["tags"].get("Team") or "").strip())
        if team is not None:  # throw-ins are tagged N/A: dead ball, but no side
            out[middle:stop, team] += RESTART_TEAM_EVIDENCE
    return out


def log_transitions(fps):
    """3x3 log transition matrix from the mean durations above. Rows are from, columns are to."""
    leave_possession = 1.0 / (MEAN_POSSESSION_S * fps)
    leave_stoppage = 1.0 / (MEAN_STOPPAGE_S * fps)
    turnover = leave_possession * TURNOVER_SHARE
    stoppage = leave_possession - turnover
    return np.log([
        [1.0 - leave_possession, turnover, stoppage],
        [turnover, 1.0 - leave_possession, stoppage],
        [leave_stoppage / 2.0, leave_stoppage / 2.0, 1.0 - leave_stoppage],
    ])


def viterbi(log_emit, log_trans):
    """Most likely state sequence, deterministic.

    Through a stretch with no evidence -- the ball in flight, outside every control radius --
    every frame is an equally good place for a change of state, and rounding error would pick
    one. TIE_BREAK settles it: the change lands on the last such frame, which is when the next
    player receives the ball.
    """
    n_frames = len(log_emit)
    path = np.empty(n_frames, dtype=np.int8)
    if not n_frames:
        return path

    back = np.zeros((n_frames, 3), dtype=np.int8)
    score = log_emit[0] - np.log(3.0)
    prefer_change = TIE_BREAK * (1.0 - np.eye(3))
    for frame in range(1, n_frames):
        candidates = score[:, np.newaxis] + log_trans
        back[frame] = (candidates + prefer_change).argmax(axis=0)
        score = candidates.max(axis=0) + log_emit[frame]

    path[-1] = score.argmax()
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def predict(distances, game=GAME, missing_evidence=missing_ball_log_evidence, extra_evidence=None,
            blocks=None):
    """Labels for every video frame: decoded inside each half, X everywhere else.

    extra_evidence, if given, is (n_frames, 3) log evidence added to the tracking's own.
    blocks, if given, is [(first_frame, last_frame)] to decode instead of this game's halves --
    which is how the same code runs on other matches.
    """
    fps = GAMES[game]["fps"]
    if blocks is None:
        blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    log_emit = log_emissions(distances, fps, missing_evidence)
    if extra_evidence is not None:
        log_emit = log_emit + extra_evidence
    log_trans = log_transitions(fps)

    labels = np.full(len(distances), "X", dtype="<U1")
    for first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(log_emit[first:last + 1], log_trans)]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "solution_hmm.csv"),
                        help="where to write the submission")
    parser.add_argument("--no-tags", action="store_true",
                        help="tracking only, without the Smart Tagging restarts")
    parser.add_argument("--no-broadcast", action="store_true",
                        help="tactical ball only, without positions from the broadcast tracking")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    team_of_track = team_of_each_track(meta_path)

    ball_fill = None
    if not args.no_broadcast:
        other_meta, other_raw, other_homography = require(
            ["atd_meta", "atd_raw", "homography"], GAME, source="broadcast")
        other_team = team_of_each_track(other_meta)
        ball, players = broadcast_ball.read_tracks(raw_path, n_frames, len(team_of_track))
        other_ball, other_players = broadcast_ball.read_tracks(
            other_raw, source_frames("broadcast", GAME), len(other_team))
        ball_fill, report = broadcast_ball.fill_from_broadcast(
            ball, players, team_of_track, other_ball, other_players, other_team,
            broadcast_ball.read_valid(other_homography),
            [(first, last) for _, first, last in half_video_blocks(GAME)], pitch,
            GAMES[GAME]["fps"])
        for block in report:
            print("frames {first}-{last}: tactical frame = broadcast frame + {offset}, agreeing on "
                  "{agreement:.0%} of shared ball frames; kept {kept} of {candidates} positions"
                  .format(**block))

    distances = nearest_by_team(raw_path, team_of_track, n_frames, pitch, ball_fill)

    extra = None
    if not args.no_tags:
        events = load_smart_tagging.load(GAME)
        codes = tag_team_codes([e["tags"].get("Team") for e in events], team_names(meta_path)[0])
        extra = restart_log_evidence(events, codes, n_frames)
        n_set_pieces = sum((e["code"] or "").strip().lower() == SET_PIECE_CODE for e in events)
        print("using {} Smart Tagging set pieces as restart evidence".format(n_set_pieces))

    labels = predict(distances, extra_evidence=extra)
    print("wrote {}".format(write_submission(args.out, labels)))
