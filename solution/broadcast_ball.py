"""Ball positions from the broadcast tracking, for frames where the tactical tracking has none.

The broadcast recording is on its own timeline (DATA.md). It is a live feed, so within one half
its frames are a constant number of frames away from the tactical ones; replays and close-ups do
not shift it, they only leave stretches with no pitch transform. This module

  1. finds that constant per half, by cross-correlating the two ball trajectories,
  2. carries the broadcast ball over to the tactical timeline, and
  3. keeps a position only where it can be trusted -- see accept_broadcast_ball.

Everything here is computed from the provided files. No offset, frame number or position is
written down by hand, which is what RULES.md asks of any correction to the tracking.

numpy only.
"""

import csv

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN"

MOVEMENT_FRAMES = 24  # about a second: long enough to rise above jitter, short enough to be sharp
# Trusting a broadcast position. These cannot be tuned on other games, which have one camera, so
# they are set from how the quantities are distributed in the tracking itself, never from labels.
# Scene error is bimodal: the same moment seen twice sits around 0.3 m, two unrelated moments at
# 5 m or more, with almost nothing between 1 and 4 m. 2 m is in the middle of that empty band.
MAX_SCENE_ERROR_M = 2.0
# A ball that the broadcast tracker holds for under a second, at a time the tactical tracker has
# lost it for five or more, is more likely a false detection than a sighting.
MIN_RUN_S = 1.0
MIN_PLAYERS_COMPARED = 4  # fewer than this and two scenes cannot be told apart
# An offset is only believed if, where both recordings see the ball, they mostly agree on where
# it is. A wrong offset pairs unrelated moments, and unrelated ball positions are rarely this close.
AGREEMENT_M = 3.0
MIN_AGREEMENT_SHARE = 0.5


def read_tracks(raw_path, n_frames, n_slots):
    """(ball, players) from a raw ATD file, normalised 0-1, NaN where not observed.

    ball is (n_frames, 2) and players is (n_frames, n_slots, 2). Row i is frame_id i + 1.
    """
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if parts[2][0] != MISSING:
                ball[row] = [float(v) for v in parts[2].split(",")[:2]]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    players[row, slot] = [float(v) for v in group.split(",")[:2]]
    return ball, players


def read_valid(homography_path):
    """Boolean per video frame: whether a pitch transform exists for it (DATA.md, Homography)."""
    with open(homography_path, newline="") as handle:
        return np.array([row["valid"] == "1" for row in csv.DictReader(handle)])


def run_lengths(mask):
    """For each True in a boolean array, the length of the run of True containing it. 0 elsewhere."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    lengths = np.zeros(len(mask), dtype=np.int64)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lengths[start:stop] = stop - start
    return lengths


def _signal(ball):
    """A ball track as one complex number per frame: its movement over MOVEMENT_FRAMES.

    Movement rather than position, because where the ball is changes slowly -- it stays in one
    half of the pitch for a long time -- so positions correlate almost as well at a wrong shift
    as at the right one. How it moves is different from one second to the next. Zero where
    either end of the movement is missing.
    """
    half = MOVEMENT_FRAMES // 2
    moved = np.zeros(ball.shape)
    moved[half:-half] = np.nan_to_num(ball[2 * half:] - ball[:-2 * half])
    return moved[:, 0] + 1j * moved[:, 1]


def estimate_offset(reference_ball, other_ball, pitch):
    """(offset, agreement): reference frame = other frame + offset.

    The offset is the shift at which the two ball trajectories correlate best. agreement is the
    share of frames, among those where both see the ball at that shift, on which they are within
    AGREEMENT_M of each other -- the check that the peak is a real match and not the best of noise.
    """
    size = 1 << int(np.ceil(np.log2(len(reference_ball) + len(other_ball))))
    correlation = np.fft.ifft(np.fft.fft(_signal(reference_ball), size)
                              * np.conj(np.fft.fft(_signal(other_ball), size))).real
    peak = int(correlation.argmax())
    offset = peak if peak < size // 2 else peak - size

    rows, other_rows = paired_rows(offset, 0, len(reference_ball) - 1, len(other_ball))
    metres = np.hypot(*((reference_ball[rows] - other_ball[other_rows]) * np.array(pitch)).T)
    both = ~np.isnan(metres)
    agreement = float((metres[both] <= AGREEMENT_M).mean()) if both.any() else 0.0
    return offset, agreement


def paired_rows(offset, first, last, n_other):
    """(rows, other_rows): the reference rows first..last that have a partner, and those partners."""
    rows = np.arange(first, last + 1)
    other_rows = rows - offset
    inside = (other_rows >= 0) & (other_rows < n_other)
    return rows[inside], other_rows[inside]


def scene_error(players, team, other_players, other_team, pitch):
    """How differently two recordings place the players, frame by frame.

    players is (n, k, 2) with team (k,), other_players is (n, m, 2) with other_team (m,). For
    every player in `other`, take the distance in metres to the nearest player of the same team
    in the first. Return (error, compared): the median of those distances per frame, and how many
    players it is the median of. error is NaN where fewer than MIN_PLAYERS_COMPARED could be
    compared.

    When both recordings show the same moment this is well under a metre. When the broadcast is
    showing a replay it is the distance between two unrelated moments, several metres.
    """
    nearest = []
    for side in (0, 1):
        ours, theirs = players[:, team == side], other_players[:, other_team == side]
        delta = (theirs[:, :, np.newaxis, :] - ours[:, np.newaxis, :, :]) * np.array(pitch)
        distance = np.hypot(delta[..., 0], delta[..., 1])
        distance[np.isnan(distance)] = np.inf
        nearest.append(distance.min(axis=2, initial=np.inf))  # inf: nobody to compare with
    nearest = np.concatenate(nearest, axis=1)
    nearest[np.isinf(nearest)] = np.nan

    compared = (~np.isnan(nearest)).sum(axis=1)
    error = np.full(len(nearest), np.nan)
    enough = compared >= MIN_PLAYERS_COMPARED
    error[enough] = np.nanmedian(nearest[enough], axis=1)
    return error, compared


def accept_broadcast_ball(scene_error_m, n_compared, run_s):
    """Which broadcast ball positions to trust. Return a boolean array, one per candidate frame.

    A candidate is a frame where the tactical tracking has no ball and the broadcast, carried
    over to the tactical timeline, has one. For each candidate:

      scene_error_m  median metres between the two recordings' player positions on that frame;
                     NaN where too few players were visible to compare
      n_compared     how many players that median is over
      run_s          length in seconds of the unbroken stretch of candidates the frame is part of

    A wrong position costs more than a missing one: it becomes distance evidence exactly as if
    the tactical tracking had seen the ball there, while a rejected one leaves the frame as
    "ball missing", which the model already handles. So both tests must pass, and a frame whose
    scene could not be compared (NaN) is rejected -- NaN <= x is False, which is what we want.
    n_compared is not tested beyond the MIN_PLAYERS_COMPARED already built into scene_error.
    """
    same_moment = scene_error_m <= MAX_SCENE_ERROR_M
    sustained = run_s >= MIN_RUN_S
    return same_moment & sustained


def fill_from_broadcast(ball, players, team, other_ball, other_players, other_team, other_valid,
                        blocks, pitch, fps, accept=accept_broadcast_ball):
    """(fill, report): broadcast ball positions for tactical frames that have no ball.

    fill is (n_frames, 2), normalised, NaN wherever nothing trusted is available -- including
    every frame where the tactical tracking already has the ball, which is never overridden.
    blocks is [(first_frame, last_frame)], one per half; each gets its own offset. report is one
    dict per block with the offset, its agreement and the frame counts.
    """
    other_ball = np.where(other_valid[:, np.newaxis], other_ball, np.nan)  # no pitch transform
    fill = np.full(ball.shape, np.nan)
    report = []
    for first, last in blocks:
        in_block = np.full(ball.shape, np.nan)
        in_block[first:last + 1] = ball[first:last + 1]
        offset, agreement = estimate_offset(in_block, other_ball, pitch)
        if agreement < MIN_AGREEMENT_SHARE:
            raise ValueError("frames {}-{}: no offset aligns the two ball tracks (best {} agrees "
                             "on {:.0%} of shared frames)".format(first, last, offset, agreement))

        rows, other_rows = paired_rows(offset, first, last, len(other_ball))
        candidate = np.isnan(ball[rows, 0]) & ~np.isnan(other_ball[other_rows, 0])
        on_timeline = np.zeros(len(ball), dtype=bool)
        on_timeline[rows[candidate]] = True
        rows, other_rows = rows[candidate], other_rows[candidate]

        error, compared = scene_error(players[rows], team, other_players[other_rows], other_team,
                                      pitch)
        keep = np.asarray(accept(error, compared, run_lengths(on_timeline)[rows] / float(fps)),
                          dtype=bool)
        fill[rows[keep]] = other_ball[other_rows[keep]]
        report.append({"first": first, "last": last, "offset": offset, "agreement": agreement,
                       "candidates": int(candidate.sum()), "kept": int(keep.sum())})
    return fill, report
