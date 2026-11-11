"""Score the possession model on OTHER games, to choose settings without touching the challenge match.

    python solution/tune_on_sample_games.py --fetch     # download tracking + events, then sweep
    python solution/tune_on_sample_games.py             # sweep what is already downloaded

The three Metrica sample-data games have tracking and human-coded events. Labels are derived from
the events exactly as in measure_priors.py, the model runs on the tracking, and the official
metric library scores the result. RULES.md allows this: the labels are on other games.

What transfers and what does not
--------------------------------
This tracking is corrected: the ball is visible whenever play is live and missing only when it is
dead. The challenge tracking loses the ball on 41% of graded frames. So here a missing ball is
simply treated as dead, and the sweep tells you about the distance evidence -- who has the ball
when it is visible -- and nothing about how to handle a missing ball.
"""

import argparse
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import hmm_possession as hp  # noqa: E402
import measure_priors as mp  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)  # stated in the sample-data README
TRACKING_FILES = (
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
)
HOME_NAMES = {"Home", "Team A"}  # Team A is listed first in game 3's metadata

# In this corrected tracking a missing ball means a dead ball, so say so firmly.
MISSING_MEANS_DEAD = 1.0

DISTANCE_SCALES_M = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0)
MAX_DISTANCES_M = (15.0, 30.0, 60.0)


def fetch(folder):
    mp.fetch(folder)
    for name in TRACKING_FILES:
        path = os.path.join(folder, os.path.basename(name))
        urllib.request.urlretrieve(mp.SOURCE_URL.format(mp.SOURCE_COMMIT, name), path)
        print("fetched {}".format(os.path.basename(path)))


def nearest_two(players, team_is_home, ball):
    """(n, 2) metres from the ball to the nearest home and away player.

    players is (n, k, 2), ball is (n, 2), both normalised 0-1; team_is_home is (k,) or (n, k).
    Same contract as hp.nearest_by_team: NaN where the ball is missing, inf for an absent team.
    """
    delta = (players - ball[:, np.newaxis, :]) * np.array(PITCH)
    distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
    distance[np.isnan(distance)] = np.inf
    home = np.where(team_is_home, distance, np.inf).min(axis=1)
    away = np.where(team_is_home, np.inf, distance).min(axis=1)
    out = np.column_stack((home, away))
    out[np.isnan(ball[:, 0])] = np.nan
    return out


def distances_csv(home_path, away_path):
    """Games 1 and 2: one CSV per team, three header rows, the ball in the last two columns."""
    home = pd.read_csv(home_path, skiprows=3, header=None).values[:, 3:].astype(np.float64)
    away = pd.read_csv(away_path, skiprows=3, header=None).values[:, 3:].astype(np.float64)
    ball = home[:, -2:]
    home_players = home[:, :-2].reshape(len(home), -1, 2)
    away_players = away[:, :-2].reshape(len(away), -1, 2)
    players = np.concatenate((home_players, away_players), axis=1)
    is_home = np.arange(players.shape[1]) < home_players.shape[1]
    return nearest_two(players, is_home, ball)


def distances_epts(meta_path, raw_path):
    """Game 3: FIFA/EPTS. Which player a column holds changes at every substitution, so the
    column-to-team map is read per frame range from the metadata."""
    root = ET.parse(meta_path).getroot()
    home_id = next(root.iter("Team")).get("id")
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    frames, rows = [], []
    with open(raw_path) as handle:
        for line in handle:
            frame, groups, ball = line.strip().split(":")
            frames.append(int(frame))
            rows.append([float(v) for v in (groups.replace(";", ",") + "," + ball).split(",")])
    frames, values = np.array(frames), np.array(rows)
    players, ball = values[:, :-2].reshape(len(values), -1, 2), values[:, -2:]

    is_home = np.zeros(players.shape[:2], dtype=bool)
    for spec in root.iter("DataFormatSpecification"):
        slots = []
        for ref in spec.iter("PlayerChannelRef"):
            player = player_of_channel[ref.get("playerChannelId")]
            if player not in slots:
                slots.append(player)
        rows_of_spec = (frames >= int(spec.get("startFrame"))) & (frames <= int(spec.get("endFrame")))
        is_home[rows_of_spec] = [team_of_player[p] == home_id for p in slots]
    return nearest_two(players, is_home, ball)


def labels_from_events(events, n_frames):
    """(codes, blocks): per-frame H/A/D/X in the metric library's coding, and the graded blocks.

    Row i is frame i + 1 of the tracking, and an event at t seconds is frame t * 25.
    """
    def row(seconds):
        return min(n_frames, max(0, int(round(seconds * FPS)) - 1))

    codes = np.full(n_frames, pm.X, dtype=np.int8)
    stretches, _ = mp.alive_stretches(events)
    blocks = []
    for period in sorted({s["period"] for s in stretches}):
        in_period = [s for s in stretches if s["period"] == period]
        first, stop = row(in_period[0]["start"]), row(in_period[-1]["stop"])
        codes[first:stop] = pm.D
        blocks.append((first, stop - 1))
        for stretch in in_period:
            for team, start, end in mp.possessions(stretch):
                codes[row(start):row(end)] = pm.H if team in HOME_NAMES else pm.A
    return codes, blocks


def load_games(folder):
    """[(name, distances, gt_codes, blocks)] for the three sample games."""
    def path(name):
        return os.path.join(folder, name)

    games = []
    for number in (1, 2, 3):
        name = "Sample_Game_{}".format(number)
        if number < 3:
            distances = distances_csv(path(name + "_RawTrackingData_Home_Team.csv"),
                                      path(name + "_RawTrackingData_Away_Team.csv"))
            events = mp.read_events(path(name + "_RawEventsData.csv"))
        else:
            distances = distances_epts(path(name + "_metadata.xml"), path(name + "_tracking.txt"))
            events = mp.read_events(path(name + "_events.json"))
        codes, blocks = labels_from_events(events, len(distances))
        games.append((name, distances, codes, blocks))
    return games


def missing_is_dead(gap_seconds):
    out = np.zeros((len(gap_seconds), 3))
    out[:, hp.D] = MISSING_MEANS_DEAD
    return out


def score(game, distance_scale, max_distance):
    """The official metrics for one game at one setting."""
    _, distances, gt_codes, blocks = game
    saved = hp.DISTANCE_SCALE_M, hp.MAX_DISTANCE_M
    hp.DISTANCE_SCALE_M, hp.MAX_DISTANCE_M = distance_scale, max_distance
    try:
        labels = hp.predict(distances, missing_evidence=missing_is_dead, blocks=blocks)
    finally:
        hp.DISTANCE_SCALE_M, hp.MAX_DISTANCE_M = saved
    return pm.evaluate(gt_codes, pm.label_codes(labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=mp.DEFAULT_DIR, help="where the sample-data files are")
    parser.add_argument("--fetch", action="store_true", help="download the files first")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.dir)
    games = load_games(args.dir)

    print("label check: how often 'ball missing' and 'labelled dead' agree on graded frames")
    for name, distances, codes, _ in games:
        graded = codes != pm.X
        agree = np.mean(np.isnan(distances[graded, 0]) == (codes[graded] == pm.D))
        print("  {}  {:.1f}%   ({} graded frames)".format(name, 100.0 * agree, int(graded.sum())))

    current = (hp.DISTANCE_SCALE_M, hp.MAX_DISTANCE_M)
    print("\nspellF1 per game; poss = predicted possessions / true possessions, all games")
    print("{:>7} {:>5}   {:>7} {:>7} {:>7}   {:>7} {:>7}   {:>5}".format(
        "scale m", "max m", "game 1", "game 2", "game 3", "mean", "macroF1", "poss"))
    for max_distance in MAX_DISTANCES_M:
        for scale in DISTANCE_SCALES_M:
            results = [score(game, scale, max_distance) for game in games]
            spell = [r["spell_f1_mean"] for r in results]
            print("{:>7.1f} {:>5.0f}   {:>7.4f} {:>7.4f} {:>7.4f}   {:>7.4f} {:>7.4f}   {:>5.2f}{}"
                  .format(scale, max_distance, spell[0], spell[1], spell[2], np.mean(spell),
                          np.mean([r["macro_f1"] for r in results]),
                          sum(r["n_pred_spells"] for r in results)
                          / float(sum(r["n_gt_spells"] for r in results)),
                          "   <- current" if (scale, max_distance) == current else ""))
