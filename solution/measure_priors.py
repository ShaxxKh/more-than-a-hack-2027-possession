"""Measure the possession model's priors on OTHER games. Nothing here reads the challenge match.

    python solution/measure_priors.py --fetch     # download the event files, then measure
    python solution/measure_priors.py             # measure what is already downloaded

Source: Metrica Sports sample data, https://github.com/metrica-sports/sample-data, three
anonymised games with human-coded events. Pinned to one commit so the numbers are reproducible.
Only the event files are used; they land in data/external/, which is gitignored.

How possession is derived from events
-------------------------------------
* Every restart is a SET PIECE event. The ball is dead from the end of the last event before a
  restart until that restart, and alive otherwise.
* While the ball is alive, the team in possession is the team of the latest on-ball event.
  CHALLENGE and FAULT RECEIVED are logged for both sides, so they never change possession.
* A change only counts if it is controlled: the new team completes an action (PASS, CARRY, SHOT)
  or keeps the ball for CONTROL_MIN_S. Otherwise possession stays with the previous team, which
  is the rule CHALLENGE.md states for loose and contested balls.

Standard library only, so it runs in any interpreter.
"""

import argparse
import csv
import json
import os
import statistics
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(ROOT, "data", "external", "sample-data")

SOURCE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
SOURCE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/{}"
FILES = (
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
)

RESTART = "SET PIECE"
IGNORED = {"CARD"}  # shown while the ball is dead; says nothing about when play stopped
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT", RESTART}
COMPLETED = {"PASS", "CARRY", "SHOT", RESTART}
CONTROL_MIN_S = 2.0  # "a player keeps the ball for a few seconds" -- CHALLENGE.md


def fetch(folder):
    """Download the event files at the pinned commit."""
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for name in FILES:
        path = os.path.join(folder, os.path.basename(name))
        urllib.request.urlretrieve(SOURCE_URL.format(SOURCE_COMMIT, name), path)
        print("fetched {}".format(os.path.basename(path)))


def read_events(path):
    """[{team, type, period, start, end}] in file order, times in seconds."""
    if path.endswith(".json"):
        with open(path) as handle:
            rows = [(e["team"]["name"], e["type"]["name"], e["period"],
                     e["start"]["time"], e["end"]["time"]) for e in json.load(handle)["data"]]
    else:
        with open(path, newline="") as handle:
            rows = [(r["Team"], r["Type"], r["Period"], r["Start Time [s]"], r["End Time [s]"])
                    for r in csv.DictReader(handle)]

    events = []
    for team, kind, period, start, end in rows:
        start = float(start)
        # Restarts carry no end time in the CSV format; an event cannot end before it starts.
        events.append({"team": team, "type": kind, "period": int(period), "start": start,
                       "end": max(start, float(end or 0.0))})
    return events


def alive_stretches(events):
    """(stretches, stoppages). A stretch is one restart-to-stoppage run of live play.

    stretch = {period, start, stop, touches: [(time, team, type)],
               ended_by: 'stoppage' | 'period'}
    stoppages = seconds the ball was dead before each restart, the first of a period excluded.
    """
    stretches, stoppages = [], []
    for period in sorted({event["period"] for event in events}):
        current, last_end = None, None
        for event in events:
            if event["period"] != period or event["type"] in IGNORED:
                continue
            if event["type"] == RESTART:
                if current is not None:
                    current["stop"] = last_end
                    stretches.append(current)
                    stoppages.append(max(0.0, event["start"] - last_end))
                current = {"period": period, "start": event["start"], "touches": [],
                           "ended_by": "stoppage"}
                last_end = event["start"]
            if current is None:
                continue  # before the period's first restart
            last_end = max(last_end, event["end"])
            if event["type"] in ON_BALL:
                current["touches"].append((event["start"], event["team"], event["type"]))
        if current is not None:
            current["stop"] = last_end
            current["ended_by"] = "period"
            stretches.append(current)
    return stretches, stoppages


def possessions(stretch, controlled_only=True):
    """[(team, start, stop)] possessions inside one stretch of live play."""
    runs = []  # [team, start, completed an action]
    for time, team, kind in stretch["touches"]:
        if runs and runs[-1][0] == team:
            runs[-1][2] = runs[-1][2] or kind in COMPLETED
        else:
            runs.append([team, time, kind in COMPLETED])

    spells = []
    for index, (team, start, completed) in enumerate(runs):
        stop = runs[index + 1][1] if index + 1 < len(runs) else stretch["stop"]
        controlled = completed or stop - start >= CONTROL_MIN_S
        if spells and (team == spells[-1][0] or (controlled_only and not controlled)):
            spells[-1][2] = stop  # same team, or an uncontrolled touch: the possession carries on
        else:
            spells.append([team, start, stop])
    return [(team, start, max(start, stop)) for team, start, stop in spells]


def measure(events, controlled_only=True):
    """Raw measurements for one game, kept as lists so games can be pooled."""
    stretches, stoppages = alive_stretches(events)
    durations, turnovers, to_stoppage = [], 0, 0
    for stretch in stretches:
        spells = possessions(stretch, controlled_only)
        durations.extend(stop - start for _, start, stop in spells)
        if spells:
            turnovers += len(spells) - 1
            to_stoppage += stretch["ended_by"] == "stoppage"
    return {"durations": durations, "stoppages": stoppages,
            "turnovers": turnovers, "to_stoppage": to_stoppage}


def pool(results):
    return {"durations": [d for r in results for d in r["durations"]],
            "stoppages": [s for r in results for s in r["stoppages"]],
            "turnovers": sum(r["turnovers"] for r in results),
            "to_stoppage": sum(r["to_stoppage"] for r in results)}


def priors(result):
    """(mean possession s, mean stoppage s, turnover share) -- the model's three priors."""
    endings = result["turnovers"] + result["to_stoppage"]
    return (statistics.mean(result["durations"]), statistics.mean(result["stoppages"]),
            result["turnovers"] / float(endings))


def print_row(name, result):
    possession, stoppage, share = priors(result)
    print("{:<16} {:>5} {:>7.1f} {:>7.1f}   {:>5} {:>7.1f} {:>7.1f}   {:>6.2f}".format(
        name, len(result["durations"]), possession, statistics.median(result["durations"]),
        len(result["stoppages"]), stoppage, statistics.median(result["stoppages"]), share))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=DEFAULT_DIR, help="where the event files are")
    parser.add_argument("--fetch", action="store_true", help="download the event files first")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.dir)
    paths = [os.path.join(args.dir, os.path.basename(name)) for name in FILES]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit("missing:\n  {}\nrun with --fetch".format("\n  ".join(missing)))
    games = [(os.path.basename(path).split("_Raw")[0].split("_events")[0], read_events(path))
             for path in paths]

    for title, controlled_only in (("controlled possession (used for the priors)", True),
                                   ("any touch changes possession (for comparison)", False)):
        print("\n{}".format(title))
        print("{:<16} {:>5} {:>7} {:>7}   {:>5} {:>7} {:>7}   {:>6}".format(
            "game", "poss", "mean s", "med s", "stops", "mean s", "med s", "turnov"))
        results = [measure(events, controlled_only) for _, events in games]
        for (name, _), result in zip(games, results):
            print_row(name, result)
        print_row("pooled", pool(results))
        if controlled_only:
            chosen = priors(pool(results))

    print("\npriors for solution/hmm_possession.py")
    print("  MEAN_POSSESSION_S = {:.1f}".format(chosen[0]))
    print("  MEAN_STOPPAGE_S   = {:.1f}".format(chosen[1]))
    print("  TURNOVER_SHARE    = {:.2f}".format(chosen[2]))
