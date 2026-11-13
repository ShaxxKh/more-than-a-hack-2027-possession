# What we tried

A record of the work that followed the first submission: what we looked at, what we built, what
we dropped, and what each step did. [TECHNICAL.md](TECHNICAL.md) describes the model as it is
now; this file describes how it got there.

The question we started from was the sentence in the challenge description: the tracking is
automatic and uncorrected, and the ball is missing on 41% of graded frames. Can the tracking be
improved?

## Summary

| step | outcome |
|---|---|
| Interpolate short ball gaps | not built: there are almost none to interpolate |
| Clean up ball jumps and duplicate tracks | not built: too rare to matter |
| Take ball positions from the broadcast tracking | built, commit `36a71e7` |
| Let a visible ball count towards a stoppage | built, commit `1c64c9c` |

| version | flags to reproduce | spellF1 | macroF1 | dead-ball F1 | share labelled `D` | segments |
|---|---|---|---|---|---|---|
| first submission (`dc99f74`) | `--no-broadcast --no-visible-dead` | 0.477 | 0.668 | 0.421 | 40.7% | 404 |
| + broadcast ball | `--no-visible-dead` | 0.533 | 0.687 | 0.446 | 36.9% | 397 |
| + visible dead ball (current) | none | 0.547 | 0.709 | 0.501 | 40.1% | 435 |

The three score columns are on the public five minutes. They show that the pipeline is aligned
and that nothing broke. They are not evidence that one version is better than another: the window
holds 17 possessions, and [EVALUATION.md](../EVALUATION.md) explains why that cannot rank
anything. The last two columns are over all 158,857 graded frames and involve no labels.

No setting was chosen on the public ground truth. Where we looked at it, we say so below.

## 1. Interpolating short ball gaps: not built

The plan was to interpolate the ball through short gaps, where a straight line between the last
and next sighting is a fair guess. We measured the gaps first.

Missing-ball gaps on graded frames, 64,643 frames in 150 gaps:

| gap length | gaps | missing frames | share of missing |
|---|---|---|---|
| under 1 s | 17 | 75 | 0.1% |
| 1 s to 4.9 s | 1 | 122 | 0.2% |
| 5 s to 10 s | 53 | 9,343 | 14.5% |
| 10 s to 30 s | 60 | 26,770 | 41.4% |
| over 30 s | 19 | 28,333 | 43.8% |

There is nothing between 1 and 4.9 seconds. The provider's pipeline appears to bridge short
losses itself and give up after about 5 seconds, so the "raw" tracking is already partly
repaired. We checked whether that bridging is linear interpolation, which we could have detected
and down-weighted: it is not, there are no long constant-velocity stretches in the visible ball.

Interpolating what is left would recover 75 frames. The long gaps cannot be interpolated, because
over 5 to 30 seconds the ball can change hands several times.

## 2. Other tracking noise: not built

Measured on graded frames and judged too rare to be worth a correction step:

- 216 frame-to-frame ball jumps faster than 40 m/s, about 2 m each.
- Player tracks jump faster than 12 m/s on 0.02% of steps.
- Two tracks of the same team within 0.5 m of each other on about 3.6% of sampled frames.
- Fewer than 20 tracked players on 15% of frames, more than 22 on 2.4%.

Missing players near the ball could still flip which team is nearest. We did not measure that.

## 3. What the missing frames are

Before choosing what to build, we looked at the six missing-ball gaps inside the public five
minutes, as a check on what the model does with them. Two last a few frames and do no harm. Two
are stoppages, or mostly so, and the model called them dead. The other two, of 5.3 s and 9.3 s,
are entirely one team's possession, and the model called both dead as well.
So long gaps are a mix, and only real ball positions can separate the two kinds.

The same check showed a second problem, unrelated to missing data: on frames where the ball is
visible, the ground truth has 975 dead-ball frames and the model had none. Section 5 covers that.

## 4. Ball positions from the broadcast tracking: built

The broadcast recording has its own tracking, on its own timeline, with no mapping provided.

### Alignment

A live TV feed stays in real time through replays, so within a half the mapping should be a
constant shift. A first exploratory pass confirmed it: cross-correlating ball positions in chunks
of 4,000 broadcast frames gave the same shift in every chunk of a half to within 13 frames, with
a median disagreement of 0.9 m between the two ball positions.

That first estimator correlated ball positions. When we wrote tests for it, it failed one: on a
synthetic ball that drifts slowly, it was off by 200 frames. Position changes slowly, so a wrong
shift correlates almost as well as the right one. The estimator in
[broadcast_ball.py](broadcast_ball.py) correlates the ball's movement over 24 frames instead,
which gives a sharp peak. It finds 4,014 frames for the first half and 26,920 for the second,
within one frame of the exploratory values.

### Coverage

| | frames |
|---|---|
| graded frames with no tactical ball | 64,643 |
| of those, broadcast has a ball on a frame with a valid homography | 8,103 |
| of those, the broadcast is showing the same moment and holds the ball for 1 s | 5,605 |

The two recordings lose the ball at the same moments. Of the 133 long gaps, 48 have no broadcast
ball at all and 53 have under a quarter of their frames covered. The help is concentrated in the
5 to 10 second gaps, where the broadcast covers a third of the frames.

### Telling a replay from live play

Inside a tactical gap there is no tactical ball to compare with, but both recordings track the
players. For each broadcast player we take the distance to the nearest tactical player of the
same team, and the median per frame. When the two recordings show the same moment this is about
0.3 m. As a control we shifted the broadcast by 30 seconds, and it became 5 m or more. On the
8,103 candidates: 5,862 under 1 m, 127 between 1 and 4 m, 1,739 over 4 m, the rest not
comparable. So about a fifth of the candidates were replays.

Matching scenes do not guarantee a correct ball. Where both recordings see the ball and the
players agree, about 5% of the two ball positions are still more than 10 m apart. The 1-second
minimum is the only guard against that, and it is a judgement value.

### Result

Ball coverage on graded frames rises from 59.3% to 62.8%. Against the first submission, 6,510
graded frames changed label: 6,020 from dead ball to a team and 490 from one team to the other.
132 of the changed frames are inside the public window.

These settings could not be tuned: the sample games have one camera.

## 5. Letting a visible ball count towards a stoppage: built

On a frame with a visible ball, the dead-ball score was the better team's score minus 0.5. It
could never win, so the model called a stoppage only when the ball disappeared. This is a
property of the scoring, visible in the code; the public window only made it obvious.

Two signs separate a stopped ball from a live one without any label: it is off the pitch, or it
is not moving.

### What the sample games can and cannot say

Their tracking is corrected and shows a dead ball on few frames, so they cannot show the
evidence helping. They can show how often a live ball triggers it.

| condition | share of live frames, three sample games |
|---|---|
| off the pitch by more than 0 m | 0.6% to 2.8% |
| off the pitch by more than 1 m | 0.1% to 1.6% |
| off the pitch by more than 2 m | 0.0% to 0.3% |
| within 0.3 m of one spot for 1 s or more | 0.7% to 1.9% |
| within 0.5 m of one spot for 1 s or more | 1.7% to 3.9% |
| within 1.0 m of one spot for 1 s or more | 5.3% to 10.5% |

That fixed the stillness radius at 0.3 m and the pitch margin at 2 m.

### The stillness minimum: 1 second, then 3

The first version used 1 second. Mean spellF1 on the sample games, from
`tune_on_sample_games.py`, at a 2 m margin:

| setting | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| evidence off | 0.667 | 0.710 | 0.598 | 0.658 |
| still for 1 s, evidence 1.0 (first version) | 0.641 | 0.693 | 0.588 | 0.641 |
| still for 2 s, evidence 1.0 | 0.650 | 0.703 | 0.595 | 0.649 |
| still for 3 s, evidence 1.0 (current) | 0.657 | 0.705 | 0.591 | 0.651 |

Every setting scores below "off". Run separately, the off-pitch sign costs 0.001 and the
stillness sign accounts for the rest.

To see whether that cost was real, we looked at the events around each still stretch that the
labels call live, in games 1 and 2:

| stretch length | stretches | what follows |
|---|---|---|
| 1 to 2 s | 13 | 6 an ordinary pass, the rest mixed |
| 2 to 3 s | 5 | 3 a goal kick, 1 an ordinary pass, 1 another restart |
| 3 s or longer | 8 | 7 a goal kick, 1 the end of a half |

Short rests are often a player standing on the ball, so 1 second was wrong. The long ones are
goal-kick waits. The labels are derived from events, goal kicks have no restart event of their
own, and so the wait before them is labelled live. At 3 seconds the remaining cost on the sample
games is the labels penalising a correct call, and we accepted it.

The size of the evidence made no difference between 0.75 and 2.0 once stillness needed 3 s. It
is 1.0.

### Result

The evidence is present on 5,064 graded frames, in 81 stretches. Against the broadcast-only
version, 6,403 graded frames changed label: 5,188 from a team to dead ball, and 1,215 from one
team to the other, because a new stoppage splits a possession and play resumes with whoever has
the ball next. 81 of the changed frames are inside the public window.

The benefit is not measured on any labelled game.

## What we did not try

- Re-detecting the ball from the video. Allowed, and the only way to get positions for the gaps
  where both trackings are blind.
- A per-chunk offset for the broadcast. One offset per half leaves a few frames of timing error.
- Filling in missing players, or merging duplicate tracks.
- Player movement as evidence while the ball is missing.
- The Smart Tagging phases other than set pieces.
- Recognising a ball carried to a restart, or a goalkeeper holding the ball in open play.

## Reproducing the numbers

The versions in the summary table come from the flags listed there:

```bash
python solution/hmm_possession.py --no-broadcast --no-visible-dead --out first.csv
```

```bash
python solution/hmm_possession.py --no-visible-dead --out broadcast_only.csv
```

```bash
python solution/hmm_possession.py
```

The sample-game table comes from `python solution/tune_on_sample_games.py`.

The gap, noise, coverage and event tables in sections 1 to 5 were produced with one-off analysis
scripts that are not in this repository. The counts that the pipeline itself prints, such as the
offsets and the number of broadcast positions kept, are reproduced by every run.
