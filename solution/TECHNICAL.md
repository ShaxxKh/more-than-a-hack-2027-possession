# Technical document

**Team:** Bodring
**Challenge:** More than a Hack 2027, Phase 1, ball possession

## Approach

We treat possession as a hidden state with three values, home (`H`), away (`A`) and dead ball
(`D`), and decode each half as a whole with the Viterbi algorithm.

Each frame contributes evidence for the three states. Every change of state pays a fixed cost
derived from how long possessions and stoppages last in football. A possession therefore ends
only when the evidence for ending it outweighs that cost. We chose this over per-frame
classification followed by smoothing because the ranking metric scores whole possessions, and a
transition cost acts on exactly that: it suppresses flicker without a window that blurs every
real change by the same amount.

Frames outside the two halves are written as `X`.

### Evidence per frame

- **Ball visible, within reach.** For each team we take the distance in metres from the ball to
  its nearest tracked player, capped at 30 m. When at least one player is within a control radius
  of 2 m, the log evidence for a team is minus its distance divided by a scale of 8 m.
- **Ball visible, out of reach.** When no player is within 2 m, the ball is in flight or loose
  and neither team gets evidence, so possession stays with the last team. Without this, a long
  pass travelling past opponents hands them the ball on the way.
- **Dead ball with a visible ball.** Dead ball gets the better of the two team values minus 0.5,
  so a visible ball is mild evidence that play is live.
- **Ball missing.** Neither team is favoured. Dead ball gets evidence that depends on the length
  of the whole missing-ball gap: none up to 2 s, then rising linearly to 0.3 per frame at 10 s.
  With our transition costs a gap must last about 5 s before it can split a possession.
- **Ball missing in the tactical tracking but seen by the broadcast tracking.** The frame is
  treated as a visible-ball frame, using the broadcast ball position and the tactical player
  positions. See "Ball positions from the broadcast tracking" below. A tactical ball position is
  never replaced.
- **Smart Tagging set pieces.** Inside each tagged set-piece window, the first half adds 0.05 per
  frame to dead ball and the second half adds 0.05 per frame to the tagged team. Throw-ins carry
  no team tag and contribute only the dead-ball part.

### Transitions

A possession lasts 12.7 s on average and a stoppage 21.1 s, and 74% of possessions end with the
other team rather than with a stoppage. These give a turnover a cost of about 6.1 units of log
evidence and a possession, stoppage, possession sequence a cost of about 14.1.

Through a stretch with no evidence, every frame is an equally good place for a change. We date
the change to the last such frame, which is when the next player receives the ball.

### Ball positions from the broadcast tracking

The tactical tracking has no ball on 64,643 graded frames, 99.8% of them in 132 gaps of 5 s or
longer. Gaps that long cannot be interpolated, so we take real positions from the broadcast
tracking where it has them. `broadcast_ball.py` does this in three steps, all computed from the
provided files.

1. **Alignment.** The broadcast is a live feed, so within a half its frames are a constant number
   of frames away from the tactical ones. We find that number per half as the shift at which the
   two ball trajectories correlate best, using the ball's movement over 24 frames, not its
   position, because position changes too slowly to give a sharp peak. The result is 4,014 frames
   for the first half and 26,920 for the second. At those shifts the two ball positions are
   within 3 m of each other on 80% and 78% of the frames where both exist; the code refuses to
   continue below 50%.
2. **Same-moment check.** Broadcast frames with no valid homography are dropped. For the rest we
   compare player positions: for each broadcast player, the distance to the nearest tactical
   player of the same team, and the median of those per frame. When both recordings show the same
   moment this is about 0.3 m. When the broadcast shows a replay it is 5 m or more.
3. **Trust rule.** A broadcast position is used only if that median is at most 2 m, computed over
   at least 4 players, and the broadcast holds the ball for at least 1 s without interruption.

Of 8,103 candidate frames, 5,605 pass and are used: 3.5% of graded frames. Ball coverage on graded
frames rises from 59.3% to 62.8%. Compared with the same model without this step, 6,510 graded
frames change label, 6,020 of them from dead ball to a team.

## Assets used

| asset | used | how |
|---|---|---|
| tactical tracking (ATD) | yes | ball and player positions; team of each track slot from the metadata |
| tactical Smart Tagging | partly | the 59 `SET PIECES` instances, with their `Team` tag |
| tactical homography | no | |
| tactical and broadcast video | no | |
| broadcast tracking (ATD) | yes | ball positions for frames where the tactical tracking has none; player positions for the same-moment check |
| broadcast homography | partly | the `valid` column only |

Not used from the assets above: player speeds, and every Smart Tagging code other than set
pieces. We did not try them and drop them; we ran out of time before reaching them.

## Training data

No model is trained. The settings come from the public Metrica Sports sample data,
<https://github.com/metrica-sports/sample-data>, at commit
`e706dd506b360d69d9d123d5b8026e7294b13996`: three anonymised games with tracking and event data.
The events were coded by Metrica and are used as published. We acknowledge Metrica Sports as the
source, as that repository asks.

- **Duration priors.** `measure_priors.py` derives possessions and stoppages from the events:
  938 possessions and 241 stoppages over the three games. The three transition numbers above are
  its output.
- **Control radius and distance scale.** `tune_on_sample_games.py` derives per-frame labels from
  the same events, runs the model on the sample tracking and scores it with the official metric
  library. Without a control radius the best mean spellF1 over the three games was 0.586. With a
  2 m radius and an 8 m scale it is 0.658 (0.667, 0.710 and 0.598 per game), and every game
  improves. That setting sits in the middle of a stable region, 1.5 to 3 m, and kept its lead
  when we added random noise of up to 2 m to the distances.

Nothing from the provided game was labelled by hand, and no label of the provided game was used
to choose a setting. The public 5-minute ground truth was used only to check that the pipeline
runs, that frames are aligned and that the file is well formed.

### Settings that are judgement, not measurement

The sample tracking is corrected and loses the ball only when it is dead, so it cannot inform how
a missing ball should be handled. These values were set by reasoning and are not fitted to
anything: the missing-ball ramp (2 s, 10 s, 0.3), the visible-ball penalty on dead ball (0.5), the
set-piece evidence (0.05), and the 2-second control threshold used when deriving possessions from
events.

The sample games also have a single camera, so the broadcast trust rule cannot be tuned on them
either. Its 2 m limit sits in the middle of an empty band in the distribution of the player
comparison on this match's tracking: 5,862 candidate frames are under 1 m, 1,739 are over 4 m and
127 lie between. That distribution is read from the tracking alone and involves no label. The
1-second minimum and the 24-frame movement window are judgement.

## Possession rules implemented

- **Three states only.** Every frame inside a half is `H`, `A` or `D`.
- **Loose and contested balls.** Possession stays with the last team until the other team has
  sustained evidence. While nobody is within 2 m of the ball, nothing changes. Once an opponent
  is within 2 m and 5 m closer to the ball than the team in possession, possession changes after
  about 0.4 s; larger advantages need proportionally less time.
- **Dead ball.** A stretch is dead when the ball is missing for long enough, or when a set-piece
  tag supports it. No team is assigned to a dead ball.
- **Restarts.** A tagged set piece favours its team in the second half of the tagged window. The
  exact restart frame is left to the tracking.
- **Abstention.** The model never writes `X` inside a half.

When deriving labels on the sample games, a change of possession counts only if the new team
completes a pass, carry or shot, or keeps the ball for 2 s. Challenges and fouls received never
change possession.

## Limitations

- **Missing ball.** The ball is absent from the tactical tracking on 41% of graded frames and
  this is where the model is weakest. Its handling rests on untested judgement values. The
  broadcast tracking recovers under a tenth of those frames, because both recordings tend to lose
  the ball at the same moments.
- **Broadcast positions are not verified.** Where both recordings see the ball and the players
  agree, about 5% of the two ball positions are still more than 10 m apart, and we cannot tell
  which recording is wrong. Inside a tactical gap there is no second opinion, so some of the
  positions we take over are false detections. One offset per half also leaves a timing error of
  a few frames.
- **Dead ball with a visible ball.** When the ball is tracked during a stoppage, the model tends
  to give it to the nearest team.
- **Being near the ball is not control.** The control radius removes evidence while the ball is
  out of reach, but a player who is within 2 m without touching the ball still counts, so a
  defender closing down a ball carrier can win it too early.
- **No trajectories or identities.** Ball speed and direction, player speed, and track continuity
  are ignored.
- **Durations.** A hidden Markov model implies geometrically distributed durations, which real
  possessions do not follow.
- **Small tuning set.** Three games, with labels derived from events. Those labels agree with
  the tracking's own ball-missing signal on 90 to 93% of frames, so they are approximate.

With more time we would add ball-trajectory evidence to bridge missing-ball gaps, use the other
Smart Tagging phases as soft team evidence, and replace the fixed transition cost with explicit
duration modelling.

## Reproduction

See [README.md](README.md). In short, with the data bundle in `data/`:

```bash
pip install -r solution/requirements.txt
python solution/hmm_possession.py
```

The output is deterministic. Python 3.15.2, numpy 2.5.4.

## Tools

The code was written with an AI coding assistant (Claude Code), which is available commercially.
It ran the scripts on our own machine and was shown only aggregate outputs such as scores and
counts. No video, tracking rows or tagging content were sent to it.
