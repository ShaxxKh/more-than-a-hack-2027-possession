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
- **Smart Tagging set pieces.** Inside each tagged set-piece window, the first half adds 0.05 per
  frame to dead ball and the second half adds 0.05 per frame to the tagged team. Throw-ins carry
  no team tag and contribute only the dead-ball part.

### Transitions

A possession lasts 12.7 s on average and a stoppage 21.1 s, and 74% of possessions end with the
other team rather than with a stoppage. These give a turnover a cost of about 6.1 units of log
evidence and a possession, stoppage, possession sequence a cost of about 14.1.

Through a stretch with no evidence, every frame is an equally good place for a change. We date
the change to the last such frame, which is when the next player receives the ball.

## Assets used

| asset | used | how |
|---|---|---|
| tactical tracking (ATD) | yes | ball and player positions; team of each track slot from the metadata |
| tactical Smart Tagging | partly | the 59 `SET PIECES` instances, with their `Team` tag |
| tactical homography | no | |
| tactical and broadcast video | no | |
| broadcast tracking and homography | no | not aligned to the submission timeline |

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

- **Missing ball.** The ball is absent on 41% of graded frames and this is where the model is
  weakest. Its handling rests on untested judgement values.
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
