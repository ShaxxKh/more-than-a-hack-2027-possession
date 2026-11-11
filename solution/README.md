# Our solution

A three-state possession model (home, away, dead ball) decoded with Viterbi. The method, the data
it uses and its limits are described in [TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission file

Python 3.15.2. From the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r solution/requirements.txt
```

Unpack the match bundle into `data/` as [DATA.md](../DATA.md) describes, then:

```bash
.venv/bin/python solution/hmm_possession.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/solution_hmm.csv
```

The first command writes `evaluation/results/solution_hmm.csv` in a few seconds. It is
deterministic: the same data bundle always produces the same file, byte for byte. `--out` writes
it somewhere else, and `--no-tags` runs on tracking alone, without the Smart Tagging.

## Reproduce the settings

The model's settings come from other games, never from the challenge match. Both scripts download
the Metrica sample data at a pinned commit into `data/external/`, which is not committed.

```bash
.venv/bin/python solution/measure_priors.py --fetch
.venv/bin/python solution/tune_on_sample_games.py --fetch
```

The first prints the three duration priors used in `hmm_possession.py`. The second scores the
model on the three sample games across a grid of distance settings.

## Tests

```bash
.venv/bin/python -m unittest discover -s solution
```

24 tests, all on synthetic or hand-written inputs, so they run without any match data.

## Files

| file | what it does |
|---|---|
| `hmm_possession.py` | the model: evidence, transitions, Viterbi, and the command that writes the file |
| `measure_priors.py` | measures possession and stoppage durations on other games |
| `tune_on_sample_games.py` | scores the model on other games with the official metric |
| `test_*.py` | tests |
| `requirements.txt` | pinned dependencies |

The rest of this repository is the organisers' starter kit, unchanged, under its MIT licence.
