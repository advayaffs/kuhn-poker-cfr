# Kuhn Poker — CFR Solver

A solver for three-card, two-player poker using counterfactual regret minimization.

## Overview

Kuhn Poker is small enough to enumerate every possible deal, but still contains bluffing and hidden information. I use it to study how an equilibrium strategy emerges from repeated regret updates rather than from hard-coded rules.

## Method

- Enumerate all six ordered deals of J, Q and K. Each player antes one chip; a single bet is permitted.
- Represent an information set by the player's private card and public action history.
- Traverse the game tree, weighting regret updates by the opponent's reach probability and average-strategy updates by the player's own reach probability.
- Use regret matching: positive cumulative regrets determine the next action probabilities; if none are positive, use a uniform distribution.
- Evaluate the average strategy against exact best responses. The benchmark is the known first-player game value, −1/18 chips per hand.

## Run

```bash
python kuhn_poker_clean.py
```

The script trains for 10,000 iterations. Enter `n` when prompted to skip interactive play, or `y` to play against the learned policy. Optional plotting: `pip install matplotlib`.

## Reproduced results

| Metric | Value |
|---|---:|
| Iterations | 10,000 |
| First-player expected value | −0.055546 |
| Theoretical value | −0.055556 |
| Exploitability at iteration 10,000 | 0.002318 chips/hand |

![Training exploitability](results/convergence.png)

The value is close to the analytical solution; the remaining exploitability shows that the learned policy is still approximate. The CSV exports contain action probabilities and training checkpoints.

## Limitations and next steps

This is a tiny zero-sum game, not a general poker engine. Extend the same idea to a larger game such as Leduc Poker and compare full-tree CFR with sampling-based variants.

## Requirements and files

Python 3.9+; the solver uses the standard library. Matplotlib is optional for plotting.

- `kuhn_poker_clean.py` — original project implementation.
- `requirements.txt` — dependency notes.
- `results/` — outputs from the example run, including [terminal output](results/example_run.txt).

The example was rerun using the bundled source. This is a research/educational project; it is not a production system.
