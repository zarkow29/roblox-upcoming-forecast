# roblox-upcoming-forecast

A forecast of Roblox's Up-and-Coming chart, place by place, for the 72 hours after the entries
close. `main.py` (Python 3.13, standard library only, no network) reads the chart history from
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` to `$MAKEFUN_OUTPUT_DIR`
(default `/output`).

## The model

The chart is a rank process with birth and death, not a fixed order, so the model is a transition
process rather than a fixed order:

1. **States.** Only `ok` reads are kept, one state per read, games in rank order with their player
   counts. Everything is read from `task.json`: the places, the window and its slots.
2. **Kernels.** Every ordered pair of reads is a sample of the gap between it: a game at place `k`
   in the earlier read lands at `k'` in the later one, or off the ladder. Counts are pooled over
   neighbouring source places, shrunk towards a trend-agnostic row and Laplace-smoothed, and split
   by the game's trend (its players against its own recent mean, plus its rank climb) — so rising
   games climb and sinking ones slide, and long gaps come from real long-gap pairs instead of a
   constant-hazard extrapolation.
3. **Window weighting.** Each horizon band is weighted by how many hours of the window sit at it,
   so the one static forecast is the average over the slots rather than a single step. Long
   horizons also key off the game's recent strength band, not its instantaneous place.
4. **Inflow.** What falls off the ladder is given to the games just below the top 10, weighted by
   proximity and trend — the pool the chart's newcomers actually come from. Games published after
   the close cannot be named; the unnamed share of each place is left for them.
5. **Answer.** Per place, the games are ranked by that occupancy, pulled towards the place they
   already hold (the chart keeps most of its places hour to hour, and the ladder alone over-shoots
   it) and nudged by trend, then cut to a fixed number of names. The rest of the place is left
   unnamed: a place's unnamed share is spread over every game the read has that the row does not
   name, so naming a game the model barely believes is worse than leaving it. Every probability is
   at or above `0.0002`, each place sums to at most `0.7`, and the top 10 is the chart's own top 10
   in its own order — the best membership and agreement guess there is.

Where a read carries Roblox's `created` time for a game, its age shifts it up while it is young
and down as it nears the 28-day limit. A history without those times (the public archive carries
none) forecasts exactly as it would with the term switched off.

How sharp to be is not a constant: the program scores a few settings on inner holdouts of the
history it was given and uses the winner, so it adapts to how much history there is.

Randomness is seeded from the sha256 of the history, so the same history gives a byte-identical
file.

## Checking it

`eval.py` holds out the last 30% of the reads (`python3 eval.py 0.7`), forecasts from the earlier
part and scores RULES "## Scoring" against the held-out reads, printed next to the "nothing
changes" baseline on the same holdout. It exits non-zero when the model does not beat the baseline.

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=/tmp/out python3 main.py
    python3 eval.py 0.7

## Licence

MIT, see LICENSE.
