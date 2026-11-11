#!/usr/bin/env python3
"""Holdout check: forecast from the first 70% of the reads, score the last 30%.

Scores RULES "## Scoring" exactly: a place where the forecast names the game scores that p, a
place where it does not scores (1 - the place's rows) / (games on the read it did not name there),
floored at ln(0.0001). Reported next to the "nothing changes" baseline on the same holdout.
"""
import hashlib
import json
import math
import os
import random
import sys

import main as M

INPUT = os.environ.get("MAKEFUN_INPUT_DIR") or "/home/ubuntu/makefun/rbx/roblox-up-and-coming-forecast-01-kit/input"
FLOOR = 0.0001


def load():
    with open(os.path.join(INPUT, "history.ndjson"), "rb") as f:
        raw = f.read()
    return raw, M.parse(raw)


def score(rows, reads, places=10):
    return M.log_score(rows, reads, places, FLOOR)


def baseline(states):
    chart = states[-1]["order"]
    rows = []
    for k in range(1, 11):
        if k - 1 < len(chart):
            rows.append({"universe_id": chart[k - 1], "place": k, "p": 0.5 if k > 1 else 0.5})
        if k == 1:
            if len(chart) > 1:
                rows.append({"universe_id": chart[1], "place": 1, "p": 0.3})
            continue
        if k - 2 >= 0:
            rows.append({"universe_id": chart[k - 2], "place": k, "p": 0.15})
        if k < len(chart):
            rows.append({"universe_id": chart[k], "place": k, "p": 0.15})
    return rows


def run():
    raw, states = load()
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    cut = int(len(states) * frac)
    train, test = states[:cut], states[cut:]
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    par = M.estimate(train)
    out = M.build(train, par, 72, random.Random(seed), 10)
    model = score(out["probabilities"], test)
    base = score(baseline(train), test)
    print("reads train/test", len(train), len(test))
    print("model    %.6f" % model)
    print("baseline %.6f" % base)
    print("delta    %.6f" % (model - base))
    return 0 if model > base else 1


if __name__ == "__main__":
    sys.exit(run())
