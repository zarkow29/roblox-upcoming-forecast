#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10, place by place, for 72 hours.

The chart is a rank process with birth and death, not a fixed order: games climb, fall out of the
top 10, and new games (published within 28 days) join as old ones age off. So the model is

  * transition kernels per horizon band, trend class and source place, measured from ordered pairs
    of reads — where a game at place k actually landed after that many hours, Laplace-smoothed and
    pooled over neighbouring places — so long gaps come from real long-gap pairs,
  * each band weighted by how many hours of the window sit at it, giving the average over the
    window in one step instead of an extrapolated single step,
  * vertices beyond the top 10 kept as a ladder (the pool the chart's newcomers come from) with
    their mass spilled onto the games just below it,
  * a per-place ranking that pulls a game towards the place it already holds and nudges it by its
    trend, with the games published recently lifted and the nearly-28-day-old pushed down when the
    reads carry created times,
  * each place cut to its sharpest names and the rest left unnamed, for the games the history
    cannot know about.

Stdlib only, no network, byte-identical for the same history (randomness is seeded from its
sha256).
"""
import hashlib
import json
import math
import os
import random
import datetime as dt

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

BAND_TOP = 10           # places that are scored
BAND = 20               # ranks the kernels keep; 10..20 are the ladder into the top 10
FLOOR = 0.0002          # never below this: a miss costs ln(0.0002), not the ln(0.0001) floor
BETA = 0.4
ALPHA = 0.4
LAM = 0.1
AGE_B = 0.35


def read_inputs():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "rb") as f:
        raw = f.read()
    return task, raw


def parse(raw):
    """Keep only ok reads, one state per read: ordered universe ids and their players."""
    states = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("ok"):
            continue
        rows = rec.get("up_and_coming") or []
        rows = [r for r in rows if isinstance(r.get("universe_id"), int) and isinstance(r.get("rank"), int)]
        if len(rows) < BAND_TOP:
            continue
        rows.sort(key=lambda r: r["rank"])
        order = []
        players = {}
        names = {}
        created = {}
        details = rec.get("details") or {}
        for r in rows:
            u = r["universe_id"]
            order.append(u)
            players[u] = r.get("playing") or 0
            names[u] = r.get("name") or ""
            made = (details.get(str(u)) or {}).get("created") if isinstance(details, dict) else None
            if isinstance(made, str):
                try:
                    created[u] = dt.datetime.fromisoformat(made.replace("Z", "+00:00"))
                except ValueError:
                    pass
        at = dt.datetime.fromisoformat(rec["at"].replace("Z", "+00:00"))
        states.append({"at": at, "order": order, "players": players, "names": names,
                       "created": created})
    states.sort(key=lambda s: s["at"])
    return states


def age_days(states, u):
    """How old the game is, from Roblox's own created time where a read carries it."""
    for s in reversed(states):
        made = s["created"].get(u)
        if made is not None:
            return clamp((s["at"] - made).total_seconds() / 86400.0, 0.0, 40.0)
    return None


def hours_between(a, b):
    return max((b - a).total_seconds() / 3600.0, 0.25)


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def trend_of(states, i, u):
    """How fast u is going up: its players against its own mean, plus its rank climb."""
    base = []
    for j in range(i - 1, max(-1, i - 4), -1):
        v = states[j]["players"].get(u)
        if v:
            base.append(v)
    cur = states[i]["players"].get(u)
    players = 0.0
    if base and cur:
        m = sum(base) / len(base)
        if m:
            players = clamp((cur - m) / m, -0.9, 3.0)
    climb = 0.0
    for j in range(i - 1, max(-1, i - 3), -1):
        before = states[j - 1]["order"] if j else []
        here = states[j]["order"]
        try:
            climb += (before.index(u) + 1) - (here.index(u) + 1)
        except ValueError:
            continue
    climb = clamp(climb / 4.0, -1.5, 1.5)
    return clamp(players + climb, -2.0, 4.0)


def trend_class(tr):
    if tr > 0.10:
        return "rise"
    if tr < -0.10:
        return "fall"
    return "flat"


TOP_OUTCOMES = ("stay", "up", "down", "out")
BAND_OUTCOMES = ("stay", "to10", "out15")
K_UP = ((1, 0.90), (2, 0.08), (3, 0.02))
K_DOWN = ((1, 0.80), (2, 0.15), (3, 0.05))
PRIOR = 0.45


HORIZONS = ((0.5, 8.0), (8.0, 20.0), (20.0, 36.0), (36.0, 60.0), (60.0, 120.0))
SRC_DEPTH = 60          # source ranks the kernel is estimated for
STRENGTH = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 60))
AGE_C = 0.0
AGE_REF = 240.0
STRENGTH_FROM = 1       # from this horizon band on, the kernel uses strength, not the place


def window_weights(window_hours):
    """How many hours of the window sit at each horizon, so the average matches the scoring."""
    w = []
    for lo, hi in HORIZONS:
        span = min(hi, window_hours) - lo
        w.append(max(span, 0.0))
    tot = sum(w) or 1.0
    return [x / tot for x in w]


def estimate(states, window_hours=72.0):
    """Transition kernels per horizon band, trend class and source place.

    Each ordered pair of reads is a sample of the gap between it: a game at place k in the earlier
    read lands at k' (or out) in the later one. Counts are pooled over neighbouring source places,
    Laplace-smoothed and normalised, so long gaps come from real long-gap pairs instead of a
    constant-hazard extrapolation.
    """
    counts = {}
    trend_c = {c: {"up": 0.0, "down": 0.0} for c in ("rise", "flat", "fall")}
    birth_at = {}
    total_hours = 0.0
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            gap = hours_between(states[i]["at"], states[j]["at"])
            if gap < 0.5:
                continue
            b = None
            for bi, (lo, hi) in enumerate(HORIZONS):
                if lo <= gap < hi:
                    b = bi
                    break
            if b is None:
                continue
            total_hours += gap
            dest = {u: k for k, u in enumerate(states[j]["order"], 1)}
            for place, u in enumerate(states[i]["order"][:SRC_DEPTH], 1):
                cls = trend_class(trend_of(states, i, u))
                k = dest.get(u)
                cell = counts.setdefault((b, cls, place), {})
                key = k if (k is not None and k <= BAND) else 0
                cell[key] = cell.get(key, 0.0) + 1.0
                if place <= BAND:
                    if k is not None and k < place:
                        trend_c[cls]["up"] += 1.0
                    elif k is not None and k > place:
                        trend_c[cls]["down"] += 1.0
    # where newcomers land, for the inflow distribution
    for i in range(len(states) - 1):
        prev = set(states[i]["order"][:BAND])
        for rank, u in enumerate(states[i + 1]["order"][:BAND], 1):
            if u not in prev:
                birth_at[rank] = birth_at.get(rank, 0.0) + 1.0

    kernel = {}
    skernel = {}
    for b in range(len(HORIZONS)):
        for cls in ("rise", "flat", "fall"):
            for band in range(len(STRENGTH)):
                lo, hi = STRENGTH[band]
                pooled = {}
                for (bb, cc, place), cell in counts.items():
                    if bb != b or cc != cls:
                        continue
                    if not (lo <= place <= hi):
                        continue
                    for key, c in cell.items():
                        pooled[key] = pooled.get(key, 0.0) + c
                for key in range(0, BAND + 1):
                    pooled[key] = pooled.get(key, 0.0) + PRIOR
                tot = sum(pooled.values())
                skernel[(b, cls, band)] = {key: v / tot for key, v in pooled.items()}
        for place in range(1, SRC_DEPTH + 1):
            # class-agnostic row: the backoff every trend class shrinks towards
            base = {}
            for cls in ("rise", "flat", "fall"):
                for j in range(max(place - 1, 1), min(place + 2, SRC_DEPTH) + 1):
                    w = 1.0 if j == place else 0.35
                    for key, c in counts.get((b, cls, j), {}).items():
                        base[key] = base.get(key, 0.0) + w * c
            for cls in ("rise", "flat", "fall"):
                pooled = dict(base)
                for j in range(max(place - 1, 1), min(place + 2, SRC_DEPTH) + 1):
                    w = 0.25 if j != place else 0.0
                    for key, c in counts.get((b, cls, j), {}).items():
                        pooled[key] = pooled.get(key, 0.0) + w * c
                for key in range(0, BAND + 1):
                    pooled[key] = pooled.get(key, 0.0) + PRIOR
                tot = sum(pooled.values())
                kernel[(b, cls, place)] = {key: v / tot for key, v in pooled.items()}

    mult = {}
    for c, base in (("rise", 2.8), ("flat", 1.0), ("fall", 0.45)):
        share = (trend_c[c]["up"] + PRIOR) / (trend_c[c]["down"] + PRIOR)
        mult[c] = {"up": base * (share ** 0.5), "down": base / (share ** 0.5)}
    norm = (sum(mult[c]["up"] for c in mult) / len(mult)) / (sum(mult[c]["down"] for c in mult) / len(mult))
    for c in mult:
        mult[c]["up"] /= norm

    w = {k: birth_at.get(k, 0.0) + 0.5 for k in range(1, BAND + 1)}
    tot = sum(w.values())
    return {"kernel": kernel, "skernel": skernel, "birth_dist": {k: v / tot for k, v in w.items()},
            "trend": mult, "weights": window_weights(window_hours),
            "mean_dt": total_hours / max(len(counts), 1)}

OUT = 0


def pool_weight(rank, tr):
    """How likely a game below the ladder is to be the one that climbs in: nearer and rising."""
    w = 1.0 / (1.0 + max(rank - BAND, 0) ** 0.6)
    return w * (1.0 + 1.6 * max(tr, 0.0))


def roll(states, par, slots):
    """Occupancy of every place by every game, averaged over the window's slots.

    A game's place vector is the window-weighted mixture of the kernels for its place and trend,
    so the average already spans the whole window instead of a single step.
    """
    last = states[-1]
    order = last["order"]
    strength = {}
    for u in order:
        seen = []
        for s_ in reversed(states):
            if u in s_["order"]:
                seen.append(s_["order"].index(u) + 1)
                if len(seen) == 3:
                    break
        strength[u] = sum(seen) / len(seen)
    age_h = {}
    for u in order:
        age_h[u] = (last["at"] - next((s_["at"] for s_ in states if u in s_["order"]), last["at"])).total_seconds() / 3600.0
    occ = {}
    spill = [0.0] * (BAND_TOP + 1)
    for rank, u in enumerate(order, 1):
        if rank > SRC_DEPTH:
            break
        cls = trend_class(trend_of(states, len(states) - 1, u))
        sband = 0
        for bi, (lo, hi) in enumerate(STRENGTH):
            if lo <= strength[u] <= hi:
                sband = bi
                break
        vec = [0.0] * (BAND_TOP + 1)
        for b, w in enumerate(par["weights"]):
            if w <= 0:
                continue
            row = par["kernel"][(b, cls, rank)] if b < STRENGTH_FROM else par["skernel"][(b, cls, sband)]
            for k in range(1, BAND_TOP + 1):
                vec[k] += w * row.get(k, 0.0)
            spill_here = w * row.get(0, 0.0)
            for k in range(1, BAND_TOP + 1):
                spill[k] += spill_here / BAND_TOP
        a = clamp((AGE_REF - age_h.get(u, 0.0)) / AGE_REF, -1.0, 1.0)
        if AGE_C:
            for k in range(1, BAND_TOP + 1):
                vec[k] *= math.exp(AGE_C * a * (k - 5.5) / 5.0)
            tot_v = sum(vec[1:]) or 1.0
            for k in range(1, BAND_TOP + 1):
                vec[k] /= tot_v
        occ[u] = vec

    # what fell out of the ladder is given to the games just below it
    pool = {}
    for rank, u in enumerate(order, 1):
        if rank <= BAND:
            continue
        pool[u] = pool_weight(rank, trend_of(states, len(states) - 1, u))
    tot = sum(pool.values()) or 1.0
    for k in range(1, BAND_TOP + 1):
        share = spill[k]
        if share <= 0:
            continue
        for u, w in pool.items():
            occ.setdefault(u, [0.0] * (BAND_TOP + 1))[k] += share * (w / tot)
    # many games each carry a distribution over the places, so a place's mass sums far above one:
    # scale each place so it reads as a probability over games again
    for k in range(1, BAND_TOP + 1):
        s = sum(v[k] for v in occ.values())
        if s > 0:
            for v in occ.values():
                v[k] /= s
    return occ


def log_score(rows, reads, places=10, floor=0.0001):
    """RULES "## Scoring": mean over reads of the 10 per-place ln p, each floored."""
    byplace = {}
    for r in rows:
        byplace.setdefault(r["place"], {})[r["universe_id"]] = r["p"]
    total = 0.0
    for st in reads:
        chart = st["order"][:places]
        for k in range(1, places + 1):
            rows_k = byplace.get(k, {})
            u = chart[k - 1] if len(chart) >= k else None
            if u is not None and u in rows_k:
                p = rows_k[u]
            else:
                rest = [x for x in chart if x not in rows_k]
                p = (1.0 - sum(rows_k.values())) / len(rest) if rest else 0.0
            total += math.log(max(p, floor))
    return total / len(reads) if reads else float("-inf")


SHAPES = ((22, 0.88),)


def choose_shape(states, seed, places):
    """Pick how sharp to be: score each setting on two inner holdouts and take the best mean."""
    best, best_score = SHAPES[0], float("-inf")
    for shape in SHAPES:
        scores = []
        for frac in (0.6, 0.78):
            cut = int(len(states) * frac)
            train, test = states[:cut], states[cut:]
            if len(test) < 3 or len(train) < 8:
                continue
            out = _build(train, estimate(train), 72, random.Random(seed), places, shape)
            scores.append(log_score(out["probabilities"], test))
        if not scores:
            return SHAPES[0]
        s = sum(scores) / len(scores)
        if s > best_score + 1e-9:
            best, best_score = shape, s
    return best


def build(states, par, slots, rng, places):
    shape = choose_shape(states, int.from_bytes(hashlib.sha256(b"shape").digest()[:4], "big"), places)
    print("shape", shape)
    return _build(states, par, slots, rng, places, shape)


def _build(states, par, slots, rng, places, shape):
    max_rows, named_share = shape
    occ = roll(states, par, slots)
    order = states[-1]["order"]
    rank_of = {u: i + 1 for i, u in enumerate(order)}
    rows = []
    weights = {}
    for k in range(1, places + 1):
        mass = {}
        for u in order[:SRC_DEPTH]:
            v = occ.get(u, [0.0] * (BAND_TOP + 1))[k] + 1e-4
            # the kernel's occupancy, pulled towards the game's own place: the chart keeps most
            # of its places hour to hour, and the ladder alone over-shoots it
            w = (v ** ALPHA) * math.exp(-LAM * abs(rank_of[u] - k)) \
                * math.exp(BETA * trend_of(states, len(states) - 1, u))
            made = age_days(states, u)
            if made is not None:
                # a game published days ago has the 28-day rule still ahead of it; an old one is
                # close to ageing out. Unknown ages are left neutral, so a history without
                # created times forecasts exactly as it would without this term.
                w *= math.exp(AGE_B * (14.0 - made) / 14.0)
            if w > 0:
                mass[u] = w
                weights[(k, u)] = w
        keep = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:max_rows]
        tot = sum(p for _, p in keep) or 1.0
        # named_share of the place is named, the rest stays unnamed for the games the roll
        # cannot know about (ones published after the close, or deep down the chart)
        for u, p in keep:
            rows.append({"universe_id": u, "place": k, "p": round(max(named_share * p / tot, FLOOR), 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    score = {}
    for (k, u), v in weights.items():
        score[u] = score.get(u, 0.0) + v * (places + 1 - k)
    # the chart's own top 10, in its own order: the best membership and agreement there is
    top10 = order[:places]
    return {"probabilities": rows, "top10": top10}


def flat_forecast(states, places):
    """Last resort when the history is too thin to model: even spread over the chart we can see."""
    if not states:
        return {"probabilities": [], "top10": []}
    order = states[-1]["order"]
    rows = []
    for k in range(1, places + 1):
        band = order[max(k - 4, 0):k + 4] or order[:1]
        share = 0.6 / max(len(band), 1)
        for u in band:
            rows.append({"universe_id": u, "place": k, "p": round(share, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    return {"probabilities": rows, "top10": order[:places]}


def main():
    task, raw = read_inputs()
    states = parse(raw)
    slots = int(task["window"]["slots"])
    places = int(task.get("places", 10))
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    rng = random.Random(seed)
    if len(states) >= 3:
        par = estimate(states, slots * float(task["window"].get("slot_minutes", 60)) / 60.0)
        out = build(states, par, slots, rng, places)
        note = "mean_gap %.2f" % par["mean_dt"]
    else:
        out = flat_forecast(states, places)
        note = "fallback"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        f.write("\n")
    print("rows", len(out["probabilities"]), "top10", len(out["top10"]), note)


if __name__ == "__main__":
    main()



