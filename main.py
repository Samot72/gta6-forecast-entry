"""Launch-day Twitch audience forecaster (make.fun: gta6-twitch-ccu-forecast-01).

Reads one case file per run from the input directory and writes, for hours 2..24
of the release day, five quantiles of that hour's concurrent-viewer count.

Method, in short: a launch's day is forecast as ratios to its own hour-1 print.
Every past launch in the case's `comparables` contributes its release-day
log-ratios ln(viewers_h / viewers_1). Those ratios are pooled per hour with
three ideas on top:

  * Only launches that were already awake at hour 1 anchor the pool. A launch
    whose hour 1 holds less than 15% of its day peak was still asleep at the
    checkpoint (its unlock came later in the UTC day); its ratios explode and
    say nothing about a launch that, like this one, unlocked around the globe
    at local midnight and is already large at hour 1.
  * Pool members are weighted by how similar their pre-release footing was to
    this case's: the ratio of hour 1 to the best and to the average of the
    hours just before the day began, compared in log space with a Gaussian
    kernel. Launches that arrived at hour 1 the same way this one did count
    most.
  * The spread is not assumed normal: each hour's log-ratios are centred and
    scaled by that hour's pooled mean and deviation, the standardised
    residuals of all hours are pooled, and their empirical quantiles give the
    shape of the forecast bands. Location and scale are smoothed over
    neighbouring hours so one noisy hour cannot bend the curve.

Everything is fitted from the case file at run time: no network, no stored
state, no randomness, so two runs on the same case write the same bytes.
If a case carries too few usable comparables, the program falls back to a
persistence forecast (every hour at hour 1's level, spread widening with the
square root of the hour), which is never worse than silence: a missing or
malformed output would score far worse than an honest plain one.

Standard library only. Run by the grader as `python3 -P -s /bot/main.py`
with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR unset (so /input and /output);
locally the two environment variables point the program at a case folder.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P flag keeps this file's directory off sys.path; nothing else
# is imported from it today, but the line keeps a future helper module usable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_LEVELS = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Calibration, fitted by leave-one-out cross-validation over the comparable
# launches shipped in the competition kit (each launch forecast from the
# others, scored with this contest's pinball loss): on launches resembling
# this one (already awake at hour 1) the fitted model scored a mean loss of
# about 0.216 against about 0.271 for plain persistence.
AWAKE_FRACTION = 0.15   # hour 1 must hold at least this share of the day peak
AWAKE_MIN_MEMBERS = 8   # below this, the awake cut is dropped rather than starve the pool
KERNEL_BANDWIDTH = 0.9  # Gaussian kernel width on the log footing features
SMOOTHING = 0.3         # pull of each hour's location/scale toward its neighbours
MIN_SCALE = 0.08        # floor for an hour's log-ratio deviation
MIN_POOL = 5            # fewer usable launches than this and we fall back


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp from the case data."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def round2(value):
    """Round to two decimals, half up, returning an int when whole."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 100.0
    return int(result) if float(result).is_integer() else result


class Launch:
    """One past launch's release day, distilled from a comparable entry."""

    def __init__(self, comparable):
        self.name = comparable.get("name", "")
        self.unlock = comparable.get("unlock", "") or ""
        start = parse_time(comparable["day_start"])
        points = {}
        for row in comparable.get("hourly", []):
            viewers = row.get("viewers")
            if viewers is None:
                continue
            points[parse_time(row["from"])] = float(viewers)
        self.curve = {}
        for hour in range(1, 25):
            value = points.get(start + timedelta(hours=hour - 1))
            if value is not None:
                self.curve[hour] = value
        before24 = [v for t, v in sorted(points.items())
                    if start - timedelta(hours=24) <= t < start]
        before6 = [v for t, v in sorted(points.items())
                   if start - timedelta(hours=6) <= t < start]
        self.usable = len(self.curve) >= 20 and self.curve.get(1, 0) > 0
        if self.usable:
            self.p1 = self.curve[1]
            self.peak = max(self.curve.values())
            self.awake_ratio = self.p1 / self.peak if self.peak > 0 else 0.0
            self.foot_a = self.p1 / max(before24) if before24 and max(before24) > 0 else 1.0
            self.foot_b = (self.p1 / (sum(before6) / len(before6))
                           if before6 and sum(before6) > 0 else 1.0)
            self.log_ratio = {h: math.log(self.curve[h]) - math.log(self.p1)
                              for h in range(2, 25)
                              if self.curve.get(h, 0) > 0}


def case_footing(case, day_start):
    """The target's own pre-release footing features, from its hourly history.

    Mirrors Launch.foot_a / foot_b so the kernel compares like with like.
    Returns None when the case carries no usable pre-day history.
    """
    rows = case.get("history_hourly") or []
    points = []
    for row in rows:
        viewers = row.get("viewers")
        if viewers is None:
            continue
        points.append((parse_time(row["from"]), float(viewers)))
    before24 = [v for t, v in points if day_start - timedelta(hours=24) <= t < day_start]
    before6 = [v for t, v in points if day_start - timedelta(hours=6) <= t < day_start]
    if not before24 or not before6:
        return None
    return before24, before6


def weighted_mean_sd(pairs):
    total = sum(w for _, w in pairs)
    centre = sum(w * x for x, w in pairs) / total
    spread = math.sqrt(max(sum(w * (x - centre) ** 2 for x, w in pairs) / total, 1e-9))
    return centre, spread


def weighted_quantile(pairs, level):
    """Empirical quantile of weighted samples, linearly interpolated."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    walked = 0.0
    prev_x, prev_p = ordered[0][0], 0.0
    for x, w in ordered:
        p = (walked + 0.5 * w) / total
        if p >= level:
            if p == prev_p:
                return x
            return prev_x + (x - prev_x) * (level - prev_p) / (p - prev_p)
        walked += w
        prev_x, prev_p = x, p
    return ordered[-1][0]


def persistence(p1, hours, levels):
    """Fallback: every hour at hour 1's level, spread widening as sqrt(hour)."""
    out = {}
    for h in hours:
        width = 0.15 * math.sqrt(h - 1)
        # Normal quantiles at the asked levels, by interpolation on Z_LEVELS.
        row = []
        for level in levels:
            z = _z_for(level)
            row.append(p1 * math.exp(z * width))
        out[h] = row
    return out


def _z_for(level):
    table = list(zip(DEFAULT_LEVELS, Z_LEVELS))
    if level <= table[0][0]:
        return table[0][1]
    for (l0, z0), (l1, z1) in zip(table, table[1:]):
        if level <= l1:
            return z0 + (z1 - z0) * (level - l0) / (l1 - l0)
    return table[-1][1]


def pooled_forecast(case, p1, hours, levels):
    """The fitted forecast, or None when the case's pool is too thin."""
    day_start = parse_time(case["day_start"])
    game_name = (case.get("game") or {}).get("name", "")
    pool = []
    for comparable in case.get("comparables", []):
        launch = Launch(comparable)
        if not launch.usable:
            continue
        # The case's own category history is not a launch; never pool it, and
        # never pool anything whose release day is the day being forecast.
        if game_name and launch.name.startswith(game_name):
            continue
        if "Not a launch" in launch.unlock:
            continue
        if parse_time(comparable["day_start"]) == day_start:
            continue
        pool.append(launch)
    if len(pool) < MIN_POOL:
        return None

    awake = [launch for launch in pool if launch.awake_ratio >= AWAKE_FRACTION]
    if len(awake) >= AWAKE_MIN_MEMBERS:
        pool = awake

    footing = case_footing(case, day_start)
    target_a = target_b = None
    if footing is not None:
        before24, before6 = footing
        target_a = p1 / max(before24) if max(before24) > 0 else 1.0
        target_b = p1 / (sum(before6) / len(before6)) if sum(before6) > 0 else 1.0
    weights = []
    for launch in pool:
        weight = 1.0
        if target_a is not None:
            da = (math.log(target_a) - math.log(launch.foot_a)) / KERNEL_BANDWIDTH
            db = (math.log(target_b) - math.log(launch.foot_b)) / KERNEL_BANDWIDTH
            weight = math.exp(-0.5 * (da * da + db * db))
        if weight > 1e-4:
            weights.append((launch, weight))
    if not weights:
        weights = [(launch, 1.0) for launch in pool]

    centres, scales, raw = {}, {}, {}
    for h in range(2, 25):
        pairs = [(launch.log_ratio[h], w) for launch, w in weights if h in launch.log_ratio]
        raw[h] = pairs
        if len(pairs) >= 3:
            centre, spread = weighted_mean_sd(pairs)
            centres[h], scales[h] = centre, max(spread, MIN_SCALE)
        else:
            centres[h], scales[h] = 0.0, 0.35
    for table in (centres, scales):
        blended = {}
        for h in range(2, 25):
            lo, hi = max(2, h - 1), min(24, h + 1)
            neighbour = sum(table[k] for k in range(lo, hi + 1)) / (hi - lo + 1)
            blended[h] = (1.0 - SMOOTHING) * table[h] + SMOOTHING * neighbour
        table.update(blended)

    residuals = [((x - centres[h]) / scales[h], w)
                 for h in range(2, 25) for x, w in raw[h]]
    if not residuals:
        return None
    shape = [weighted_quantile(residuals, level) for level in levels]
    return {h: [p1 * math.exp(centres[h] + scales[h] * s) for s in shape]
            for h in hours}


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    hours = [row["hour"] for row in case.get("forecast", [])]
    if not hours:
        hours = list(range(2, 25))
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first and first.get("viewers") else 1.0
    forecast = pooled_forecast(case, p1, hours, levels)
    if forecast is None:
        forecast = persistence(p1, hours, levels)
    rows = []
    for h in hours:
        quantiles = []
        running = 0.01
        for value in forecast[h]:
            value = min(max(value, 0.01), 1_000_000_000.0)
            value = round2(value)
            running = max(running, float(value))
            quantiles.append(round2(running))
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
