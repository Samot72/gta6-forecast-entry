# GTA VI Twitch CCU Forecaster

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Samot72/gta6-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Samot72/gta6-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

An entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each of
hours 2–24 of its release day, 2026-11-19 (UTC), given everything recorded up
to the end of hour 1 — as five quantiles per hour, scored by pinball loss on
the log scale.

## How it works

`main.py` (Python 3, standard library only, deterministic, no network) reads
the case file in the input directory and:

1. Takes hour 1's print as the anchor. Every forecast is a set of ratios to
   that print, hour by hour.
2. Builds the ratio pool from the case's own `comparables`: each past
   launch's release-day log-ratios ln(viewers_h / viewers_1).
3. Keeps only launches that were already *awake* at hour 1 (hour 1 holding at
   least 15% of their day peak). A launch still asleep at the checkpoint —
   its unlock came later in the UTC day — has explosive ratios and nothing to
   say about a rolling-midnight launch like this one.
4. Weights pool members by pre-release footing: how the target's hour 1
   compares with its own best and average hours just before the day began,
   matched against each member's same two features with a Gaussian kernel.
5. Centres and scales each hour's pooled log-ratios (smoothed over
   neighbouring hours), pools the standardised residuals of all hours, and
   reads the five quantiles off their empirical distribution — the spread is
   measured, not assumed normal.

If a case carries too few usable comparables, the program falls back to a
persistence forecast (every hour at hour 1's level, spread widening with the
square root of the hour).

## Calibration

The constants (awake cut, kernel width, smoothing, scale floor) were fitted
by leave-one-out cross-validation over the comparable launches shipped in
the competition kit — each launch forecast from the others and scored with
the contest's own pinball loss:

| launches resembling this one (awake at hour 1) | mean loss |
| --- | --- |
| this model | **0.216** |
| persistence baseline (the house starter) | 0.271 |

Across all 37 past launches the model scores 0.856 against the baseline's
0.930, and its 5%–95% band covered 87.7% of realised hours on the
resembling subset (an honest band covers about 90%).

## Running it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<case folder> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes one `<id>.out` per `<id>.in` found in the case folder.

## License

MIT — see `LICENSE`.
