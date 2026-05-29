# Premier League Match Prediction

Predicting Premier League match outcomes using a Dixon-Coles model
and evaluating against Pinnacle's closing odds.

## Approach

1. Load 3 seasons of Premier League match data (2021-2024) from football-data.co.uk
2. Train Poisson and Dixon-Coles models on 2021/22 and 2022/23
3. Evaluate predictions against Pinnacle closing odds on 2023/24
4. Score using log loss and Brier score on 306 matches
   (excluding matches involving newly-promoted teams not in training data)

## Key findings

- Pinnacle's closing odds are nearly perfectly calibrated 
  (see `calibration.png`)
- Pinnacle averaged 2.93% margin (overround) across the season
- Best model achieved log loss 0.9574 vs market's 0.9264 — 
  the market is approximately 3% sharper

## Files

- `explore.py` — data exploration, calibration analysis
- `train.py` — full training pipeline: Poisson, Dixon-Coles, time-weighted DC, evaluation
- `calibration.png` — Pinnacle's calibration plot
- `E21-22.csv`, `E22-23.csv`, `E23-24.csv` — data from football-data.co.uk

## Methodology notes

- Bookmaker margin removed via proportional normalization
- Train/test split is by season (no look-ahead bias)
- Promoted teams (Luton, Sheffield United) excluded from evaluation
- Optimization via scipy L-BFGS-B with positivity constraints
