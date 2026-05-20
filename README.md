# World Cup 2026 Match Predictor

Predicts match outcomes (Win / Draw / Loss) for the 2026 FIFA World Cup using
historical competitive international football data and interpretable ML features.

## Project structure

```
worldcup_predictor/
├── data/
│   ├── raw/                  ← place downloaded files here
│   └── processed/            ← pipeline outputs go here
├── src/
│   ├── data_pipeline.py      ← feature engineering & splits
│   └── model.py              ← (next step) logistic regression & XGBoost
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn xgboost
```

## Data download (manual, free)

### 1. Match results — Kaggle
- URL: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- Download `results.csv` → place in `data/raw/results.csv`

### 2. Elo ratings — eloratings.net
- URL: http://eloratings.net/World.tsv
- Download the TSV → place in `data/raw/elo_ratings.tsv`
- Expected columns: date, team, elo  (adjust `load_elo()` if your version differs)

## Run the pipeline

```bash
cd worldcup_predictor
python src/data_pipeline.py
```

This produces:
- `data/processed/matches_featured.csv` — full feature matrix
- `data/processed/train.csv` — ~70% earliest matches
- `data/processed/val.csv`   — ~15% middle window
- `data/processed/test.csv`  — ~15% most recent matches

## Features

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo − Away Elo at match date |
| `home_elo` / `away_elo` | Absolute Elo ratings |
| `legacy_diff` | 20-year avg Elo difference (historical strength) |
| `home_win_rate` / `away_win_rate` | Win rate in last 10 competitive matches |
| `home_gd_pg` / `away_gd_pg` | Goal difference per game in last 10 matches |
| `home_federation_*` / `away_federation_*` | One-hot encoded confederation |
| `neutral` | 1 if played at neutral venue |

## Target variable

`result` from home team perspective: **2 = home win, 1 = draw, 0 = away win**

## Time split

Data is split strictly by time (no leakage):
- Train: 2010 → ~70th percentile match by date
- Val: next 15%
- Test: final 15% (most recent)
