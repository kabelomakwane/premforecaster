# premforecaster

A personal, non-commercial Premier League forecasting model. It predicts, for
each upcoming fixture:

1. **Match result** — home / draw / away probabilities
2. **Score lines** — a correct-score probability matrix, plus over/under and
   both-teams-to-score
3. **Goal scorers** — an anytime-scorer probability for each player

The modelling core is a time-decayed Dixon-Coles Poisson model (via
`penaltyblog`) built on xG-based team strengths. See [CLAUDE.md](CLAUDE.md) for
the full design decisions — read that before changing anything.

## Getting set up

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Layout

```
data/
  raw/          untouched downloads, one folder per source (not in git)
  processed/    cleaned, joined tables (not in git)
  lookups/      hand-maintained reference data (IS in git)
src/
  scrape/       one module per data source
  ratings/      team and player strength calculations
  model/        Dixon-Coles core and the goalscorer model
  backtest/     walk-forward evaluation
  export/       Google Sheets writer
pipelines/      scripts that chain the above together
tests/          pytest tests
```

## Building the results table

```bash
python -m pipelines.build_results
```

Downloads every Premier League season CSV from football-data.co.uk (2014/15 to
now) into `data/raw/footballdata/`, then writes one tidy row per match to
`data/processed/results.parquet` — canonical team names, UTC kick-off times,
best available closing odds and de-margined market probabilities.

Neither directory is in git, so run this once after cloning. Add `--no-download`
to rebuild from the raw files already on disk.

## Things worth knowing

- **All joins go through `data/lookups/team_names.csv`.** Every source spells
  club names differently. Never join on a raw source name.
- **`data/lookups/NOTES.md` lists the names that still need verifying.** Some
  were confirmed against the live APIs; some are from memory and are flagged.
- **Scraping limits are non-negotiable:** FBref no faster than 1 request per
  7 seconds, Understat 1 per 6 seconds, always with a descriptive User-Agent.
- **All datetimes are UTC internally.** They are converted to UK time only in
  the export layer.
- **Back-testing is walk-forward only.** Never shuffle matches across time.

## Status

- ✅ Repo scaffolding, lookup tables
- ✅ football-data.co.uk results and closing odds → `results.parquet`
- ⬜ FBref / Understat xG, Club Elo, FPL, weather, referees
- ⬜ Dixon-Coles model, goalscorer model, back-test, Google Sheets export
