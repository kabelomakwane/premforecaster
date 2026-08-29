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

Project scaffolding only — data sources, models and pipelines are not built yet.
