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

## Building the xG tables

```bash
python -m pipelines.build_xg                 # Understat + FBref + reconciliation
python -m pipelines.build_xg --skip-fbref    # Understat only, much quicker
```

Writes `team_match_xg.parquet`, `player_season.parquet`,
`understat_player_match.parquet`, `shots.parquet`, and (when FBref is
reachable) `fbref_team_match.parquet` and `fbref_player_match.parquet`. It
finishes by writing `reconciliation_report.csv`, which lists any match that is
in `results.parquet` but missing from an xG table, or the other way round.

Both scrapers are **cache-first and resumable**. Every downloaded page is cached
under `data/raw/<source>/`, and each season is staged to parquet and recorded in
a checkpoint as soon as it parses — so an interrupted run resumes where it
stopped, and a rerun makes **zero** network requests (measured: a full rebuild of
the team xG table over a warm checkpoint issues none). Delete the checkpoint and
it still serves all the data from the page cache, at the cost of one
cookie-priming request per season.

**The per-match tables are slow.** Shots and player match logs need one request
per match, so at the six-second Understat minimum a single season takes about 40
minutes. They default to the last three completed seasons plus the current one.
To go further back, pass `--shot-seasons`/`--player-match-seasons` and leave it
running; it is resumable, so you can stop and restart it freely.

**FBref needs a browser.** It sits behind Cloudflare, which rejects ordinary HTTP
clients, so `soccerdata` drives a real Chrome/Chromium. If it cannot find one,
set `PREMFORECASTER_BROWSER` to the browser binary. If FBref is unreachable the
pipeline logs it and carries on — Understat is what supplies the xG the model
actually depends on.

## Building the context tables

```bash
python -m pipelines.build_context                # Elo, FPL, weather, referees
python -m pipelines.build_context --only fpl     # just the availability snapshot
```

Writes `elo_history.parquet`, `fpl_players.parquet`, `fpl_fixtures.parquet`,
`match_weather.parquet` and `referee_profiles.parquet`. Sources are independent,
so one being unreachable does not stop the others.

**Run the FPL step often.** It snapshots who is injured or doubtful *right now*,
and that is the only way the history is ever recorded — the API cannot tell you
who was injured last month, so a snapshot you don't take is lost for good. The
players table is append-only; the others can be rebuilt any time.

Weather batches by stadium and year (every club plays 19 home games at one
ground), so the whole 2014/15-onwards history costs about 300 requests rather
than 4,570, and reruns are served from cache.

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
- ✅ Understat xG (team, player, shot level) + reconciliation report
- ✅ FPL availability snapshots, match weather (100% of matches), referee profiles
- ⚠️ FBref and Club Elo scrapers written and tested, but both blocked by the
  network in the environment they were built in — see `data/lookups/NOTES.md`
  for what to run to verify them
- ⬜ Dixon-Coles model, goalscorer model, back-test, Google Sheets export
