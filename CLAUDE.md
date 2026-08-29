You are building a Premier League football forecasting model with me. Read this context before every task.

PROJECT GOAL
Predict three things for upcoming Premier League fixtures:
1. Match Result (Home/Draw/Away probabilities)
2. Score Lines (correct score probability matrix, plus over/under and BTTS)
3. Goal Scorers (anytime scorer probability per player)

CORE DESIGN DECISIONS (do not deviate without asking me)
- Language: Python 3.11+. Package management with a requirements.txt or pyproject.toml.
- Modelling core: time-decayed Dixon-Coles Poisson via the `penaltyblog` package, upgraded with xG-based team strengths. ML (XGBoost) may be added later as a blend, never as a replacement, and only if it improves out-of-sample log loss.
- Data sources (free only): football-data.co.uk (results + closing odds), FBref via `soccerdata` (xG and advanced stats), Understat via `soccerdata` (shot-level xG), Club Elo API, Fantasy Premier League API (player availability, minutes signals, injuries), Open-Meteo API (weather), referee data scraped politely from public sources.
- Storage: local parquet/CSV files in a `data/` directory as the store of record for now, with a thin export layer that writes current predictions to Google Sheets via gspread + a service account. BigQuery sandbox is an optional later migration, not part of the initial build.
- Automation: GitHub Actions on a cron schedule.
- Evaluation: walk-forward back-testing only (never shuffle across time). Primary metrics: log loss and RPS. Benchmark: de-margined closing odds from football-data.co.uk. Every new feature must earn its place by improving out-of-sample log loss.

SCRAPING ETHICS (non-negotiable)
- FBref: maximum 1 request every 7 seconds. Cache everything. Never refetch unchanged historical data.
- Understat: 1 request every 6 seconds minimum.
- Always set a descriptive User-Agent. Add random jitter to request intervals.
- This project is strictly personal and non-commercial. Never build anything that redistributes raw scraped data.

REPO STRUCTURE (create and maintain this)
pl-forecast/
  CLAUDE.md
  requirements.txt
  data/
    raw/          # untouched scraped data, one subfolder per source
    processed/    # cleaned, joined tables
    lookups/      # team name mapping, stadium coords, referees, penalty takers
  src/
    scrape/       # one module per data source
    ratings/      # team and player rating calculations
    model/        # Dixon-Coles core, scorer model
    backtest/     # walk-forward evaluation
    export/       # Google Sheets writer
  pipelines/      # orchestration scripts that chain the above
  tests/          # pytest tests
  .github/workflows/

CONVENTIONS
- Every scraper writes raw output to data/raw/<source>/ with a date-stamped filename and never overwrites prior raw files.
- Every module has a docstring explaining what it does in plain English, because a non-engineer maintains this project.
- Defensive parsing everywhere: if a source changes format, fail loudly with a clear error message, never silently write bad data.
- Team names differ across sources. ALL joins go through data/lookups/team_names.csv which maps every source's naming to one canonical name. Never join on raw source names.
- All datetimes in UTC internally. Display in UK time only at the export layer.
- Write pytest tests for every non-trivial function, especially date logic, team name mapping, and probability calculations (probabilities must sum to 1 within tolerance).