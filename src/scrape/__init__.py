"""Data collection: one module per source.

Planned modules:

- ``footballdata``  : match results and closing bookmaker odds (football-data.co.uk)
- ``fbref``         : team and player match stats incl. xG, via ``soccerdata``
- ``understat``     : shot-level xG, via ``soccerdata``
- ``clubelo``       : daily club Elo ratings from the Club Elo API
- ``fpl``           : Fantasy Premier League API (availability, minutes, injuries)
- ``weather``       : Open-Meteo forecasts and history for each stadium
- ``referees``      : referee appointments and card tendencies

House rules for everything in this folder:

1. Raw output goes to ``data/raw/<source>/`` with the download date in the
   filename. We never overwrite a file we downloaded earlier.
2. Rate limits are non-negotiable: FBref no faster than one request per 7
   seconds, Understat no faster than one per 6 seconds, plus random jitter.
3. Always send a descriptive User-Agent, and cache aggressively so we never
   re-download history that cannot have changed.
4. If a source changes its format, raise a clear error. Never write data we are
   not sure about.
"""
