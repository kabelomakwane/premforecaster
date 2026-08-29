"""Team and player strength ratings.

This is the layer between raw data and the forecasting model. It turns match
histories into the numbers the model actually consumes, for example:

- time-decayed attack and defence strengths built from xG rather than goals,
- Club Elo ratings aligned onto our canonical team names,
- player-level shot and minutes shares used by the goalscorer model.

Nothing here scrapes. It reads cleaned tables from ``data/processed/``.
"""
