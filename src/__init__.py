"""Premier League forecasting model.

This package holds all the code for the forecaster. It is split up so that each
part has one job:

- ``scrape``   : downloads data from the outside world (one module per source).
- ``ratings``  : turns raw data into team and player strength numbers.
- ``model``    : the Dixon-Coles goal model and the goalscorer model.
- ``backtest`` : walk-forward testing, so we know whether the model is any good.
- ``export``   : writes the finished predictions out to Google Sheets.

Nothing in here talks to the internet on import. Scraping only happens when a
pipeline script explicitly asks for it.
"""

__version__ = "0.1.0"
