"""Walk-forward back-testing.

The only honest way to test a forecasting model is to replay history in order:
fit on everything up to a date, predict the next round of fixtures, score the
predictions, roll forward. We never shuffle matches across time, because that
lets the model peek at the future and flatters it badly.

Scores we care about:

- log loss (the primary metric - every new feature must improve it),
- ranked probability score (RPS), which rewards being close on ordered outcomes,
- comparison against the de-margined closing odds from football-data.co.uk,
  which is the benchmark to beat.
"""
