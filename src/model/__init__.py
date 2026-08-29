"""The forecasting models.

Two models live here:

1. A time-decayed Dixon-Coles Poisson model (via the ``penaltyblog`` package)
   that produces, for each fixture, a matrix of correct-score probabilities.
   Everything else - home/draw/away, over/under, both teams to score - is read
   off that matrix, so the outputs are always mutually consistent.

2. A goalscorer model that splits each team's expected goals across its likely
   starters to give an "anytime scorer" probability per player.

Any machine learning model added later blends with the Dixon-Coles core; it
never replaces it, and only earns its place by improving out-of-sample log loss.
"""
