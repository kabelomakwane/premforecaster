"""Publishing predictions.

Takes the finished forecasts and writes them to a Google Sheet using ``gspread``
with a service account, so the numbers can be read on a phone without running
any code.

This is the only layer allowed to convert times out of UTC: everything upstream
works in UTC, and times are shown in UK local time only at the point of display.
"""
