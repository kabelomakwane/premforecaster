"""Tests for the football-data.co.uk scraper and processor.

These tests never touch the network. They build small CSV files in the same
shape as the real ones - including the awkward bits, like the two different date
formats, the missing Time column in older seasons, and Pinnacle odds
disappearing partway through a season - and check the processing copes.

The tests that need the real downloaded data live in test_results_parquet.py.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.lookups import UnknownTeamError
from src.scrape import footballdata as fd

# ---------------------------------------------------------------------------
# Helpers for building fake raw CSVs
# ---------------------------------------------------------------------------

MODERN_HEADER = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Referee",
    "PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA", "AvgCH", "AvgCD", "AvgCA",
]
LEGACY_HEADER = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Referee",
    "PSCH", "PSCD", "PSCA",
]


def modern_row(date_text, time_text, home, away, hg, ag, ftr, referee="M Oliver", **odds):
    """One row in the post-2019/20 column layout. Missing odds default to blank."""
    prices = {column: "" for column in MODERN_HEADER[9:]}
    prices.update({key: str(value) for key, value in odds.items()})
    return [
        "E0", date_text, time_text, home, away, str(hg), str(ag), ftr, referee,
        *[prices[column] for column in MODERN_HEADER[9:]],
    ]


def write_csv(path, header, rows, *, bom=False, trailing_blank=False):
    lines = [",".join(header)] + [",".join(row) for row in rows]
    if trailing_blank:
        lines.append("," * (len(header) - 1))
    text = "\n".join(lines) + "\n"
    path.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Season naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_year", "label", "code"),
    [(2014, "2014-15", "1415"), (1999, "1999-00", "9900"), (2026, "2026-27", "2627")],
)
def test_season_naming(start_year, label, code):
    assert fd.season_label(start_year) == label
    assert fd.season_code(start_year) == code
    assert fd.season_start_year(label) == start_year


def test_season_start_year_rejects_rubbish():
    with pytest.raises(ValueError):
        fd.season_start_year("last season")
    with pytest.raises(ValueError, match="Malformed"):
        fd.season_start_year("2014-99")


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 8, 29), 2026),  # mid-season
        (date(2026, 7, 1), 2026),  # July: the new season has begun
        (date(2026, 6, 30), 2025),  # June: still last season
        (date(2027, 5, 24), 2026),  # final day of the season
        (date(2027, 1, 2), 2026),  # after New Year, still the season that started in 2026
    ],
)
def test_current_season_start_year(today, expected):
    assert fd.current_season_start_year(today) == expected


def test_season_start_years_covers_the_whole_horizon():
    years = fd.season_start_years(today=date(2026, 8, 29))
    assert years[0] == 2014
    assert years[-1] == 2026
    assert len(years) == 13


def test_season_start_years_rejects_seasons_before_our_horizon():
    with pytest.raises(ValueError, match="before the first season"):
        fd.season_start_years(through=2010)


def test_raw_filename_is_date_stamped():
    name = fd.raw_filename(2014, date(2026, 8, 29))
    assert name == "E0_2014-15_downloaded-2026-08-29.csv"


def test_season_url():
    assert fd.season_url(2014).endswith("/mmz4281/1415/E0.csv")


# ---------------------------------------------------------------------------
# Date and kick-off parsing
# ---------------------------------------------------------------------------


def test_parse_dates_handles_both_year_formats():
    """2014/15 and 2016/17 use two-digit years, the seasons around them four."""
    parsed = fd.parse_dates(pd.Series(["16/08/14", "09/08/2019", "01/01/17"]))
    assert list(parsed) == [
        pd.Timestamp("2014-08-16"),
        pd.Timestamp("2019-08-09"),
        pd.Timestamp("2017-01-01"),
    ]


def test_parse_dates_is_day_first_not_month_first():
    """06/08 must be the 6th of August, not the 8th of June."""
    assert fd.parse_dates(pd.Series(["06/08/2022"]))[0] == pd.Timestamp("2022-08-06")


def test_parse_dates_fails_loudly_on_an_unknown_format():
    with pytest.raises(ValueError, match="Could not parse"):
        fd.parse_dates(pd.Series(["2019-08-09", "not a date"]))


def test_kickoff_times_convert_from_uk_time_to_utc():
    """August is BST (UTC+1); January is GMT (UTC+0). The offset is not constant."""
    kickoff, known = fd.build_kickoff_utc(
        pd.Series(["09/08/2019", "01/01/2020"]), pd.Series(["20:00", "20:00"])
    )
    assert known.all()
    assert kickoff[0] == pd.Timestamp("2019-08-09 19:00", tz="UTC")  # BST
    assert kickoff[1] == pd.Timestamp("2020-01-01 20:00", tz="UTC")  # GMT


def test_missing_kickoff_time_becomes_midnight_utc_on_the_match_date():
    """The pre-2019/20 seasons have no Time column at all.

    Midnight *UTC* matters: localising midnight UK time in summer would roll the
    timestamp back to 23:00 the previous day and change the match date.
    """
    kickoff, known = fd.build_kickoff_utc(pd.Series(["16/08/14"]), None)
    assert not known.any()
    assert kickoff[0] == pd.Timestamp("2014-08-16 00:00", tz="UTC")
    assert kickoff[0].date() == date(2014, 8, 16)


def test_mixed_known_and_unknown_kickoff_times():
    kickoff, known = fd.build_kickoff_utc(
        pd.Series(["09/08/2019", "10/08/2019"]), pd.Series(["20:00", None])
    )
    assert list(known) == [True, False]
    assert kickoff[0] == pd.Timestamp("2019-08-09 19:00", tz="UTC")
    assert kickoff[1] == pd.Timestamp("2019-08-10 00:00", tz="UTC")


def test_unparseable_kickoff_time_is_an_error():
    with pytest.raises(ValueError, match="kick-off time"):
        fd.build_kickoff_utc(pd.Series(["09/08/2019"]), pd.Series(["8pm"]))


# ---------------------------------------------------------------------------
# Odds selection
# ---------------------------------------------------------------------------


def test_pinnacle_is_preferred_when_available():
    frame = pd.DataFrame(
        {"PSCH": [2.0], "PSCD": [3.5], "PSCA": [4.0],
         "B365CH": [1.9], "B365CD": [3.4], "B365CA": [3.9],
         "AvgCH": [1.95], "AvgCD": [3.45], "AvgCA": [3.95]}
    )
    odds = fd.select_closing_odds(frame)
    assert odds.loc[0, "odds_source"] == "pinnacle_closing"
    assert odds.loc[0, "odds_home"] == 2.0


def test_odds_fall_back_row_by_row_not_season_by_season():
    """Pinnacle vanished partway through January 2026, mid-season."""
    frame = pd.DataFrame(
        {
            "PSCH": [2.0, np.nan, np.nan],
            "PSCD": [3.5, np.nan, np.nan],
            "PSCA": [4.0, np.nan, np.nan],
            "B365CH": [1.9, 1.8, np.nan],
            "B365CD": [3.4, 3.3, np.nan],
            "B365CA": [3.9, 3.8, np.nan],
            "AvgCH": [1.95, 1.85, 1.7],
            "AvgCD": [3.45, 3.35, 3.2],
            "AvgCA": [3.95, 3.85, 4.5],
        }
    )
    odds = fd.select_closing_odds(frame)
    assert list(odds["odds_source"]) == [
        "pinnacle_closing", "bet365_closing", "market_average_closing",
    ]
    assert list(odds["odds_home"]) == [2.0, 1.8, 1.7]


def test_a_partial_price_set_is_skipped_entirely():
    """De-margining needs all three prices, so two out of three is no use."""
    frame = pd.DataFrame(
        {"PSCH": [2.0], "PSCD": [np.nan], "PSCA": [4.0],
         "B365CH": [1.9], "B365CD": [3.4], "B365CA": [3.9]}
    )
    odds = fd.select_closing_odds(frame)
    assert odds.loc[0, "odds_source"] == "bet365_closing"


def test_rows_with_no_usable_odds_are_marked_not_guessed():
    frame = pd.DataFrame({"PSCH": [np.nan], "PSCD": [np.nan], "PSCA": [np.nan]})
    odds = fd.select_closing_odds(frame)
    assert odds.loc[0, "odds_source"] == fd.NO_ODDS
    assert pd.isna(odds.loc[0, "odds_home"])


def test_impossible_prices_are_rejected():
    """Decimal odds of 1.00 mean you win nothing; they are placeholders."""
    frame = pd.DataFrame(
        {"PSCH": [1.0], "PSCD": [1.0], "PSCA": [1.0],
         "AvgCH": [1.95], "AvgCD": [3.45], "AvgCA": [3.95]}
    )
    odds = fd.select_closing_odds(frame)
    assert odds.loc[0, "odds_source"] == "market_average_closing"


def test_missing_odds_columns_are_tolerated():
    """The 2014/15 file has no Bet365 or average closing columns at all."""
    frame = pd.DataFrame({"PSCH": [2.0], "PSCD": [3.5], "PSCA": [4.0]})
    odds = fd.select_closing_odds(frame)
    assert odds.loc[0, "odds_source"] == "pinnacle_closing"


# ---------------------------------------------------------------------------
# De-margining
# ---------------------------------------------------------------------------


def test_market_probabilities_sum_to_one():
    frame = pd.DataFrame(
        {
            "odds_home": [2.0, 1.5, 10.0],
            "odds_draw": [3.5, 4.2, 6.0],
            "odds_away": [4.0, 7.0, 1.3],
        }
    )
    result = fd.add_market_probabilities(frame)
    totals = result[["market_p_home", "market_p_draw", "market_p_away"]].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_market_probabilities_strip_the_bookmakers_margin():
    """Three even prices of 2.5 imply 40% each - 120% in total, a 20% overround.

    After de-margining each outcome should be exactly a third.
    """
    frame = pd.DataFrame({"odds_home": [2.5], "odds_draw": [2.5], "odds_away": [2.5]})
    result = fd.add_market_probabilities(frame)
    assert result.loc[0, "market_overround"] == pytest.approx(1.2)
    for column in ("market_p_home", "market_p_draw", "market_p_away"):
        assert result.loc[0, column] == pytest.approx(1 / 3)


def test_the_favourite_gets_the_highest_probability():
    frame = pd.DataFrame({"odds_home": [1.5], "odds_draw": [4.0], "odds_away": [7.0]})
    result = fd.add_market_probabilities(frame).iloc[0]
    assert result["market_p_home"] > result["market_p_draw"] > result["market_p_away"]


def test_rows_without_odds_get_nan_probabilities_not_zeros():
    frame = pd.DataFrame(
        {"odds_home": [2.0, np.nan], "odds_draw": [3.5, np.nan], "odds_away": [4.0, np.nan]}
    )
    result = fd.add_market_probabilities(frame)
    assert result.loc[0, "market_p_home"] > 0
    assert result[["market_p_home", "market_p_draw", "market_p_away"]].iloc[1].isna().all()


def test_absurd_overround_is_an_error_not_silently_accepted():
    """An overround of 3.0 means the prices are corrupt or misaligned."""
    frame = pd.DataFrame({"odds_home": [1.1], "odds_draw": [1.1], "odds_away": [1.1]})
    with pytest.raises(ValueError, match="overround"):
        fd.add_market_probabilities(frame)


def test_market_probabilities_needs_the_odds_columns():
    with pytest.raises(ValueError, match="missing column"):
        fd.add_market_probabilities(pd.DataFrame({"date": [1]}))


# ---------------------------------------------------------------------------
# Reading raw files
# ---------------------------------------------------------------------------


def test_read_raw_csv_drops_the_trailing_blank_row(tmp_path):
    """The real 2014/15 file has 381 rows, the last one entirely empty."""
    path = write_csv(
        tmp_path / "E0_2014-15_downloaded-2026-01-01.csv",
        LEGACY_HEADER,
        [["E0", "16/08/14", "Arsenal", "Crystal Palace", "2", "1", "H", "J Moss",
          "1.3", "5.5", "12.0"]],
        trailing_blank=True,
    )
    assert len(fd.read_raw_csv(path)) == 1


def test_read_raw_csv_strips_the_byte_order_mark(tmp_path):
    """Newer files carry a BOM; read as plain utf-8 the first column is mangled."""
    path = write_csv(
        tmp_path / "E0_2024-25_downloaded-2026-01-01.csv",
        MODERN_HEADER,
        [modern_row("16/08/2024", "20:00", "Man United", "Fulham", 1, 0, "H")],
        bom=True,
    )
    assert list(fd.read_raw_csv(path).columns)[0] == "Div"


def test_read_raw_csv_complains_if_the_format_changes(tmp_path):
    path = tmp_path / "E0_2024-25_downloaded-2026-01-01.csv"
    path.write_text("Div,Date,Home,Away\nE0,16/08/2024,Man United,Fulham\n")
    with pytest.raises(ValueError, match="missing the column"):
        fd.read_raw_csv(path)


def test_find_raw_files_picks_the_newest_download_per_season(tmp_path):
    for stamp in ("2026-01-01", "2026-08-29"):
        write_csv(
            tmp_path / f"E0_2024-25_downloaded-{stamp}.csv",
            MODERN_HEADER,
            [modern_row("16/08/2024", "20:00", "Man United", "Fulham", 1, 0, "H")],
        )
    found = fd.find_raw_files(tmp_path)
    assert found[2024].name == "E0_2024-25_downloaded-2026-08-29.csv"


def test_find_raw_files_complains_when_there_is_nothing_to_read(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No football-data season CSVs"):
        fd.find_raw_files(tmp_path / "empty")


# ---------------------------------------------------------------------------
# Processing a season
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_season(tmp_path):
    """Two matches in the 2014/15 layout: two-digit dates, no Time column."""
    return write_csv(
        tmp_path / "E0_2014-15_downloaded-2026-08-29.csv",
        LEGACY_HEADER,
        [
            ["E0", "16/08/14", "Arsenal", "Crystal Palace", "2", "1", "H", "J Moss",
             "1.32", "5.50", "12.00"],
            ["E0", "16/08/14", "Leicester", "Everton", "2", "2", "D", "M Jones",
             "2.90", "3.40", "2.65"],
        ],
        trailing_blank=True,
    )


def test_process_season_produces_canonical_names_and_utc_dates(legacy_season):
    tidy = fd.process_season(fd.read_raw_csv(legacy_season), 2014)

    assert len(tidy) == 2
    assert list(tidy["home_team"]) == ["Arsenal", "Leicester City"]
    assert list(tidy["away_team"]) == ["Crystal Palace", "Everton"]
    assert list(tidy["season"]) == ["2014-15", "2014-15"]
    assert list(tidy["result"]) == ["H", "D"]
    assert not tidy["kickoff_time_known"].any()
    assert str(tidy["date"].dt.tz) == "UTC"


def test_every_team_name_maps_to_a_canonical_name(legacy_season):
    """"Leicester" in the file must become "Leicester City" in the output."""
    tidy = fd.process_season(fd.read_raw_csv(legacy_season), 2014)
    assert tidy["home_team"].notna().all()
    assert tidy["away_team"].notna().all()
    assert "Leicester" not in set(tidy["home_team"])


def test_an_unknown_team_name_stops_the_build(tmp_path):
    """A newly promoted club must be added to the lookup, not silently dropped."""
    path = write_csv(
        tmp_path / "E0_2026-27_downloaded-2026-08-29.csv",
        LEGACY_HEADER,
        [["E0", "21/08/2026", "Arsenal", "Wrexham", "3", "0", "H", "T Bramall",
          "1.2", "6.0", "15.0"]],
    )
    with pytest.raises(UnknownTeamError, match="Wrexham"):
        fd.process_season(fd.read_raw_csv(path), 2026)


def test_a_result_that_disagrees_with_the_score_stops_the_build(tmp_path):
    path = write_csv(
        tmp_path / "E0_2024-25_downloaded-2026-08-29.csv",
        MODERN_HEADER,
        [modern_row("16/08/2024", "20:00", "Man United", "Fulham", 1, 0, "A")],
    )
    with pytest.raises(ValueError, match="disagrees with the score"):
        fd.process_season(fd.read_raw_csv(path), 2024)


def test_an_unexpected_result_code_stops_the_build(tmp_path):
    path = write_csv(
        tmp_path / "E0_2024-25_downloaded-2026-08-29.csv",
        MODERN_HEADER,
        [modern_row("16/08/2024", "20:00", "Man United", "Fulham", 1, 0, "X")],
    )
    with pytest.raises(ValueError, match="unexpected result code"):
        fd.process_season(fd.read_raw_csv(path), 2024)


def test_non_numeric_goals_stop_the_build(tmp_path):
    path = write_csv(
        tmp_path / "E0_2024-25_downloaded-2026-08-29.csv",
        MODERN_HEADER,
        [modern_row("16/08/2024", "20:00", "Man United", "Fulham", "P", 0, "H")],
    )
    with pytest.raises(ValueError, match="non-numeric goal"):
        fd.process_season(fd.read_raw_csv(path), 2024)


# ---------------------------------------------------------------------------
# Building the whole table
# ---------------------------------------------------------------------------


@pytest.fixture
def two_seasons(tmp_path):
    """One legacy season and one modern season, in the same raw directory."""
    write_csv(
        tmp_path / "E0_2014-15_downloaded-2026-08-29.csv",
        LEGACY_HEADER,
        [
            ["E0", "16/08/14", "Arsenal", "Crystal Palace", "2", "1", "H", "J Moss",
             "1.32", "5.50", "12.00"],
            ["E0", "17/08/14", "Leicester", "Everton", "2", "2", "D", "M Jones",
             "2.90", "3.40", "2.65"],
        ],
    )
    write_csv(
        tmp_path / "E0_2025-26_downloaded-2026-08-29.csv",
        MODERN_HEADER,
        [
            # Pinnacle available.
            modern_row("15/08/2025", "20:00", "Liverpool", "Bournemouth", 4, 2, "H",
                       PSCH="1.35", PSCD="5.60", PSCA="8.50",
                       B365CH="1.33", B365CD="5.50", B365CA="8.00",
                       AvgCH="1.34", AvgCD="5.55", AvgCA="8.20"),
            # Pinnacle gone, as it is from January 2026.
            modern_row("17/01/2026", "15:00", "Everton", "Arsenal", 0, 1, "A",
                       B365CH="4.20", B365CD="3.60", B365CA="1.90",
                       AvgCH="4.10", AvgCD="3.55", AvgCA="1.88"),
        ],
    )
    return tmp_path


def test_build_results_combines_seasons_in_date_order(two_seasons):
    results = fd.build_results(two_seasons)
    assert len(results) == 4
    assert list(results["season"]) == ["2014-15", "2014-15", "2025-26", "2025-26"]
    assert results["date"].is_monotonic_increasing
    assert list(results.columns) == fd.RESULTS_COLUMNS


def test_build_results_records_which_bookmaker_each_row_used(two_seasons):
    results = fd.build_results(two_seasons)
    assert list(results["odds_source"]) == [
        "pinnacle_closing", "pinnacle_closing", "pinnacle_closing", "bet365_closing",
    ]


def test_probabilities_sum_to_one_within_tolerance(two_seasons):
    results = fd.build_results(two_seasons)
    totals = results[["market_p_home", "market_p_draw", "market_p_away"]].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_build_results_has_no_duplicate_matches(two_seasons):
    results = fd.build_results(two_seasons)
    assert not results.duplicated(subset=["season", "home_team", "away_team"]).any()


def test_a_duplicated_fixture_stops_the_build(tmp_path):
    """The same pairing twice in one season means the raw file is wrong."""
    write_csv(
        tmp_path / "E0_2025-26_downloaded-2026-08-29.csv",
        MODERN_HEADER,
        [
            modern_row("15/08/2025", "20:00", "Liverpool", "Bournemouth", 4, 2, "H",
                       PSCH="1.35", PSCD="5.60", PSCA="8.50"),
            modern_row("22/08/2025", "15:00", "Liverpool", "Bournemouth", 1, 1, "D",
                       PSCH="1.40", PSCD="5.00", PSCA="8.00"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate fixture"):
        fd.build_results(tmp_path)


def test_build_results_can_be_limited_to_named_seasons(two_seasons):
    results = fd.build_results(two_seasons, start_years=[2025])
    assert set(results["season"]) == {"2025-26"}


def test_asking_for_a_season_we_have_not_downloaded_is_an_error(two_seasons):
    with pytest.raises(FileNotFoundError, match="2019-20"):
        fd.build_results(two_seasons, start_years=[2019])


def test_write_results_round_trips_through_parquet(two_seasons, tmp_path):
    results = fd.build_results(two_seasons)
    path = fd.write_results(results, tmp_path / "out" / "results.parquet")
    reloaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(results, reloaded)
