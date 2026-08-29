# Lookup notes — please verify the flagged rows

These notes exist because CSV files cannot hold comments. This is the list of
things I filled in that you should double-check before we trust any join.

The club list covers the 20 clubs in the Premier League right now **plus** every
club that has played in the Premier League since 2014/15 (the Understat data
horizon). That is 36 clubs.

## What was verified live, and what was not

| Source column | Status |
|---|---|
| `fpl_name` | **Verified** for the 20 current clubs against the live FPL API (`fantasy.premierleague.com/api/bootstrap-static/`, fetched 2026-08-29). Historic clubs are from memory — see below. |
| `clubelo_name` | **Verified** for 17 clubs against the live Club Elo site (`clubelo.com/ENG`, English level 1). The rest are from memory. |
| `fbref_name` | **Not verified** — no request made. FBref is rate limited to 1 request / 7 seconds, so we should confirm these when the FBref scraper first runs. |
| `understat_name` | **Not verified** — the Understat page did not return its embedded team JSON when fetched. Confirm on the first Understat pull. |
| `footballdata_name` | **Not verified** — these are the `HomeTeam`/`AwayTeam` values in the E0 season CSVs, from memory. Easy to confirm: download one season CSV and list the unique values. |

Verified Club Elo names: Arsenal, Man City, Liverpool, Aston Villa, Chelsea,
Man United, Newcastle, Brighton, Bournemouth, Crystal Palace, Everton,
Tottenham, Leeds, Fulham, Forest, Sunderland, Ipswich.

Verified FPL names: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
Chelsea, Coventry City, Crystal Palace, Everton, Fulham, Hull City,
Ipswich Town, Leeds, Liverpool, Man City, Man Utd, Newcastle, Nott'm Forest,
Spurs, Sunderland.

## Specific rows to check — TODO

1. **Coventry City** — newly promoted, and it has not been in the Premier League
   since 1999/2000. That means Understat has no historic EPL entry for it at
   all, so `understat_name` = `Coventry` is a guess based on how Understat
   shortens other names. `clubelo_name` and `footballdata_name` are also
   guesses. Check all three on the first scrape of the new season.
2. **Understat short names generally** — Understat abbreviates some clubs
   (`Leeds`, `Hull`, `Cardiff`, `Norwich`, `Stoke`, `Swansea`, `Luton`,
   `Leicester`, `Huddersfield`, `Ipswich`) but writes others out in full
   (`Wolverhampton Wanderers`, `West Bromwich Albion`, `Queens Park Rangers`,
   `Newcastle United`, `Manchester United`). I am confident about the full-form
   ones and less confident about the short ones. Verify the whole column.
3. **Nottingham Forest** — five different spellings across five sources
   (`Nott'ham Forest`, `Nottingham Forest`, `Nott'm Forest`, `Forest`,
   `Nott'm Forest`). The Club Elo `Forest` and the FPL `Nott'm Forest` are
   verified; the FBref apostrophe form is not.
4. **FPL historic names** — the FPL API only ever exposes the current season, so
   for relegated clubs (Leicester, Southampton, Wolves, West Ham, Sheffield Utd,
   Luton, Burnley, and the older ones) these are from memory. They only matter
   if we ever backfill historic FPL data; for live predictions the current 20
   are what count, and those are verified.
5. **Everton's stadium** — Everton left Goodison Park and now play at the
   **Hill Dickinson Stadium** at Bramley-Moore Dock. The coordinates in
   `stadiums.csv` (53.3906, -3.0023) are approximate and should be checked.

## A structural gap to fix later

`stadiums.csv` holds **one stadium per club**, which is fine for forecasting
upcoming fixtures but wrong for back-testing, because several clubs have moved
inside our data window:

- Everton: Goodison Park (53.4388, -2.9663) until the end of 2024/25.
- Tottenham: White Hart Lane / Wembley until 2018/19.
- West Ham: Upton Park (Boleyn Ground) until 2015/16.
- Brentford: Griffin Park until 2019/20.

When weather features get added to the back-test, this needs to become a
date-effective table (`canonical_name, stadium, valid_from, valid_to, lat, lon`).
Until then, weather is only trustworthy for recent and upcoming fixtures.

## How `soccerdata` handles names

`soccerdata` ships **no** built-in team-name replacements. It logs
"No custom team name replacements found" and passes source names through
unchanged, unless you create
`$SOCCERDATA_DIR/config/teamname_replacements.json`. So the `fbref_name` and
`understat_name` columns here should hold the **raw** source spellings. If we
ever add a soccerdata replacements file, this CSV must be updated to match, or
joins will silently start missing rows.
