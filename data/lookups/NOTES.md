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
| `fbref_name` | **Still not verified.** The first Actions probe crashed before making a request (a `PosixPath` passed where seleniumbase wanted a string; fixed 2026-08-29), so FBref reachability is still untested from anywhere. Confirm on the first successful pull — the probe lists any name that does not match. |
| `understat_name` | **Verified** for 2014/15–2026/27 on 2026-08-29 via the first full `soccerdata` pull. All 13 seasons map with nothing unmapped and nothing spare, which confirms the short forms (`Leicester`, `Ipswich`, `Brighton`, `Hull`) and the long ones (`Wolverhampton Wanderers`, `Nottingham Forest`, `Queens Park Rangers`). |
| `footballdata_name` | **Verified** against all 13 season CSVs (2014/15–2026/27) on 2026-08-29. Every `HomeTeam`/`AwayTeam` value in every file maps to a canonical name, and every name in this column is actually used. Nothing missing, nothing spare. |

Verified Club Elo names: Arsenal, Man City, Liverpool, Aston Villa, Chelsea,
Man United, Newcastle, Brighton, Bournemouth, Crystal Palace, Everton,
Tottenham, Leeds, Fulham, Forest, Sunderland, Ipswich.

Verified FPL names: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
Chelsea, Coventry City, Crystal Palace, Everton, Fulham, Hull City,
Ipswich Town, Leeds, Liverpool, Man City, Man Utd, Newcastle, Nott'm Forest,
Spurs, Sunderland.

## Specific rows to check — TODO

1. **Coventry City** — newly promoted, and it has not been in the Premier League
   since 1999/2000. `footballdata_name` = `Coventry` is now confirmed from the
   2026/27 CSV. Understat has no historic EPL entry for the club at all, so
   `understat_name` = `Coventry` is still a guess based on how Understat
   shortens other names, and `clubelo_name` is also unconfirmed. Check both on
   the first scrape of the new season.
2. ~~**Understat short names generally**~~ — resolved. The full pull confirmed
   every name across all 13 seasons: Understat abbreviates some clubs (`Leeds`,
   `Hull`, `Cardiff`, `Norwich`, `Stoke`, `Swansea`, `Luton`, `Leicester`,
   `Huddersfield`, `Ipswich`) and writes others out in full (`Wolverhampton
   Wanderers`, `West Bromwich Albion`, `Queens Park Rangers`, `Newcastle
   United`, `Manchester United`), exactly as recorded.
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

## Elo: which source the model uses (updated 2026-08-29)

**Primary source: `internal`.** `data/processed/elo_history.parquet` is computed
from our own `results.parquet` by `src/ratings/elo.py`. It needs no network, so
it always exists and cannot be broken by a third party. `get_elo()` reads it by
default. **If you want the model to consume live Club Elo instead, change this
line** — the pipeline writes the live pull to a separate file and never
overwrites the primary one.

**Cross-check: `clubelo`.** When the live pull succeeds it is written to
`data/processed/elo_history_clubelo.parquet` and compared against ours; the
agreement (rank correlation, mean rank difference) is printed by
`python -m pipelines.build_context --only elo`.

### Club Elo was never blocked — it is just very slow

An earlier note here said the API was unreachable. That was wrong, and the
mistake is worth recording. api.clubelo.com **does** answer; it is simply slow
enough that a 30-second read timeout expired first, and every route we tried
used that same 30-second default. So three separate "the network is blocking
it" conclusions were all really one bug in our own client.

Fixed: the read timeout is now 180 seconds with two retries and backoff. The
client also prefers `api.clubelo.com/YYYY-MM-DD`, which returns the **entire
league table in one response**, so a full refresh is about 14 requests instead
of 36 — on a slow server that is the difference between seconds and hours.

Even at 180 seconds it still times out from the sandbox this was built in, so
that environment may genuinely be restricted as well as the server being slow.
It is expected to work from a normal network.

**`clubelo_name` is still unverified for 19 of 36 clubs.** One successful
snapshot settles the whole column at once, because the dated table contains
every club — that is what the `check-sources` workflow now probes.

## FBref: reachability still untested (updated 2026-08-29)

FBref sits behind Cloudflare. From the sandbox this project was built in, every
route was blocked:

- a plain HTTPS request returns a Cloudflare `403` challenge page;
- so does a request with a spoofed TLS fingerprint (the trick `soccerdata` uses
  successfully for Understat);
- the real-browser fallback `soccerdata` falls back to had its connections cut
  by the network before any page loaded.

**The first GitHub Actions probe did not settle this.** It crashed with
`AttributeError: 'PosixPath' object has no attribute 'lower'` before making a
request: `soccerdata` annotates `path_to_browser` as a `Path`, but hands it
straight to seleniumbase, which calls `.lower()` on it. Fixed by passing a
string; the neighbouring `data_dir` argument genuinely does want a `Path`, which
is what made it easy to get wrong. A regression test now pins the types. So
FBref has still never been tried properly from a permissive network.

Otherwise this looks like an environment restriction rather than anything wrong
with the scraper, which is written, tested and cache-first. On a normal machine with
Chrome or Chromium installed it should work; if it does not, point
`PREMFORECASTER_BROWSER` at the browser binary. Understat supplies the xG the
model actually depends on, so the pipeline is built to carry on without FBref.

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
