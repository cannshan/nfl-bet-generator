"""Tracks every bet this app suggests, then automatically checks what
actually happened once the game is over -- the only way to know whether any
of this app's heuristics genuinely help or just look reasonable.

Nothing manual required: predictions are recorded the moment they're shown
on the page, and settled automatically on a later page load once the real
result is available (final score from ESPN, final box-score stat from
nflverse). Deliberately does NOT feed results back into the model yet (no
auto-recalibration) -- with only a handful of settled bets at first, any
"learning" would just be overfitting to noise. What this gives you right
now is the honest calibration report: does a bet this app called 70% likely
actually hit around 70% of the time? That's the number no amount of clever
feature engineering can fake.

Also tracks closing line value (CLV) -- whether the suggested price beat the
market's own closing number -- since that's a faster, sharp-betting-standard
skill signal that doesn't need to wait for a game's outcome, just for the
line to move. See capture_closing_lines(). Alongside the raw price CLV it
records the market's closing FAIR probability for the exact bet we suggested
(closing_fair_prob), which turns into "closing EV" on the Track Record page:
with a few hundred bets, that is a far less noisy read on whether the picks
were good than hit/miss ever can be.

Whole parlay tickets are tracked too (log_tickets / settle_tickets): the
app's actual product is a $5 -> $1000 ticket, and per-leg records alone
can't say whether the tickets themselves hit as often as the shown
probability claimed.

Storage: Supabase (Postgres), tables `nfl_predictions` and `nfl_tickets`,
in a project shared with other unrelated apps -- the table names are
deliberately namespaced so they can't collide with anything else living in
that same database. Uses the service_role key since this module only ever
runs server-side and there's no per-user auth in this app (Row Level
Security bypass is intentional).

Schema changes ship as a migration in supabase_setup.sql that has to be run
by hand in the Supabase SQL editor (this app has no DDL access). Until it
is, every function here keeps working on the old schema: a write that names
a not-yet-created column is retried without it, a missing table is skipped,
and that is remembered for the rest of the process so it isn't re-tried on
every call. See _run_with_schema_fallback().
"""
import csv
import datetime as dt
import hashlib
import io
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
from supabase import create_client, ClientOptions
from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, RATINGS_CACHE_TTL_SECONDS
from app import espn_client, nflverse_client, formatting, cache_utils

SETTLEMENT_DELAY_HOURS = 4  # wait this long past kickoff so final stats have posted
CALIBRATION_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
MIN_SAMPLE_FOR_CONFIDENCE = 20  # below this, don't imply the numbers mean much yet
MIN_SAMPLE_FOR_CLV_CONFIDENCE = 200  # sharp-betting convention: CLV needs a bigger sample than raw hit rate to mean much
# A $5 -> $1000 ticket is a ~0.5% shot: even a perfectly honest model
# expects about one hit per 200 tickets, so the ticket hit rate says almost
# nothing until there are a few hundred of them.
MIN_SAMPLE_FOR_TICKET_CONFIDENCE = 200
# A "closing" price only counts if it was captured at least this long after
# the suggestion itself -- see _close_after_suggestion().
CLOSE_MIN_GAP = dt.timedelta(minutes=5)

TABLE = "nfl_predictions"
TICKETS_TABLE = "nfl_tickets"
# Bumped whenever the probability model changes materially, so the Track
# Record page can report the CURRENT model's calibration rather than blend
# it with suggestions an earlier, since-replaced model made. v1 = the
# original history-only prop model (shown ~75% confidence, hit 37%);
# v2 = market-anchored props + correlation-adjusted parlays.
MODEL_VERSION = "v2"

# PostgREST silently truncates every response at the project's max-rows
# setting (Supabase default: 1000) -- no error, just fewer rows. The
# prediction log passes that within a few weeks of a season, after which
# settlement and the Track Record would quietly start ignoring rows. Every
# full-table read goes through _select_all(), which pages until a page
# comes back empty, so it's correct whatever that cap is set to.
PAGE_SIZE = 1000

# Columns added to nfl_predictions after the table first shipped. A write
# naming one of these against a database that hasn't run the migration yet
# is retried without it (see _run_with_schema_fallback).
NEW_PREDICTION_COLUMNS = ("history_prob", "closing_fair_prob")
_missing_columns = set()       # NEW_PREDICTION_COLUMNS this process found missing
_tickets_table_missing = False  # nfl_tickets not created yet (this process)

# nflverse's player-week stats only contain a player who recorded some stat
# (a target, a carry, a tackle, a return...), so "no box-score row" can mean
# EITHER "didn't play" (books void the prop) OR "played and was never
# involved" (books grade it: 0 receptions, Over loses). Snap counts tell the
# two apart; they're posted a day or two after the game. A prop with no
# stat row waits up to this long for them before falling back to voiding --
# the fallback is right for the usual case (an injured or inactive player),
# but it would wrongly void a played-but-zero game, which is exactly the
# shape of the low-usage "Under 1.5 receptions" legs this app likes to pick.
DNP_VOID_FALLBACK_HOURS = 72
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
SNAP_FIELDS = ["game_id", "player", "offense_snaps", "defense_snaps", "st_snaps"]
SNAP_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}  # the only positions this app bets props on
SNAP_TIMEOUT = 20

# nflverse numbers the playoff rounds 19-22 (Wild Card .. Super Bowl). Dated
# weeks line up with that through the conference round; the Super Bowl
# comes after a bye week, so anything past 22 by date is still week 22.
LAST_WEEK_NUMBER = 22

# Per-thread client, for the same reason as cache_utils._sb(): the
# settle/closing-line writes run 20 at a time from a thread pool, and a
# single shared httpx session under that load threw socket read errors
# on Windows that were swallowed as failed writes.
_local = threading.local()


def _sb():
    client = getattr(_local, "client", None)
    if client is None:
        client = create_client(
            SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
            options=ClientOptions(postgrest_client_timeout=20),
        )
        _local.client = client
    return client


# --- schema fallback ---------------------------------------------------------

def _error_code_and_text(exc):
    """postgrest's APIError carries .code/.message; anything else (a
    transport error, an older client) only has its string form."""
    return getattr(exc, "code", None), (getattr(exc, "message", None) or str(exc))


def _missing_column_name(exc):
    """The column a failed request complained about, or None if the failure
    was anything else. Two shapes, both verified against this project:
    a write naming an unknown column -> PGRST204 "Could not find the
    'history_prob' column of 'nfl_predictions' in the schema cache"; a
    select naming one -> 42703 "column nfl_predictions.history_prob does
    not exist"."""
    code, text = _error_code_and_text(exc)
    m = re.search(r"Could not find the '([^']+)' column", text)
    if m and code in (None, "PGRST204"):
        return m.group(1)
    m = re.search(r"column (?:\w+\.)?(\w+) does not exist", text)
    if m and code in (None, "42703"):
        return m.group(1)
    return None


def _is_missing_table(exc):
    """PGRST205 "Could not find the table 'public.nfl_tickets' in the schema
    cache" (verified against this project), or Postgres's own 42P01 when
    the request reaches the database."""
    code, text = _error_code_and_text(exc)
    return code in ("PGRST205", "42P01") or "Could not find the table" in text


def _strip_missing(payload):
    if not _missing_columns:
        return payload
    if isinstance(payload, list):
        return [{k: v for k, v in row.items() if k not in _missing_columns} for row in payload]
    return {k: v for k, v in payload.items() if k not in _missing_columns}


def _run_with_schema_fallback(execute, payload):
    """Runs execute(payload) -- one insert/upsert/update against
    nfl_predictions -- and reports success. If the database hasn't had
    the migration that adds one of NEW_PREDICTION_COLUMNS yet, the request
    fails as a whole (PostgREST rejects unknown columns before touching any
    row), so the column is remembered as missing for this process and the
    same write is retried without it: the old fields still get recorded
    exactly as before the column existed. Never raises."""
    payload = _strip_missing(payload)
    for _attempt in range(len(NEW_PREDICTION_COLUMNS) + 1):
        try:
            execute(payload)
            return True
        except Exception as e:  # noqa: BLE001 - any API/transport failure
            column = _missing_column_name(e)
            if column not in NEW_PREDICTION_COLUMNS or column in _missing_columns:
                return False
            _missing_columns.add(column)
            payload = _strip_missing(payload)
    return False


def _select_all(build_query):
    """Every row a query matches, paged (see PAGE_SIZE). `build_query` must
    return a FRESH query builder each call -- the builders are single-use."""
    rows, start = [], 0
    while True:
        page = build_query().order("id").range(start, start + PAGE_SIZE - 1).execute().data
        if not page:
            return rows
        rows.extend(page)
        start += len(page)


def _update_rows(table, updates):
    """{row id: payload} -> number of rows written. Fired concurrently: one
    request per row was measured as a major share of page load time when
    run sequentially (several dozen rows settle at once after a Sunday)."""
    if not updates:
        return 0

    def _write(item):
        row_id, payload = item
        if table == TABLE:
            return _run_with_schema_fallback(
                lambda p: _sb().table(table).update(p).eq("id", row_id).execute(), payload,
            )
        try:
            _sb().table(table).update(payload).eq("id", row_id).execute()
            return True
        except Exception:  # noqa: BLE001
            return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        return sum(pool.map(_write, updates.items()))


# --- time / week helpers -----------------------------------------------------

def _parse_commence(iso_str):
    """The odds feed's UTC kickoff ('2026-09-15T00:15:00Z'); also tolerates
    an explicit offset or fractional seconds. Naive values are UTC."""
    if not iso_str:
        return None
    try:
        return dt.datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        parsed = dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _utc_iso_z(moment):
    """Same text shape as commence_time, so a TEXT comparison in a PostgREST
    filter orders correctly."""
    return moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_one_tuesday(season):
    return espn_client.season_kickoff_date(season) - dt.timedelta(days=2)


def season_week_for_kickoff(commence_iso):
    """(season, week) of the game kicking off at `commence_iso`, from the
    kickoff itself -- the single source of truth for every row's
    season/week label (log_suggestions, log_tickets, the backfill script).

    Why not ESPN's "current week": that's the week of whenever the page
    happened to be refreshed, which on a cache-only (passive) page view can
    be days stale, and a Thursday refresh legitimately shows next Sunday's
    games -- rows for 10/04 games were being labeled week 3.

    Weeks run Tuesday -> Monday on the US Eastern calendar date: a Monday
    night game (01:15 UTC Tuesday) stays in its Sunday's week, and a rare
    Wednesday game (Christmas 2024) lands in the week of the Thursday after
    it, matching nflverse's own week numbers. Verified against every 2026
    game nflverse had posted (weeks 1-4, 44 games, 0 mismatches). Preseason
    dates come back as week 0 of the upcoming season. (None, None) if the
    kickoff can't be parsed."""
    commence = _parse_commence(commence_iso)
    if commence is None:
        return None, None
    day = commence.astimezone(formatting.EASTERN).date()
    if day >= _week_one_tuesday(day.year):
        season = day.year
    elif day.month >= 3:
        return day.year, 0  # preseason (August / early September)
    else:
        season = day.year - 1  # January/February: last year's season
    week = (day - _week_one_tuesday(season)).days // 7 + 1
    return season, min(week, LAST_WEEK_NUMBER)


def _logged_before_kickoff(row):
    """False for a suggestion recorded after its game had already started.
    The odds feed keeps listing a game once it's live -- with in-game
    prices and commence_time rewritten to the actual kickoff -- so before
    log_suggestions learned to skip those, 17 rows were logged mid-game
    (one 3 hours after kickoff). Those prices already knew most of the
    outcome; counting them would flatter the record."""
    created = _parse_commence(row.get("created_at"))
    commence = _parse_commence(row.get("commence_time"))
    if created is None or commence is None:
        return True
    return created < commence


def _close_after_suggestion(row):
    """True if the row's closing price was captured on a LATER refresh than
    the one that suggested it. capture_closing_lines runs in the same
    request that logs a suggestion, so a bet whose game started before the
    app was refreshed again "closed" at exactly the price it was shown at:
    109 of the first 156 same-line closes were that (captured under a
    minute after logging, CLV 0 by construction), which buried the 47 real
    ones -- 55% of those beat the close, the page said 17%."""
    created = _parse_commence(row.get("created_at"))
    captured = _parse_commence(row.get("closing_captured_at"))
    if created is None or captured is None:
        return False
    return captured - created >= CLOSE_MIN_GAP


# --- logging -----------------------------------------------------------------

def _split_matchup(matchup):
    if " @ " not in (matchup or ""):
        return None, None
    away, home = matchup.split(" @ ", 1)
    return home, away


def log_suggestions(legs, season, week, section):
    """Records each leg the FIRST time it's suggested; later refreshes that
    re-suggest the same bet are silently ignored (unique constraint + upsert
    ignore_duplicates) so the log reflects what was said before the outcome
    was known, not a constantly-overwritten latest guess.

    season/week are stamped from each leg's own kickoff (see
    season_week_for_kickoff); the arguments are only a fallback for a leg
    without a parseable kickoff. A leg whose game has already kicked off is
    not recorded at all: its price is a live in-game number, not a pregame
    suggestion (see _logged_before_kickoff)."""
    if not legs:
        return
    now = dt.datetime.now(dt.timezone.utc)
    created = now.replace(tzinfo=None).isoformat()
    rows = []
    for leg in legs:
        home, away = _split_matchup(leg.get("matchup"))
        if not home:
            continue
        commence = _parse_commence(leg.get("commence_time"))
        if commence is not None and commence <= now:
            continue
        leg_season, leg_week = season_week_for_kickoff(leg.get("commence_time"))
        if leg_season is None:
            leg_season, leg_week = season, week
        rows.append({
            "created_at": created, "season": leg_season, "week": leg_week,
            "home_team": home, "away_team": away, "commence_time": leg.get("commence_time"),
            "market": leg["market"], "selection": leg["selection"],
            "player": leg.get("player"), "stat_category": leg.get("stat_category"),
            "side": leg.get("side"), "line": leg.get("line"),
            "american_odds": leg["american_odds"], "decimal_odds": leg["decimal_odds"],
            "bookmaker": leg["bookmaker"], "model_prob": leg["model_prob"],
            "book_fair_prob": leg["book_fair_prob"], "edge": leg["edge"],
            # What the player's own game log alone said (before the small
            # capped tilt into model_prob) -- stored so the tilt weight can be
            # re-fitted on real outcomes later. None for game-level legs.
            "history_prob": leg.get("history_prob"),
            "section": section, "status": "pending",
        })
    if not rows:
        return
    _run_with_schema_fallback(
        lambda payload: _sb().table(TABLE).upsert(
            payload,
            on_conflict="home_team,away_team,commence_time,market,selection",
            ignore_duplicates=True,
        ).execute(),
        rows,
    )


def _leg_identity(leg):
    """The same five fields nfl_predictions' unique constraint uses -- one
    bet, wherever and however often it was shown."""
    home, away = _split_matchup(leg.get("matchup"))
    if not home:
        return None
    return [home, away, leg.get("commence_time"), leg.get("market"), leg.get("selection")]


def ticket_key(legs):
    """Stable id for a ticket: a hash of its legs' identities, sorted, so the
    same combination recomputed on a later refresh (or listed in another
    order) is recognized as the same ticket. None if any leg can't be
    identified."""
    identities = [_leg_identity(leg) for leg in legs]
    if not identities or any(i is None for i in identities):
        return None
    # Sorted as JSON text, so a None field can't break the ordering.
    blob = json.dumps(sorted(json.dumps(i) for i in identities), separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _ticket_row(parlay, section, now, created):
    legs = parlay.get("legs") or []
    key = ticket_key(legs)
    if key is None:
        return None
    kickoffs = [_parse_commence(leg.get("commence_time")) for leg in legs]
    if any(k is None for k in kickoffs) or min(kickoffs) <= now:
        return None  # a pregame ticket can't be placed once any leg has started
    first_commence = min(legs, key=lambda leg: _parse_commence(leg["commence_time"]))["commence_time"]
    season, week = season_week_for_kickoff(first_commence)
    stored_legs = []
    for leg, (home, away, commence, market, selection) in zip(legs, (_leg_identity(l) for l in legs)):
        # Identity + everything needed to grade the leg on its own (if its
        # nfl_predictions row is ever missing) + the price/probability the
        # ticket was shown with -- the ticket's own leg prices, which can
        # differ from the leg row's first-seen price.
        stored_legs.append({
            "home_team": home, "away_team": away, "commence_time": commence,
            "market": market, "selection": selection,
            "player": leg.get("player"), "stat_category": leg.get("stat_category"),
            "side": leg.get("side"), "line": leg.get("line"),
            "decimal_odds": leg.get("decimal_odds"), "model_prob": leg.get("model_prob"),
            "book_fair_prob": leg.get("book_fair_prob"),
        })
    return {
        "ticket_key": key, "created_at": created, "season": season, "week": week,
        "section": section, "num_legs": len(legs), "legs": stored_legs,
        "stake": parlay.get("stake"), "decimal_odds": parlay.get("decimal_odds"),
        "payout": parlay.get("payout"), "combined_prob": parlay.get("combined_prob"),
        "breakeven_prob": parlay.get("breakeven_prob"), "ev_pct": parlay.get("ev_pct"),
        "has_same_game_legs": parlay.get("has_same_game_legs"),
        "first_commence_time": first_commence, "status": "pending",
    }


def log_tickets(parlays, section):
    """Records each whole parlay ticket shown (parlay_builder.build_result
    dicts) the first time it's shown -- same first-write-wins rule as
    log_suggestions, keyed on ticket_key(). Pregame tickets only. The legs
    themselves are logged separately by log_suggestions; settle_tickets()
    reads their results from there. Best-effort: never raises, and a
    database without the nfl_tickets table yet is simply skipped (and not
    re-tried for the rest of the process)."""
    global _tickets_table_missing
    if _tickets_table_missing or not parlays:
        return
    try:
        now = dt.datetime.now(dt.timezone.utc)
        created = now.replace(tzinfo=None).isoformat()
        rows = {}
        for parlay in parlays:
            row = _ticket_row(parlay, section, now, created)
            if row:
                rows[row["ticket_key"]] = row
        if not rows:
            return
        _sb().table(TICKETS_TABLE).upsert(
            list(rows.values()), on_conflict="ticket_key", ignore_duplicates=True,
        ).execute()
    except Exception as e:  # noqa: BLE001
        if _is_missing_table(e):
            _tickets_table_missing = True


# --- settlement ----------------------------------------------------------------

def _settle_game_market(row, completed_games):
    """completed_games: {(home_team, away_team): (home_score, away_score)}"""
    scores = completed_games.get((row["home_team"], row["away_team"]))
    if not scores:
        return None
    home_score, away_score = scores
    market, selection = row["market"], row["selection"]

    if market == "Moneyline":
        if home_score == away_score:
            return "void"
        winner = row["home_team"] if home_score > away_score else row["away_team"]
        return "hit" if selection == winner else "miss"

    if market == "Spread":
        m = re.match(r"^(.*) ([+-]?\d+(?:\.\d+)?)$", selection)
        if not m:
            return None
        team, point = m.group(1), float(m.group(2))
        margin = (home_score - away_score) if team == row["home_team"] else (away_score - home_score)
        if margin + point == 0:
            return "void"
        return "hit" if margin + point > 0 else "miss"

    if market == "Total":
        m = re.match(r"^(Over|Under) (\d+(?:\.\d+)?)$", selection)
        if not m:
            return None
        side, line = m.group(1), float(m.group(2))
        total = home_score + away_score
        if total == line:
            return "void"
        actual_over = total > line
        return "hit" if (actual_over and side == "Over") or (not actual_over and side == "Under") else "miss"

    return None


def _team_pair(team_a, team_b):
    return frozenset((team_a, team_b))


def _row_week(row):
    try:
        return int(row.get("week"))
    except (TypeError, ValueError):
        return None


def _load_season_results(season):
    """Everything settlement needs for one season, fetched once per call:
    final scores, the player box-score index, and which games' box scores
    have been posted ({team pair: {week, ...}}, from this season's
    player-week rows)."""
    try:
        games = espn_client.get_season_games_to_date(season)
    except Exception:  # noqa: BLE001
        games = []
    try:
        player_index = nflverse_client.build_player_index(season)
    except Exception:  # noqa: BLE001
        player_index = {}
    posted = defaultdict(set)
    for rows in player_index.values():
        for r, w in rows:
            if w < 1.0:
                continue
            team_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("team"))
            opp_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("opponent_team"))
            week = _row_week(r)
            if team_full and opp_full and week is not None:
                posted[_team_pair(team_full, opp_full)].add(week)
    return {
        "season": season,
        "completed_games": {(g["home"], g["away"]): (g["home_score"], g["away_score"]) for g in games},
        "player_index": player_index,
        "posted_weeks": posted,
        "snap_counts": None,  # loaded lazily -- only a no-stat-row prop ever needs it
    }


def _box_score_week(results, row):
    """nflverse's week number for this row's game if its box score has been
    posted, else None. Matched on the team PAIR plus the week, not the pair
    alone: division rivals meet twice a season, and matching by pair only
    (what this used to do) would grade a bet on the second meeting with the
    first meeting's stats. The kickoff-derived week is allowed to be off by
    one (a postponed Tuesday/Wednesday game) -- nflverse's own number wins."""
    weeks = results["posted_weeks"].get(_team_pair(row["home_team"], row["away_team"]))
    if not weeks:
        return None
    _season, derived = season_week_for_kickoff(row.get("commence_time"))
    if derived is None:
        derived = _row_week(row)
    if derived is None:
        return None
    nearby = [w for w in weeks if abs(w - derived) <= 1]
    return min(nearby, key=lambda w: abs(w - derived)) if nearby else None


def _grade_prop(row, actual):
    if actual == row["line"]:
        return "void"
    is_over = actual > row["line"]
    hit = is_over if row["side"] == "Over" else not is_over
    return "hit" if hit else "miss"


def _normalize_name(name):
    """Snap counts come from Pro Football Reference, whose names can differ
    cosmetically from nflverse's ("D.J. Moore" / "DJ Moore", "Jr.")."""
    name = re.sub(r"[.'’,-]", "", (name or "").lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _fetch_snap_counts(season):
    """{nflverse game_id: {normalized player name: total snaps}} from
    nflverse's snap-counts release (offense skill positions only, trimmed
    before caching for the same reason nflverse_client trims its releases).
    Free, no key. Follows cache_utils' passive/active rule: a plain page
    view never fetches. {} when unavailable."""
    cache_key = f"nflverse_snap_counts_{season}"
    rows = cache_utils.cache_get(cache_key, RATINGS_CACHE_TTL_SECONDS)
    if rows is None:
        if not cache_utils.live_fetch_allowed():
            return {}
        try:
            resp = requests.get(SNAP_COUNTS_URL.format(season=season), timeout=SNAP_TIMEOUT)
            if resp.status_code == 404:
                cache_utils.cache_set(cache_key, [])
                return {}
            resp.raise_for_status()
        except requests.RequestException:
            return {}
        rows = [
            {f: r.get(f) for f in SNAP_FIELDS}
            for r in csv.DictReader(io.StringIO(resp.text))
            if r.get("game_type", "REG") == "REG" and r.get("position", "WR") in SNAP_POSITIONS
        ]
        cache_utils.cache_set(cache_key, rows)

    by_game = {}
    for r in rows or []:
        try:
            snaps = sum(float(r.get(f) or 0) for f in ("offense_snaps", "defense_snaps", "st_snaps"))
        except (TypeError, ValueError):
            continue
        players = by_game.setdefault(r.get("game_id"), {})
        name = _normalize_name(r.get("player"))
        players[name] = max(snaps, players.get(name, 0.0))
    return by_game


_NAME_TO_ABBR = {name: abbr for abbr, name in nflverse_client.TEAM_ABBR_TO_NAME.items()}


def _played_in_game(results, row, week):
    """True / False if snap counts for this game are posted (any snap
    counts as playing, the books' rule), None if they aren't yet."""
    if results["snap_counts"] is None:
        try:
            results["snap_counts"] = _fetch_snap_counts(results["season"])
        except Exception:  # noqa: BLE001
            results["snap_counts"] = {}
    away, home = _NAME_TO_ABBR.get(row["away_team"]), _NAME_TO_ABBR.get(row["home_team"])
    if not away or not home:
        return None
    game = results["snap_counts"].get(f"{results['season']}_{week:02d}_{away}_{home}")
    if not game:
        return None
    return game.get(_normalize_name(row["player"]), 0) > 0


def _settle_player_prop(row, results, now=None, explain=None):
    """(status, actual) or (None, None) to leave it pending. Requires the
    game's final score AND its posted box score, then:
      - the player's stat row for that game -> graded on the line (exactly
        on the line is a push -> void);
      - no stat row, but the player is known to nflverse -> he either
        didn't play (void, as books do) or played without recording a stat
        (graded at 0) -- decided by snap counts, with a void fallback after
        DNP_VOID_FALLBACK_HOURS if they never post;
      - a player nflverse has never heard of stays pending: that's a name
        mismatch, and voiding on a guess would be worse than waiting."""
    from app.player_props import MARKET_CONFIG  # local import: avoids a module import cycle

    field_info = MARKET_CONFIG.get(row["stat_category"])
    if not field_info or row.get("line") is None or row.get("side") not in ("Over", "Under"):
        return None, None
    field = field_info[0]

    if (row["home_team"], row["away_team"]) not in results["completed_games"]:
        return None, None  # no final score yet
    week = _box_score_week(results, row)
    if week is None:
        return None, None  # box score not posted yet
    rows = nflverse_client.lookup_player(results["player_index"], row["player"])
    if not rows:
        return None, None  # unknown name -- never void on uncertainty

    target_teams = _team_pair(row["home_team"], row["away_team"])
    match = None
    for r, w in rows:
        if w < 1.0:  # only this-season games, not last year's blended-in rows
            continue
        team_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("team"))
        opp_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("opponent_team"))
        if _team_pair(team_full, opp_full) == target_teams and _row_week(r) == week:
            match = r
            break

    if match is not None:
        try:
            actual = float(match.get(field) or 0)
        except (TypeError, ValueError):
            return None, None
        return _grade_prop(row, actual), actual

    played = _played_in_game(results, row, week)
    if played is True:
        if explain is not None:
            explain["reason"] = "played (snap counts) without recording a stat -> graded at 0"
        return _grade_prop(row, 0.0), 0.0
    if played is False:
        if explain is not None:
            explain["reason"] = "did not play (absent from posted snap counts) -> void"
        return "void", None
    commence = _parse_commence(row.get("commence_time"))
    now = now or dt.datetime.now(dt.timezone.utc)
    if commence is None or now - commence < dt.timedelta(hours=DNP_VOID_FALLBACK_HOURS):
        return None, None  # give snap counts time to post
    if explain is not None:
        explain["reason"] = (
            f"no box-score row in a final, posted game; snap counts unavailable "
            f"{DNP_VOID_FALLBACK_HOURS}h+ after kickoff -> void (did not play)"
        )
    return "void", None


def _grade_row(row, results_by_season, now, explain=None):
    """(status, actual) for one prediction-shaped dict, (None, None) if it
    can't be settled yet. Shared by prediction rows and ticket legs so both
    are graded by exactly the same rules."""
    season, _week = season_week_for_kickoff(row.get("commence_time"))
    season = season or row.get("season")
    if not season:
        return None, None
    if season not in results_by_season:
        results_by_season[season] = _load_season_results(season)
    results = results_by_season[season]
    if (row.get("market") or "").startswith("Player Prop"):
        return _settle_player_prop(row, results, now=now, explain=explain)
    return _settle_game_market(row, results["completed_games"]), None


def compute_settlements(pending_rows, now=None, results_by_season=None, explanations=None):
    """{row id: update payload} for every pending row that can be settled
    now -- pure computation plus cached-data reads, NO writes. settle_pending
    writes the result; scripts/backfill_tracking.py prints it, so a dry run
    shows exactly what the live code would do. `explanations`, if given, is
    filled with {row id: reason} for the non-obvious outcomes (DNP voids)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=SETTLEMENT_DELAY_HOURS)
    results_by_season = {} if results_by_season is None else results_by_season
    settled_at = now.replace(tzinfo=None).isoformat()
    updates = {}
    for row in pending_rows:
        commence = _parse_commence(row.get("commence_time"))
        if not commence or commence > cutoff:
            continue
        explain = {} if explanations is not None else None
        status, actual = _grade_row(row, results_by_season, now, explain=explain)
        if status is None:
            continue  # data not posted yet -- leave pending, retry on a later refresh
        updates[row["id"]] = {"status": status, "actual_value": actual, "settled_at": settled_at}
        if explain:
            explanations[row["id"]] = explain["reason"]
    return updates


def settle_pending():
    """Attempts to settle every pending prediction whose game started more
    than SETTLEMENT_DELAY_HOURS ago, then every pending ticket. Safe to call
    on every page load -- anything without final data available yet is
    simply left pending and retried next time. Returns how many predictions
    were newly settled."""
    try:
        pending = _select_all(lambda: _sb().table(TABLE).select("*").eq("status", "pending"))
    except Exception:  # noqa: BLE001
        pending = []

    # Work out which rows actually settle (pure computation, no I/O) before
    # writing anything, THEN fire the writes concurrently -- same fix as
    # capture_closing_lines: sequential one-row-at-a-time updates were a
    # major contributor to page load time whenever several bets settled at
    # once (e.g. right after a Sunday slate finishes).
    results_by_season = {}
    settled = _update_rows(TABLE, compute_settlements(pending, results_by_season=results_by_season))
    try:
        settle_tickets(results_by_season=results_by_season)
    except Exception:  # noqa: BLE001
        pass
    return settled


# --- tickets -----------------------------------------------------------------

def _leg_key_tuple(leg):
    return (leg.get("home_team"), leg.get("away_team"), leg.get("commence_time"), leg.get("market"), leg.get("selection"))


def ticket_status(leg_statuses):
    """How a sportsbook grades a parlay: lost as soon as any leg loses; a
    void (push / DNP) leg drops out and the ticket is re-priced on the
    rest; won once every remaining leg has won; void if every leg was. None
    while anything is still undecided."""
    if any(s == "miss" for s in leg_statuses):
        return "miss"
    if any(s not in ("hit", "void") for s in leg_statuses):
        return None
    if all(s == "void" for s in leg_statuses):
        return "void"
    return "hit"


def compute_ticket_settlements(pending_tickets, leg_rows, now=None, results_by_season=None):
    """{ticket id: update payload}, pure (no writes). A leg's status comes
    from its nfl_predictions row -- the same result the Track Record shows
    for it -- and only a leg with NO row at all is graded directly from the
    copy stored on the ticket, by the same rules."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=SETTLEMENT_DELAY_HOURS)
    results_by_season = {} if results_by_season is None else results_by_season
    status_by_key = {_leg_key_tuple(r): r.get("status") for r in leg_rows}
    settled_at = now.replace(tzinfo=None).isoformat()
    updates = {}
    for ticket in pending_tickets:
        legs = ticket.get("legs") or []
        if not legs:
            continue
        statuses = []
        for leg in legs:
            status = status_by_key.get(_leg_key_tuple(leg))
            if status is None:
                commence = _parse_commence(leg.get("commence_time"))
                if commence and commence <= cutoff:
                    status, _actual = _grade_row(leg, results_by_season, now)
            statuses.append(status or "pending")
        result = ticket_status(statuses)
        if result is None:
            continue
        payload = {"status": result, "settled_at": settled_at, "settled_decimal_odds": None}
        if result == "hit":
            # Paid at the ticket's own leg prices, void legs removed.
            odds = 1.0
            for leg, status in zip(legs, statuses):
                if status == "hit":
                    odds *= float(leg.get("decimal_odds") or 1.0)
            payload["settled_decimal_odds"] = odds
        updates[ticket["id"]] = payload
    return updates


def _pending_tickets_and_leg_rows():
    """(pending tickets, the nfl_predictions rows their legs could match),
    or (None, None) if the tickets table doesn't exist yet."""
    global _tickets_table_missing
    if _tickets_table_missing:
        return None, None
    try:
        pending = _select_all(lambda: _sb().table(TICKETS_TABLE).select("*").eq("status", "pending"))
    except Exception as e:  # noqa: BLE001
        if _is_missing_table(e):
            _tickets_table_missing = True
        return None, None
    if not pending:
        return [], []
    earliest = min(
        (leg.get("commence_time") for t in pending for leg in (t.get("legs") or []) if leg.get("commence_time")),
        default=None,
    )
    if earliest is None:
        return pending, []
    leg_rows = _select_all(
        lambda: _sb().table(TABLE)
        .select("id,home_team,away_team,commence_time,market,selection,status")
        .gte("commence_time", earliest)
    )
    return pending, leg_rows


def settle_tickets(results_by_season=None):
    """Settles pending tickets from their legs' results. Runs at the end of
    settle_pending(), after the leg rows themselves were written. Returns
    how many tickets were newly settled; 0 (never an error) when the
    nfl_tickets table hasn't been created yet."""
    try:
        pending, leg_rows = _pending_tickets_and_leg_rows()
    except Exception:  # noqa: BLE001
        return 0
    if not pending:
        return 0
    updates = compute_ticket_settlements(pending, leg_rows, results_by_season=results_by_season)
    return _update_rows(TICKETS_TABLE, updates)


# --- closing lines -------------------------------------------------------------

def _spread_team(selection):
    m = re.match(r"^(.*) [+-]?\d+(?:\.\d+)?$", selection)
    return m.group(1) if m else selection


def _total_side(selection):
    m = re.match(r"^(Over|Under) ", selection)
    return m.group(1) if m else None


def _selection_line(market, selection):
    """The number in a selection string: a spread's points ("Team -3.5" ->
    -3.5), a total's or a prop's line ("Over 44.5", "X Over 6.5
    Receptions" -> 44.5 / 6.5). None for a moneyline."""
    if not selection or market == "Moneyline":
        return None
    if market == "Spread":
        m = re.match(r"^.* ([+-]?\d+(?:\.\d+)?)$", selection)
    else:
        m = re.search(r"(?:^| )(?:Over|Under) (\d+(?:\.\d+)?)(?: |$)", selection)
    return float(m.group(1)) if m else None


def _closest_live_leg(row, live_matches):
    """The live leg to call "closing" for a row: the exact same bet if it's
    still offered, otherwise the nearest line (best price on a tie). This
    used to take the best PRICE across every line, which for a prop picks
    the worst line -- Over 7.5 at +150 standing in for Over 6.5 at -130 --
    and turned the CLV comparison into a comparison of two different bets."""
    exact = [l for l in live_matches if l["selection"] == row["selection"]]
    if exact:
        return max(exact, key=lambda l: l["decimal_odds"])
    ours = _selection_line(row["market"], row["selection"])
    if ours is None:
        return max(live_matches, key=lambda l: l["decimal_odds"])

    def _distance(l):
        line = _selection_line(row["market"], l["selection"])
        return abs(line - ours) if line is not None else float("inf")

    return min(live_matches, key=lambda l: (_distance(l), -l["decimal_odds"]))


def closing_fair_prob(row, close):
    """The market's closing fair probability for OUR exact bet -- our line,
    our side -- from the live leg captured as closing:
      - same selection still offered: that leg's book_fair_prob (the
        consensus, devigged across every book);
      - a prop whose line moved: the closing consensus distribution
        (market_center/sigma/log_space on the leg, see player_props) priced
        at OUR line -- the same pricing the original leg used;
      - a spread/total whose line moved: None. Converting a probability
        across football's key numbers (3, 7, 10) isn't something a normal
        curve does honestly, and a wrong number here would be worse than
        none."""
    if close["selection"] == row["selection"]:
        return close.get("book_fair_prob")
    if not row.get("player") or row.get("line") is None or row.get("side") not in ("Over", "Under"):
        return None
    center, sigma, log_space = close.get("market_center"), close.get("sigma"), close.get("log_space")
    if center is None or not sigma or log_space is None:
        return None
    from app.player_props import _prob_under  # local import: avoids a module import cycle
    try:
        p_under = _prob_under(float(row["line"]), center, sigma, bool(log_space))
    except (TypeError, ValueError):
        return None
    return round(p_under if row["side"] == "Under" else 1 - p_under, 4)


def capture_closing_lines(pool):
    """Best-effort closing-line capture, reusing the SAME live odds pool the
    page just fetched for its own suggestions -- zero extra API calls. Every
    page load before a game's kickoff overwrites the closing fields with
    whatever is live right now; once kickoff passes, this stops touching
    that row, so whatever was captured on the LAST page load before kickoff
    is what sticks as "closing." This is an approximation of a true
    closing-line snapshot (which would need to poll continuously right up to
    kickoff) -- it's only as fresh as how often the app happens to be opened
    before a given game starts. A bet whose game starts before the app is
    ever reopened after being suggested just never gets a closing line,
    which get_track_record() accounts for by only including captured rows.

    Matches the same bet even if its line has moved since suggestion time
    (moneyline by team, spread/total by side, props by player+stat+side),
    preferring the exact line and otherwise the nearest one (see
    _closest_live_leg); closing_fair_prob is always for OUR line.
    """
    now = dt.datetime.now(dt.timezone.utc)
    try:
        # Only games that haven't kicked off can still change; anything
        # older is frozen, so there's no reason to read it at all.
        candidates_rows = _select_all(
            lambda: _sb().table(TABLE).select("*").gt("commence_time", _utc_iso_z(now))
        )
    except Exception:  # noqa: BLE001
        return 0
    if not candidates_rows:
        return 0

    by_matchup_market = defaultdict(list)
    for leg in pool:
        by_matchup_market[(leg["matchup"], leg["market"])].append(leg)

    # Figure out (row id -> update payload) for every row that needs one --
    # pure Python, no I/O -- THEN fire off the actual writes concurrently.
    # This used to update one row at a time sequentially; with 30-40+
    # pending predictions typical mid-week, that alone was the single
    # biggest contributor to page load time (measured: ~5.7s of a ~10s
    # load), since it ran on every request, not just an explicit refresh.
    to_update = {}
    for row in candidates_rows:
        commence = _parse_commence(row["commence_time"])
        if commence and commence < now:
            continue  # kickoff has passed -- freeze whatever was last captured
        created = _parse_commence(row.get("created_at"))
        if created and now - created < CLOSE_MIN_GAP:
            continue  # suggested in this very refresh -- its "close" would just be its own price

        matchup = f"{row['away_team']} @ {row['home_team']}"
        if row["player"]:
            live_matches = [
                l for l in pool
                if l["matchup"] == matchup
                and l.get("player") == row["player"]
                and l.get("stat_category") == row["stat_category"]
                and l.get("side") == row["side"]
            ]
        elif row["market"] == "Moneyline":
            live_matches = [
                l for l in by_matchup_market.get((matchup, "Moneyline"), [])
                if l["selection"] == row["selection"]
            ]
        elif row["market"] == "Spread":
            team = _spread_team(row["selection"])
            live_matches = [
                l for l in by_matchup_market.get((matchup, "Spread"), [])
                if _spread_team(l["selection"]) == team
            ]
        elif row["market"] == "Total":
            side = _total_side(row["selection"])
            live_matches = [
                l for l in by_matchup_market.get((matchup, "Total"), [])
                if _total_side(l["selection"]) == side
            ]
        else:
            live_matches = []

        if not live_matches:
            continue
        close = _closest_live_leg(row, live_matches)
        to_update[row["id"]] = {
            "closing_decimal_odds": close["decimal_odds"],
            "closing_american_odds": close["american_odds"],
            "closing_selection": close["selection"],
            "closing_captured_at": now.isoformat(),
            "closing_fair_prob": closing_fair_prob(row, close),
        }

    return _update_rows(TABLE, to_update)


# --- reporting -----------------------------------------------------------------

def _section_version(section):
    """'best_odds_parlay' (no suffix) was the original v1 model; later rows
    carry an explicit '_v2', '_v3', ... suffix."""
    if not section or "_v" not in section:
        return "v1"
    return section.rsplit("_", 1)[1]


def _line_moved_our_way(row):
    """For a bet whose closing line differs from ours: did the market move
    TOWARD our side? (We took Over 43.5, it closed 44.5 -- yes; we took
    +3.5, it closed +2.5 -- yes.) None if either line can't be read."""
    ours = _selection_line(row["market"], row["selection"])
    closing = _selection_line(row["market"], row.get("closing_selection"))
    if ours is None or closing is None or ours == closing:
        return None
    if row["market"] == "Spread":
        return ours > closing
    side = row.get("side") or _total_side(row["selection"])
    if side == "Over":
        return closing > ours
    if side == "Under":
        return closing < ours
    return None


def _ticket_record():
    """Summary of whole-ticket results for the current model, or None if
    the nfl_tickets table doesn't exist yet."""
    global _tickets_table_missing
    if _tickets_table_missing:
        return None
    try:
        rows = _select_all(lambda: _sb().table(TICKETS_TABLE).select("*"))
    except Exception as e:  # noqa: BLE001
        if _is_missing_table(e):
            _tickets_table_missing = True
        return None
    rows = [r for r in rows if _section_version(r.get("section")) == MODEL_VERSION]
    settled = [r for r in rows if r["status"] in ("hit", "miss")]
    n = len(settled)
    hits = [r for r in settled if r["status"] == "hit"]
    staked = sum(r.get("stake") or 0 for r in settled)
    returned = sum((r.get("stake") or 0) * (r.get("settled_decimal_odds") or r.get("decimal_odds") or 0) for r in hits)
    with_prob = [r for r in settled if r.get("combined_prob") is not None]
    return {
        "logged": len(rows),
        "settled": n,
        "hits": len(hits),
        "pending": sum(1 for r in rows if r["status"] == "pending"),
        "void": sum(1 for r in rows if r["status"] == "void"),
        "avg_shown_prob": (sum(r["combined_prob"] for r in with_prob) / len(with_prob)) if with_prob else None,
        # With ~0.5% tickets, "expected hits so far" reads far more honestly
        # than comparing two tiny percentages.
        "expected_hits": sum(r["combined_prob"] for r in with_prob) if with_prob else None,
        "actual_hit_rate": (len(hits) / n) if n else None,
        "avg_shown_ev_pct": (sum(r["ev_pct"] for r in settled if r.get("ev_pct") is not None) / n) if n else None,
        "staked": staked,
        "returned": returned,
        "roi_pct": ((returned - staked) / staked * 100) if staked else None,
        "confident": n >= MIN_SAMPLE_FOR_TICKET_CONFIDENCE,
        "recent": sorted(settled, key=lambda r: r.get("settled_at") or "", reverse=True)[:15],
    }


def get_track_record(stake=5.0):
    """Calibration/hit-rate/ROI for suggestions made by the CURRENT model
    version only, plus a separate summary line for any earlier versions --
    so a model change gets judged on its own record rather than inheriting
    (or hiding behind) the old one's. Rows logged after their game had
    already kicked off are left out of every number (see
    _logged_before_kickoff) and counted separately."""
    try:
        all_rows = _select_all(lambda: _sb().table(TABLE).select("*"))
    except Exception:  # noqa: BLE001
        all_rows = []

    in_play = [r for r in all_rows if not _logged_before_kickoff(r)]
    pregame = [r for r in all_rows if _logged_before_kickoff(r)]
    current = [r for r in pregame if _section_version(r.get("section")) == MODEL_VERSION]
    older_rows = [r for r in pregame if _section_version(r.get("section")) != MODEL_VERSION]

    settled = [r for r in current if r["status"] in ("hit", "miss")]
    pending_count = sum(1 for r in current if r["status"] == "pending")
    void_count = sum(1 for r in current if r["status"] == "void")
    older = [r for r in older_rows if r["status"] in ("hit", "miss")]
    older_summary = None
    if older:
        older_summary = {
            "n": len(older),
            "predicted_avg": sum(r["model_prob"] for r in older) / len(older),
            "actual_hit_rate": sum(1 for r in older if r["status"] == "hit") / len(older),
            "pending": sum(1 for r in older_rows if r["status"] == "pending"),
        }

    total = len(settled)
    hits = sum(1 for r in settled if r["status"] == "hit")

    buckets = []
    for lo, hi in CALIBRATION_BUCKETS:
        in_bucket = [r for r in settled if lo <= r["model_prob"] < hi]
        if not in_bucket:
            continue
        buckets.append({
            "range": f"{int(lo * 100)}-{int(min(hi, 1.0) * 100)}%",
            "predicted_avg": sum(r["model_prob"] for r in in_bucket) / len(in_bucket),
            "actual_hit_rate": sum(1 for r in in_bucket if r["status"] == "hit") / len(in_bucket),
            "n": len(in_bucket),
        })

    total_staked = total * stake
    total_returned = sum(stake * r["decimal_odds"] for r in settled if r["status"] == "hit")
    roi_pct = ((total_returned - total_staked) / total_staked * 100) if total_staked else None

    # CLV (closing line value): did we get a better price than the market's
    # own closing number? Sharp-betting convention treats this as a FASTER,
    # more reliable skill signal than win/loss -- it doesn't need to wait for
    # the game to be played, just for the line to move (or not) before
    # kickoff. clv_pct here is simply "% better decimal odds than closing";
    # positive means our price was better (the market moved toward us after
    # we took it), negative means the market moved away from us. Only
    # meaningful when the closing price is for the SAME line -- comparing
    # Over 6.5's price to Over 7.5's says nothing -- so a bet whose line
    # moved is judged instead by which way it moved, and by closing EV.
    # A close captured in the same refresh that suggested the bet isn't a
    # close at all (see _close_after_suggestion) and is left out.
    any_close = [r for r in current if r.get("closing_decimal_odds")]
    with_close = [r for r in any_close if _close_after_suggestion(r)]
    clv_rows = []
    moved = []
    for r in with_close:
        # Per-row closing EV for the table (None before the migration adds
        # the column, or when it couldn't be priced honestly).
        fair = r.get("closing_fair_prob")
        row = {**dict(r), "closing_ev_pts": ((fair - 1 / r["decimal_odds"]) * 100) if fair is not None else None}
        if r.get("closing_selection") and r["closing_selection"] != r["selection"]:
            moved.append({**row, "clv_pct": None})
            continue
        row["clv_pct"] = (r["decimal_odds"] - r["closing_decimal_odds"]) / r["closing_decimal_odds"] * 100
        clv_rows.append(row)
    clv_n = len(clv_rows)
    avg_clv_pct = (sum(r["clv_pct"] for r in clv_rows) / clv_n) if clv_n else None
    positive_clv_pct = (sum(1 for r in clv_rows if r["clv_pct"] > 0) / clv_n * 100) if clv_n else None
    moved_directions = [d for d in (_line_moved_our_way(r) for r in moved) if d is not None]

    # Closing EV: the market's closing fair probability for our exact bet
    # minus the breakeven probability of the price we took. Every captured
    # bet contributes a real number (not a 0/1 outcome), which is why a few
    # hundred of these say more about whether the picks were good than the
    # hit rate will for a long time.
    with_fair = [r for r in with_close if r.get("closing_fair_prob") is not None]
    closing_ev = [r["closing_fair_prob"] - 1 / r["decimal_odds"] for r in with_fair]
    closing_ev_n = len(closing_ev)

    return {
        "model_version": MODEL_VERSION,
        "older_models": older_summary,
        "total_settled": total,
        "pending": pending_count,
        "void": void_count,
        "in_play_excluded": len(in_play),
        "overall_hit_rate": (hits / total) if total else None,
        "buckets": buckets,
        "roi_pct": roi_pct,
        "stake": stake,
        "confident": total >= MIN_SAMPLE_FOR_CONFIDENCE,
        "recent": sorted(settled, key=lambda r: r["settled_at"] or "", reverse=True)[:25],
        "clv_n": clv_n,
        "avg_clv_pct": avg_clv_pct,
        "positive_clv_pct": positive_clv_pct,
        "clv_confident": clv_n >= MIN_SAMPLE_FOR_CLV_CONFIDENCE,
        "clv_recent": sorted(
            clv_rows + moved, key=lambda r: r["closing_captured_at"] or "", reverse=True,
        )[:25],
        "same_refresh_close_n": len(any_close) - len(with_close),
        "moved_n": len(moved),
        "moved_our_way_pct": (sum(moved_directions) / len(moved_directions) * 100) if moved_directions else None,
        "closing_ev_n": closing_ev_n,
        "avg_closing_ev_pts": (sum(closing_ev) / closing_ev_n * 100) if closing_ev_n else None,
        "positive_closing_ev_pct": (sum(1 for x in closing_ev if x > 0) / closing_ev_n * 100) if closing_ev_n else None,
        "tickets": _ticket_record(),
    }
