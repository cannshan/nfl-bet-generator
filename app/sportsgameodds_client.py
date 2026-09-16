"""Free fallback odds source (sportsgameodds.com), used ONLY when The Odds
API is unavailable -- quota exhausted (429) or the key gets rejected (401).
The Odds API stays primary always; this never runs otherwise.

Converts sportsgameodds' response shape into the EXACT shape The Odds API's
own bulk-odds endpoint returns (a list of games, each with `bookmakers` ->
`markets` -> `outcomes`), so value_finder.analyze_games() and the rest of
the pipeline need zero changes -- they can't tell which provider actually
served the data.

Free tier covers 9 real US sportsbooks (DraftKings, FanDuel, BetMGM,
Caesars, ESPN BET, Bovada, Unibet, PointsBet, William Hill) with real
moneyline/spread/total odds AND player props, all from ONE events call
(unlike The Odds API's separate bulk + per-event-props design) -- verified
against the real API with a real key before writing this, not just their
docs (their free-tier response includes a truncation notice on bookmaker
coverage that isn't mentioned up front).
"""
import datetime as dt
import requests
from app.config import SPORTSGAMEODDS_API_KEY
from app.cache_utils import cache_get, cache_set, live_fetch_allowed

BASE_URL = "https://api.sportsgameodds.com/v2"
TIMEOUT = 20
CACHE_TTL_SECONDS = 10 * 60

# sportsgameodds' player-prop statID -> this app's own player_props.py market
# key (see player_props.MARKET_CONFIG). Verified against the real API with a
# real key (not just docs) that all 7 markets this app uses have a match.
STAT_ID_TO_MARKET_KEY = {
    "passing_yards": "player_pass_yds",
    "passing_attempts": "player_pass_attempts",
    "passing_completions": "player_pass_completions",
    "passing_touchdowns": "player_pass_tds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
    "receiving_receptions": "player_receptions",
}


def is_configured():
    return bool(SPORTSGAMEODDS_API_KEY)


def _normalize_time(iso_str):
    """"2026-09-18T00:15:00.000Z" -> "2026-09-18T00:15:00Z" -- matches the
    plain-seconds ISO format The Odds API uses (and _parse_commence
    elsewhere expects) rather than sportsgameodds' milliseconds."""
    if not iso_str:
        return None
    return iso_str.split(".")[0] + "Z" if "." in iso_str else iso_str


def _fetch_events():
    cache_key = "sgo_nfl_events"
    cached = cache_get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed() or not is_configured():
        return []
    try:
        resp = requests.get(
            f"{BASE_URL}/events",
            params={
                "apiKey": SPORTSGAMEODDS_API_KEY,
                "leagueID": "NFL",
                "oddsAvailable": "true",
                "limit": 50,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []
    events = data.get("data", [])
    cache_set(cache_key, events)
    return events


def _bookmaker_price(market_obj, bookmaker):
    if not market_obj:
        return None
    entry = market_obj.get("byBookmaker", {}).get(bookmaker)
    if not entry or entry.get("available") is False:
        return None
    try:
        return int(entry["odds"])
    except (KeyError, TypeError, ValueError):
        return None


def get_game_odds():
    """Returns data in The Odds API's own bulk /odds response shape, built
    from sportsgameodds' moneyline/spread/total markets."""
    games = []
    for event in _fetch_events():
        teams = event.get("teams", {})
        home_name = teams.get("home", {}).get("names", {}).get("long")
        away_name = teams.get("away", {}).get("names", {}).get("long")
        if not home_name or not away_name:
            continue
        commence = _normalize_time(event.get("status", {}).get("startsAt"))
        odds = event.get("odds", {})

        ml_home = odds.get("points-home-game-ml-home")
        ml_away = odds.get("points-away-game-ml-away")
        sp_home = odds.get("points-home-game-sp-home")
        sp_away = odds.get("points-away-game-sp-away")
        ou_over = odds.get("points-all-game-ou-over")
        ou_under = odds.get("points-all-game-ou-under")

        bookmaker_names = set()
        for m in (ml_home, ml_away, sp_home, sp_away, ou_over, ou_under):
            if m:
                bookmaker_names |= set(m.get("byBookmaker", {}).keys())

        bookmakers = []
        for bk in bookmaker_names:
            markets = []

            ml_h, ml_a = _bookmaker_price(ml_home, bk), _bookmaker_price(ml_away, bk)
            if ml_h is not None and ml_a is not None:
                markets.append({"key": "h2h", "outcomes": [
                    {"name": home_name, "price": ml_h},
                    {"name": away_name, "price": ml_a},
                ]})

            sp_h, sp_a = _bookmaker_price(sp_home, bk), _bookmaker_price(sp_away, bk)
            sp_h_pt = (sp_home or {}).get("byBookmaker", {}).get(bk, {}).get("spread")
            sp_a_pt = (sp_away or {}).get("byBookmaker", {}).get(bk, {}).get("spread")
            if None not in (sp_h, sp_a, sp_h_pt, sp_a_pt):
                try:
                    markets.append({"key": "spreads", "outcomes": [
                        {"name": home_name, "price": sp_h, "point": float(sp_h_pt)},
                        {"name": away_name, "price": sp_a, "point": float(sp_a_pt)},
                    ]})
                except (TypeError, ValueError):
                    pass

            ou_o, ou_u = _bookmaker_price(ou_over, bk), _bookmaker_price(ou_under, bk)
            ou_o_pt = (ou_over or {}).get("byBookmaker", {}).get(bk, {}).get("overUnder")
            ou_u_pt = (ou_under or {}).get("byBookmaker", {}).get(bk, {}).get("overUnder")
            if None not in (ou_o, ou_u, ou_o_pt):
                try:
                    markets.append({"key": "totals", "outcomes": [
                        {"name": "Over", "price": ou_o, "point": float(ou_o_pt)},
                        {"name": "Under", "price": ou_u, "point": float(ou_u_pt or ou_o_pt)},
                    ]})
                except (TypeError, ValueError):
                    pass

            if markets:
                bookmakers.append({"title": bk, "markets": markets})

        if bookmakers:
            games.append({
                "home_team": home_name, "away_team": away_name,
                "commence_time": commence, "bookmakers": bookmakers,
            })

    return games


def get_events_list():
    """Drop-in replacement for The Odds API's get_events() -- needs an "id"
    field (player_props.py looks up event odds by it) plus flat
    "home_team"/"away_team" strings matching The Odds API's own event shape
    (used as a fallback-matching key in get_event_props when the id
    namespaces don't line up -- see there)."""
    out = []
    for e in _fetch_events():
        if not e.get("eventID"):
            continue
        teams = e.get("teams", {})
        out.append({
            "id": e["eventID"],
            "home_team": teams.get("home", {}).get("names", {}).get("long"),
            "away_team": teams.get("away", {}).get("names", {}).get("long"),
            "commence_time": _normalize_time(e.get("status", {}).get("startsAt")),
            **e,
        })
    return out


def get_event_props(event_id, markets, home_team=None, away_team=None):
    """Drop-in replacement for The Odds API's get_event_odds(event_id, markets)
    -- looks up the same cached event list get_events_list() already pulled
    (no extra API call) and converts its player-prop odds into the exact
    per-event shape player_props._consolidate_best_prices() expects:
    {"home_team", "away_team", "commence_time", "bookmakers": [{"title",
    "markets": [{"key", "outcomes": [{"description", "name", "point", "price"}]}]}]}.
    `markets` is the same comma-separated The-Odds-API-style market key
    string player_props.py already builds (DEFAULT_MARKETS) -- only those
    markets are included.

    The Odds API's /events endpoint is free (no quota cost), so it's common
    for get_events() to still succeed with REAL Odds-API event ids even
    while /odds (quota-limited) is failing -- meaning event_id here is often
    in a completely different ID namespace than sportsgameodds' own events.
    Falls back to matching by team names (both providers use the same full
    team-name strings, e.g. "Buffalo Bills") when a direct id match fails.
    """
    wanted = set(markets.split(","))
    all_events = _fetch_events()
    event = next((e for e in all_events if e.get("eventID") == event_id), None)
    if not event and home_team and away_team:
        event = next(
            (e for e in all_events
             if e.get("teams", {}).get("home", {}).get("names", {}).get("long") == home_team
             and e.get("teams", {}).get("away", {}).get("names", {}).get("long") == away_team),
            None,
        )
    if not event:
        return {}

    teams = event.get("teams", {})
    home_name = teams.get("home", {}).get("names", {}).get("long")
    away_name = teams.get("away", {}).get("names", {}).get("long")
    commence = _normalize_time(event.get("status", {}).get("startsAt"))
    players = event.get("players", {})
    odds = event.get("odds", {})

    # (market_key, player_name) -> {"Over": odd_obj, "Under": odd_obj}
    grouped = {}
    for obj in odds.values():
        market_key = STAT_ID_TO_MARKET_KEY.get(obj.get("statID"))
        player_id = obj.get("playerID")
        side = {"over": "Over", "under": "Under"}.get(obj.get("sideID"))
        if not market_key or market_key not in wanted or not player_id or not side:
            continue
        player_name = players.get(player_id, {}).get("name")
        if not player_name:
            continue
        grouped.setdefault((market_key, player_name), {})[side] = obj

    # bookmaker -> market_key -> [outcomes]
    by_bookmaker = {}
    for (market_key, player_name), sides in grouped.items():
        over_obj, under_obj = sides.get("Over"), sides.get("Under")
        bk_names = set()
        for obj in (over_obj, under_obj):
            if obj:
                bk_names |= set(obj.get("byBookmaker", {}).keys())
        for bk in bk_names:
            for obj, side in ((over_obj, "Over"), (under_obj, "Under")):
                if not obj:
                    continue
                entry = obj.get("byBookmaker", {}).get(bk)
                if not entry or entry.get("available") is False:
                    continue
                try:
                    outcome = {
                        "description": player_name, "name": side,
                        "point": float(entry["overUnder"]), "price": int(entry["odds"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                by_bookmaker.setdefault(bk, {}).setdefault(market_key, []).append(outcome)

    bookmakers = [
        {"title": bk, "markets": [{"key": mk, "outcomes": outs} for mk, outs in mkts.items()]}
        for bk, mkts in by_bookmaker.items()
    ]
    if not bookmakers:
        return {}
    return {"home_team": home_name, "away_team": away_name, "commence_time": commence, "bookmakers": bookmakers}
