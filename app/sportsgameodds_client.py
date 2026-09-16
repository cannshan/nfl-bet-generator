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
