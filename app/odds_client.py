import requests
from app.config import ODDS_API_KEY, ODDS_API_BASE, ODDS_CACHE_TTL_SECONDS
from app.cache_utils import cache_get, cache_set, live_fetch_allowed
from app import sportsgameodds_client

SPORT_KEY = "americanfootball_nfl"


class OddsApiError(Exception):
    pass


def get_odds(markets="h2h,spreads,totals", regions="us"):
    """Fetch current NFL odds across books. Cached to conserve the free-tier
    quota. Falls back to sportsgameodds_client (a separate free provider)
    when The Odds API's key is missing/rejected or its quota is exhausted --
    that failure mode is common enough (a single free-tier month is only
    500 requests) to be worth a real fallback rather than just an error
    banner. The Odds API is always tried first; the fallback only engages
    on an actual failure."""
    cache_key = f"odds_{markets}_{regions}"
    cached = cache_get(cache_key, ODDS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return []

    try:
        _require_key()
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        resp = requests.get(url, params=params, timeout=15)
        _handle_errors(resp)
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except OddsApiError as e:
        if not sportsgameodds_client.is_configured():
            raise
        print(f"[odds_client] The Odds API unavailable ({e}) -- falling back to sportsgameodds.com")
        data = sportsgameodds_client.get_game_odds()
        cache_set(cache_key, data)
        return data


def remaining_quota(response_headers):
    return {
        "requests_remaining": response_headers.get("x-requests-remaining"),
        "requests_used": response_headers.get("x-requests-used"),
    }


def _require_key():
    if not ODDS_API_KEY:
        raise OddsApiError(
            "No ODDS_API_KEY configured. Get a free key at https://the-odds-api.com "
            "and add it to your .env file."
        )


def _handle_errors(resp):
    if resp.status_code == 401:
        raise OddsApiError("Odds API rejected the key (401). Check ODDS_API_KEY in your .env file.")
    if resp.status_code == 429:
        raise OddsApiError("Odds API monthly quota exhausted (429). Try again next month or upgrade your plan.")
    resp.raise_for_status()


def get_events():
    """List this week's NFL events (id, teams, commence_time). Free/cheap call,
    needed to look up event ids for the per-event player-props endpoint.
    Falls back to sportsgameodds_client on the same terms as get_odds()."""
    cache_key = "odds_events"
    cached = cache_get(cache_key, ODDS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return []

    try:
        _require_key()
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events",
            params={"apiKey": ODDS_API_KEY},
            timeout=15,
        )
        _handle_errors(resp)
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except OddsApiError as e:
        if not sportsgameodds_client.is_configured():
            raise
        print(f"[odds_client] The Odds API unavailable ({e}) -- falling back to sportsgameodds.com for events")
        data = sportsgameodds_client.get_events_list()
        cache_set(cache_key, data)
        return data


def get_event_odds(event_id, markets, regions="us", home_team=None, away_team=None):
    """Player props (and any other market) live on a per-event endpoint, unlike
    the bulk h2h/spreads/totals endpoint. Each market requested here costs API
    credits, so keep the markets list intentionally small and rely on caching.
    Falls back to sportsgameodds_client on the same terms as get_odds().
    `home_team`/`away_team` are optional but recommended: The Odds API's
    /events endpoint is free (no quota cost), so it commonly still succeeds
    with REAL Odds-API event ids even while this quota-limited endpoint is
    failing -- passing team names lets the fallback match by team instead of
    by id when the id namespaces don't line up (see sportsgameodds_client)."""
    cache_key = f"event_odds_{event_id}_{markets}_{regions}"
    cached = cache_get(cache_key, ODDS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return {}

    try:
        _require_key()
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "american",
            },
            timeout=15,
        )
        _handle_errors(resp)
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except OddsApiError as e:
        if not sportsgameodds_client.is_configured():
            raise
        data = sportsgameodds_client.get_event_props(event_id, markets, home_team, away_team)
        cache_set(cache_key, data)
        return data
