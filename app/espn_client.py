import calendar
import datetime as dt
import requests
from app.config import ESPN_API_BASE
from app.cache_utils import cache_get, cache_set, live_fetch_allowed

TIMEOUT = 15
REGULAR_SEASON_WEEKS = 18

# ESPN's public API returns 400 Bad Request for requests from cloud/datacenter
# IP ranges (confirmed: identical requests work fine from a home network, fail
# from Vercel) unless they look like an ordinary browser -- a realistic
# User-Agent is enough to get past it.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def get_current_scoreboard():
    """Returns the current week's scoreboard, which also tells us season/week.
    This is the one ESPN query that's reliably accurate for 'now' -- the
    year/week query params below are NOT reliable for past seasons (ESPN
    silently ignores an out-of-range year and returns current-season data),
    so historical data is fetched by date range instead.
    """
    cache_key = "espn_current_scoreboard"
    cached = cache_get(cache_key, 60 * 30)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return {}
    resp = requests.get(f"{ESPN_API_BASE}/scoreboard", timeout=TIMEOUT, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    cache_set(cache_key, data)
    return data


def get_current_season_and_week():
    """Prefers ESPN's own "current week" endpoint when it's reachable, but
    falls back to computing season/week from today's date when it isn't --
    ESPN's WAF blocks this specific unparameterized endpoint from some cloud
    hosts (observed: works fine from a home network, 403s from Vercel) even
    though the exact same domain's dated-range queries go through fine with
    a browser-like User-Agent. Pure date arithmetic can't be blocked."""
    try:
        sb = get_current_scoreboard()
        season = sb.get("season", {}).get("year")
        week = sb.get("week", {}).get("number")
        season_type = sb.get("season", {}).get("type", 2)
        if season and week:
            return season, week, season_type
    except requests.RequestException:
        pass
    return _compute_season_and_week_from_date()


def _compute_season_and_week_from_date(today=None):
    today = today or dt.date.today()
    this_year_kickoff = season_kickoff_date(today.year)
    season = today.year if today >= this_year_kickoff else today.year - 1
    kickoff = season_kickoff_date(season)
    week = min(REGULAR_SEASON_WEEKS, max(1, (today - kickoff).days // 7 + 1))
    return season, week, 2


def season_kickoff_date(season_start_year):
    """NFL regular season traditionally opens the Thursday after Labor Day
    (the first Monday of September)."""
    c = calendar.Calendar()
    september = [d for d in c.itermonthdates(season_start_year, 9) if d.month == 9]
    labor_day = next(d for d in september if d.weekday() == 0)  # Monday
    return labor_day + dt.timedelta(days=3)


def get_season_games_to_date(season_start_year, through_date=None):
    """All completed regular-season games for the season starting in
    `season_start_year`. Sourced from nflverse's games.csv (see
    nflverse_client.get_season_games) rather than ESPN's scoreboard `dates=`
    range endpoint -- that endpoint has a much more aggressive rate limit
    than ESPN's other endpoints and this app's own backtesting tripped it
    for an extended period, which isn't tolerable for a function this
    central to the whole rating model. `through_date` is accepted for
    backwards compatibility but unused: nflverse rows simply have no score
    yet for games that haven't been played, which gives the same "to date"
    behavior without needing date arithmetic."""
    from app import nflverse_client
    return nflverse_client.get_season_games(season_start_year)


def get_full_season_games(season_start_year):
    from app import nflverse_client
    return nflverse_client.get_season_games(season_start_year)
