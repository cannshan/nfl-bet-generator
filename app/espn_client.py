import calendar
import datetime as dt
import requests
from app.config import ESPN_API_BASE, RATINGS_CACHE_TTL_SECONDS
from app.cache_utils import cache_get, cache_set

TIMEOUT = 15
REGULAR_SEASON_WEEKS = 18


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
    resp = requests.get(f"{ESPN_API_BASE}/scoreboard", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    cache_set(cache_key, data)
    return data


def get_current_season_and_week():
    sb = get_current_scoreboard()
    season = sb.get("season", {}).get("year")
    week = sb.get("week", {}).get("number")
    season_type = sb.get("season", {}).get("type", 2)
    return season, week, season_type


def season_kickoff_date(season_start_year):
    """NFL regular season traditionally opens the Thursday after Labor Day
    (the first Monday of September)."""
    c = calendar.Calendar()
    september = [d for d in c.itermonthdates(season_start_year, 9) if d.month == 9]
    labor_day = next(d for d in september if d.weekday() == 0)  # Monday
    return labor_day + dt.timedelta(days=3)


def _fetch_date_range(start_date, end_date):
    """Fetch scoreboard events across a date range, one week at a time
    (ESPN's `dates` range param behaves reliably for windows around this size)."""
    cache_key = f"espn_range_{start_date.isoformat()}_{end_date.isoformat()}"
    cached = cache_get(cache_key, RATINGS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    all_events = {}
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + dt.timedelta(days=6), end_date)
        date_param = f"{cursor.strftime('%Y%m%d')}-{window_end.strftime('%Y%m%d')}"
        try:
            resp = requests.get(
                f"{ESPN_API_BASE}/scoreboard",
                params={"dates": date_param, "limit": 100},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            data = {}
        for event in data.get("events", []):
            all_events[event.get("id")] = event
        cursor = window_end + dt.timedelta(days=1)

    events = list(all_events.values())
    cache_set(cache_key, events)
    return events


def parse_completed_games(events):
    """Extract [{home, away, home_score, away_score}] for finished games only."""
    games = []
    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_score = int(home.get("score"))
            away_score = int(away.get("score"))
        except (TypeError, ValueError):
            continue
        games.append(
            {
                "home": home["team"]["displayName"],
                "away": away["team"]["displayName"],
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return games


def get_season_games_to_date(season_start_year, through_date=None):
    """All completed regular-season games for the season starting in
    `season_start_year`, from kickoff through `through_date` (default: today)."""
    kickoff = season_kickoff_date(season_start_year)
    season_end_cap = kickoff + dt.timedelta(weeks=REGULAR_SEASON_WEEKS + 1)
    end = through_date or dt.date.today()
    end = min(end, season_end_cap)
    if end < kickoff:
        return []
    events = _fetch_date_range(kickoff, end)
    return parse_completed_games(events)


def get_full_season_games(season_start_year):
    kickoff = season_kickoff_date(season_start_year)
    end = kickoff + dt.timedelta(weeks=REGULAR_SEASON_WEEKS + 1)
    events = _fetch_date_range(kickoff, end)
    return parse_completed_games(events)
