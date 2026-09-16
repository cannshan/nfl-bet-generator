"""Free, no-key weather forecasts (Open-Meteo) for outdoor stadiums, used to
adjust passing/kicking-heavy props for wind and precipitation. Domes and
retractable-roof stadiums (treated as effectively climate-controlled -- roofs
are closed in bad weather in the vast majority of cases) are skipped
entirely rather than guessed at.

This is a well-known real handicapping factor: high wind meaningfully hurts
passing accuracy and distance, heavy rain/snow hurts footing and ball
security for both passing and rushing. The adjustment magnitudes below are
rough, transparent heuristics, not a fitted model.
"""
import datetime as dt
import requests
from app.cache_utils import cache_get, cache_set, live_fetch_allowed

TIMEOUT = 15
CACHE_TTL_SECONDS = 3 * 60 * 60

WIND_THRESHOLD_MPH = 15
HEAVY_WIND_THRESHOLD_MPH = 25
PRECIP_THRESHOLD_IN = 0.05

# Rough multipliers applied to a player's projected mean for the relevant
# stat category. "pass" covers pass_yds/pass_tds/completions/attempts and
# receiving categories (a passing-game problem shows up on both ends);
# "rush" covers rush_yds.
WEATHER_ADJUSTMENTS = {
    "heavy_wind": {"pass": 0.85, "rush": 1.03},
    "windy": {"pass": 0.93, "rush": 1.02},
    "precipitation": {"pass": 0.90, "rush": 0.95},
    "clear": {"pass": 1.0, "rush": 1.0},
}

# Approximate stadium coordinates and roof type. Retractable roofs are
# treated as domes since they're closed in the weather conditions that would
# otherwise matter here.
STADIUMS = {
    "Arizona Cardinals": (33.5276, -112.2626, True),
    "Atlanta Falcons": (33.7554, -84.4008, True),
    "Baltimore Ravens": (39.2780, -76.6227, False),
    "Buffalo Bills": (42.7738, -78.7870, False),
    "Carolina Panthers": (35.2258, -80.8528, False),
    "Chicago Bears": (41.8623, -87.6167, False),
    "Cincinnati Bengals": (39.0955, -84.5160, False),
    "Cleveland Browns": (41.5061, -81.6995, False),
    "Dallas Cowboys": (32.7473, -97.0945, True),
    "Denver Broncos": (39.7439, -105.0201, False),
    "Detroit Lions": (42.3400, -83.0456, True),
    "Green Bay Packers": (44.5013, -88.0622, False),
    "Houston Texans": (29.6847, -95.4107, True),
    "Indianapolis Colts": (39.7601, -86.1639, True),
    "Jacksonville Jaguars": (30.3239, -81.6373, False),
    "Kansas City Chiefs": (39.0489, -94.4839, False),
    "Las Vegas Raiders": (36.0909, -115.1833, True),
    "Los Angeles Chargers": (33.9535, -118.3392, True),
    "Los Angeles Rams": (33.9535, -118.3392, True),
    "Miami Dolphins": (25.9580, -80.2389, False),
    "Minnesota Vikings": (44.9738, -93.2575, True),
    "New England Patriots": (42.0909, -71.2643, False),
    "New Orleans Saints": (29.9509, -90.0815, True),
    "New York Giants": (40.8135, -74.0745, False),
    "New York Jets": (40.8135, -74.0745, False),
    "Philadelphia Eagles": (39.9008, -75.1675, False),
    "Pittsburgh Steelers": (40.4468, -80.0158, False),
    "Seattle Seahawks": (47.5952, -122.3316, False),
    "San Francisco 49ers": (37.4032, -121.9698, False),
    "Tampa Bay Buccaneers": (27.9759, -82.5033, False),
    "Tennessee Titans": (36.1665, -86.7713, False),
    "Washington Commanders": (38.9078, -76.8645, False),
}


def get_game_weather(home_team, commence_time_iso):
    """Returns {"condition", "wind_mph", "precip_in", "dome"} or None if the
    stadium is unknown, it's a dome, or the forecast can't be fetched."""
    stadium = STADIUMS.get(home_team)
    if not stadium:
        return None
    lat, lon, dome = stadium
    if dome:
        return {"condition": "dome", "wind_mph": 0, "precip_in": 0, "dome": True}

    try:
        kickoff = dt.datetime.strptime(commence_time_iso, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None

    cache_key = f"weather_{home_team}_{commence_time_iso}"
    cached = cache_get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return None

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation,windspeed_10m",
                "timezone": "UTC",
                "forecast_days": 16,
                "windspeed_unit": "mph",
                "precipitation_unit": "inch",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None

    target_hour = kickoff.strftime("%Y-%m-%dT%H:00")
    times = data.get("hourly", {}).get("time", [])
    if target_hour not in times:
        return None  # game too far out for this forecast window
    idx = times.index(target_hour)

    wind = data["hourly"]["windspeed_10m"][idx]
    precip = data["hourly"]["precipitation"][idx]

    if precip >= PRECIP_THRESHOLD_IN:
        condition = "precipitation"
    elif wind >= HEAVY_WIND_THRESHOLD_MPH:
        condition = "heavy_wind"
    elif wind >= WIND_THRESHOLD_MPH:
        condition = "windy"
    else:
        condition = "clear"

    result = {"condition": condition, "wind_mph": wind, "precip_in": precip, "dome": False}
    cache_set(cache_key, result)
    return result


def adjustment_factor(weather, category):
    """category is 'pass' or 'rush' (matching player_props.py's MARKET_CONFIG)."""
    if not weather:
        return 1.0
    return WEATHER_ADJUSTMENTS.get(weather["condition"], WEATHER_ADJUSTMENTS["clear"])[category]
