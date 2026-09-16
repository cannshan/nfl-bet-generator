"""Live injury reports from ESPN's public API -- one call for the entire
league, including beat-reporter narrative comments (practice participation,
"playing through it" context) alongside the formal designation, plus each
player's position and team directly (no cross-referencing needed).

This is exactly the kind of signal that can quietly swing a bet: a player
listed "Questionable" who barely practiced is a real risk to a prop even if
he ultimately suits up. We use this to drop injured players out of the prop
pool entirely (Out/Doubtful/Injured Reserve) and apply a conservative
discount for "Questionable" rather than pretend the model already knows.
"""
import requests
from app.config import ESPN_API_BASE
from app.cache_utils import cache_get, cache_set
from app.espn_client import HEADERS

TIMEOUT = 15
INJURY_CACHE_TTL_SECONDS = 30 * 60

# Multiplier applied to a Questionable player's projected mean stat -- a
# rough, transparent haircut for reduced/uncertain usage, not a precise model.
QUESTIONABLE_DISCOUNT = 0.85

EXCLUDE_STATUSES = {"Out", "Doubtful", "Injured Reserve"}
OFFENSIVE_SKILL_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def _fetch_raw_entries():
    """One entry per (team, player) report, most recent per player already
    collapsed. Each entry: {team, player, position, status, comment,
    long_comment, date}."""
    cache_key = "espn_league_injuries_raw"
    cached = cache_get(cache_key, INJURY_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        resp = requests.get(f"{ESPN_API_BASE}/injuries", timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    by_player = {}
    for team in data.get("injuries", []):
        for entry in team.get("injuries", []):
            athlete = entry.get("athlete", {})
            name = athlete.get("displayName")
            if not name:
                continue
            date = entry.get("date", "")
            existing = by_player.get(name)
            if existing and existing["date"] >= date:
                continue
            long_comment = entry.get("longComment") or entry.get("shortComment")
            by_player[name] = {
                "player": name,
                "team": athlete.get("team", {}).get("displayName") or team.get("displayName"),
                "position": athlete.get("position", {}).get("abbreviation"),
                "status": entry.get("status"),
                "comment": entry.get("shortComment") or long_comment,
                "long_comment": long_comment,
                "date": date,
            }

    entries = list(by_player.values())
    cache_set(cache_key, entries)
    return entries


def get_injury_lookup():
    """Returns {player_display_name: {"status", "comment", "long_comment",
    "position", "team", "date"}} for every player with a report, most recent
    per player across the whole league."""
    return {e["player"]: e for e in _fetch_raw_entries()}


def get_offensive_injuries():
    """Returns the raw injury entries for QB/RB/FB/WR/TE only -- the skill
    positions this app's props actually cover -- for display in the
    injuries table. Not filtered by status: includes Active-with-a-comment
    entries too, since those can carry real context (see player_props.py's
    return-risk discount)."""
    return [e for e in _fetch_raw_entries() if e.get("position") in OFFENSIVE_SKILL_POSITIONS]
