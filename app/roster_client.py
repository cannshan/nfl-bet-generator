"""Current NFL rosters from ESPN's public API -- the authoritative source for
"which team is this player on right now," independent of games played.

This matters because nflverse's weekly-stats CSVs only record a player's team
from games they've actually played: a player traded or signed in the
offseason who hasn't played a game yet still shows their OLD team in that
data (seen in practice: Tua Tagovailoa moved to Atlanta, but had zero 2026
rows, so nflverse-derived lookups kept attributing him to Miami). Anywhere
this app needs to know a player's current team, prefer this source over a
stat row's 'team' field.
"""
import requests
from app.config import ESPN_API_BASE
from app.cache_utils import cache_get, cache_set

TIMEOUT = 15
CACHE_TTL_SECONDS = 12 * 60 * 60


def get_current_rosters():
    """Returns {player_full_name: team_full_name} across the whole league,
    fetched fresh (32 team roster calls) at most once per TTL window."""
    cached = cache_get("espn_current_rosters", CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        resp = requests.get(f"{ESPN_API_BASE}/teams", params={"limit": 40}, timeout=TIMEOUT)
        resp.raise_for_status()
        teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    except (requests.RequestException, KeyError, IndexError):
        return {}

    rosters = {}
    for entry in teams:
        team = entry["team"]
        try:
            resp = requests.get(f"{ESPN_API_BASE}/teams/{team['id']}/roster", timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            continue
        for group in data.get("athletes", []):
            for athlete in group.get("items", []):
                name = athlete.get("fullName")
                if name:
                    rosters[name] = team["displayName"]

    cache_set("espn_current_rosters", rosters)
    return rosters


def get_qb_depth_charts():
    """Returns {team_full_name: [qb_names_in_depth_order]} from ESPN's real
    depth charts -- the actual current order, not a proxy inferred from
    historical usage. The full order (not just QB1) matters: if the starter
    is out AND the backup is also out, that's a much worse situation than
    the starter alone being out, and callers should be able to tell.

    This matters because "most pass attempts recently" is a bad proxy for
    "current starter": it gets a team's own backup-turned-emergency-QB wrong
    once the real starter changes for reasons other than a trade (beaten out,
    benched, coming back from injury at a different pace than a new signee).
    Seen in practice: a team's presumptive-starter-by-attempts was QB3 on the
    real depth chart, while the actual QB1's genuine injury went uncaught."""
    cached = cache_get("espn_qb_depth_charts", CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        resp = requests.get(f"{ESPN_API_BASE}/teams", params={"limit": 40}, timeout=TIMEOUT)
        resp.raise_for_status()
        teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    except (requests.RequestException, KeyError, IndexError):
        return {}

    depth_charts = {}
    for entry in teams:
        team = entry["team"]
        try:
            resp = requests.get(f"{ESPN_API_BASE}/teams/{team['id']}/depthcharts", timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            continue
        for chart in data.get("depthchart", []):
            for pos_key, pos_data in chart.get("positions", {}).items():
                if pos_key != "qb":
                    continue
                names = [a["displayName"] for a in pos_data.get("athletes", []) if a.get("displayName")]
                if names:
                    depth_charts[team["displayName"]] = names

    cache_set("espn_qb_depth_charts", depth_charts)
    return depth_charts
