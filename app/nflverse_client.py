"""Free, no-key team-level efficiency stats from nflverse (the open-source NFL
analytics data project). We pull the same underlying CSVs the `nfl_data_py` /
`nflreadpy` libraries serve, over plain HTTP -- those libraries need a modern
Python + a Rust/C toolchain (polars/pyarrow/fastparquet) that isn't available
in this environment, but the raw CSVs work with just `requests` + `csv`.

We use this for per-team-week EPA (Expected Points Added), which is generally
considered a stronger predictor of future performance than raw scoring margin
because it credits a team for play-level efficiency rather than the final
score alone (which is noisy: garbage-time points, defensive/special-teams
TDs, etc. show up in the score but don't reflect true team strength).
"""
import csv
import io
import requests
from app.cache_utils import cache_get, cache_set
from app.config import RATINGS_CACHE_TTL_SECONDS

TEAM_RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
PLAYER_RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TIMEOUT = 20

TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _fetch_csv_rows(cache_key, url):
    cached = cache_get(cache_key, RATINGS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 404:
            cache_set(cache_key, [])
            return []
        resp.raise_for_status()
    except requests.RequestException:
        return []

    reader = csv.DictReader(io.StringIO(resp.text))
    rows = [row for row in reader if row.get("season_type") == "REG"]
    cache_set(cache_key, rows)
    return rows


def _fetch_all_games():
    """Every NFL game since 1999, one single CSV (not split per-season like
    the stats releases) -- home/away as team ABBREVIATIONS, scores blank if
    the game hasn't been played yet. This is the historical game-SCORE
    source for the whole app's rating model, replacing ESPN's scoreboard
    `dates=` range-query endpoint: that endpoint turned out to have a much
    more aggressive rate limit than ESPN's other endpoints (bare
    /scoreboard, /teams, /injuries all kept working fine) -- a single
    backtest run was enough to get it blocked for an extended period, even
    from a home network, which makes it unusable as this app's core data
    dependency. GitHub's release CDN doesn't have that problem."""
    cache_key = "nflverse_games_csv"
    cached = cache_get(cache_key, RATINGS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = requests.get(GAMES_RELEASE_URL, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    cache_set(cache_key, rows)
    return rows


def get_season_games(season, game_type="REG", completed_only=True):
    """[{"home", "away", "home_score", "away_score"}] for one season, full
    team names -- same shape espn_client's game-score functions return, so
    this is a drop-in swap for that data source."""
    games = []
    for row in _fetch_all_games():
        if row.get("game_type") != game_type:
            continue
        try:
            if int(row.get("season") or 0) != season:
                continue
        except (TypeError, ValueError):
            continue
        home_score, away_score = row.get("home_score"), row.get("away_score")
        if completed_only and (not home_score or not away_score):
            continue
        home_name = TEAM_ABBR_TO_NAME.get(row.get("home_team"))
        away_name = TEAM_ABBR_TO_NAME.get(row.get("away_team"))
        if not home_name or not away_name:
            continue
        try:
            games.append({
                "home": home_name, "away": away_name,
                "home_score": int(float(home_score)), "away_score": int(float(away_score)),
            })
        except (TypeError, ValueError):
            continue
    return games


def _fetch_team_week_rows(season):
    """Downloads and parses the nflverse team-week stats CSV for a season.
    Returns [] if the season has no data yet (e.g. requesting next season early)."""
    return _fetch_csv_rows(f"nflverse_stats_team_week_{season}", TEAM_RELEASE_URL.format(season=season))


def _fetch_player_week_rows(season):
    """Downloads and parses the nflverse player-week stats CSV for a season."""
    return _fetch_csv_rows(f"nflverse_stats_player_week_{season}", PLAYER_RELEASE_URL.format(season=season))


def build_player_index(current_season):
    """Returns {player_display_name: [(row, weight), ...]} across the current
    season (weight 1.0) and prior season (weight 0.5), most-recent-first."""
    current_rows = _fetch_player_week_rows(current_season)
    prev_rows = _fetch_player_week_rows(current_season - 1)
    weighted = [(r, 1.0) for r in current_rows] + [(r, 0.5) for r in prev_rows]

    index = {}
    for row, w in weighted:
        name = row.get("player_display_name")
        if not name:
            continue
        index.setdefault(name, []).append((row, w))
    return index


def compute_allowed_yardage(current_season):
    """Returns {full_team_name: {'pass': avg_pass_yds_allowed, 'rush': avg_rush_yds_allowed}}
    for use as a matchup-strength adjustment on player prop projections."""
    current_rows = _fetch_team_week_rows(current_season)
    prev_rows = _fetch_team_week_rows(current_season - 1)
    weighted_rows = [(r, 1.0) for r in current_rows] + [(r, 0.5) for r in prev_rows]
    if not weighted_rows:
        return {}

    def _num(row, field):
        try:
            return float(row.get(field) or 0)
        except (TypeError, ValueError):
            return None

    offense_pass = {}
    offense_rush = {}
    for row, _w in weighted_rows:
        pv, rv = _num(row, "passing_yards"), _num(row, "rushing_yards")
        if pv is not None:
            offense_pass[(row["game_id"], row["team"])] = pv
        if rv is not None:
            offense_rush[(row["game_id"], row["team"])] = rv

    pass_sum, pass_w, rush_sum, rush_w = {}, {}, {}, {}
    for row, w in weighted_rows:
        team, opp = row["team"], row["opponent_team"]
        pa = offense_pass.get((row["game_id"], opp))
        ra = offense_rush.get((row["game_id"], opp))
        if pa is not None:
            pass_sum[team] = pass_sum.get(team, 0.0) + pa * w
            pass_w[team] = pass_w.get(team, 0.0) + w
        if ra is not None:
            rush_sum[team] = rush_sum.get(team, 0.0) + ra * w
            rush_w[team] = rush_w.get(team, 0.0) + w

    result = {}
    for abbr, name in TEAM_ABBR_TO_NAME.items():
        if pass_w.get(abbr, 0) > 0 and rush_w.get(abbr, 0) > 0:
            result[name] = {"pass": pass_sum[abbr] / pass_w[abbr], "rush": rush_sum[abbr] / rush_w[abbr]}
    return result


def _game_offensive_epa(row):
    try:
        pass_epa = float(row.get("passing_epa") or 0)
        rush_epa = float(row.get("rushing_epa") or 0)
    except (TypeError, ValueError):
        return None
    return pass_epa + rush_epa


def compute_net_epa_ratings(current_season):
    """Returns {full_team_name: net_epa_rating} mean-centered across teams,
    blending the current season (weight 1.0) with the prior season
    (weight 0.5) the same way the scoring-margin power ratings do -- so
    there's a stable signal even in the first few weeks of a new season.
    """
    current_rows = _fetch_team_week_rows(current_season)
    prev_rows = _fetch_team_week_rows(current_season - 1)
    weighted_rows = [(r, 1.0) for r in current_rows] + [(r, 0.5) for r in prev_rows]

    if not weighted_rows:
        return {}

    offense_by_game = {}
    for row, _w in weighted_rows:
        epa = _game_offensive_epa(row)
        if epa is not None:
            offense_by_game[(row["game_id"], row["team"])] = epa

    off_sum, off_w = {}, {}
    def_sum, def_w = {}, {}
    for row, w in weighted_rows:
        team = row["team"]
        off_epa = offense_by_game.get((row["game_id"], team))
        if off_epa is None:
            continue
        off_sum[team] = off_sum.get(team, 0.0) + off_epa * w
        off_w[team] = off_w.get(team, 0.0) + w

        allowed_epa = offense_by_game.get((row["game_id"], row["opponent_team"]))
        if allowed_epa is None:
            continue
        def_sum[team] = def_sum.get(team, 0.0) + allowed_epa * w
        def_w[team] = def_w.get(team, 0.0) + w

    net_epa = {}
    for abbr, name in TEAM_ABBR_TO_NAME.items():
        if off_w.get(abbr, 0) > 0 and def_w.get(abbr, 0) > 0:
            net_epa[name] = (off_sum[abbr] / off_w[abbr]) - (def_sum[abbr] / def_w[abbr])

    if not net_epa:
        return {}

    mean_epa = sum(net_epa.values()) / len(net_epa)
    return {team: val - mean_epa for team, val in net_epa.items()}
