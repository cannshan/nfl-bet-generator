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
"""
import datetime as dt
import re
import sqlite3
from app.config import PREDICTIONS_DB_PATH
from app import espn_client, nflverse_client

SETTLEMENT_DELAY_HOURS = 4  # wait this long past kickoff so final stats have posted
CALIBRATION_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
MIN_SAMPLE_FOR_CONFIDENCE = 20  # below this, don't imply the numbers mean much yet


def _connect():
    conn = sqlite3.connect(str(PREDICTIONS_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            market TEXT,
            selection TEXT,
            player TEXT,
            stat_category TEXT,
            side TEXT,
            line REAL,
            american_odds INTEGER,
            decimal_odds REAL,
            bookmaker TEXT,
            model_prob REAL,
            book_fair_prob REAL,
            edge REAL,
            section TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            actual_value REAL,
            settled_at TEXT,
            UNIQUE(home_team, away_team, commence_time, market, selection)
        )
        """
    )
    return conn


def log_suggestions(legs, season, week, section):
    """Records each leg the FIRST time it's suggested; later refreshes that
    re-suggest the same bet are silently ignored (UNIQUE constraint) so the
    log reflects what was said before the outcome was known, not a
    constantly-overwritten latest guess."""
    if not legs:
        return
    conn = _connect()
    now = dt.datetime.utcnow().isoformat()
    for leg in legs:
        if " @ " not in leg.get("matchup", ""):
            continue
        away, home = leg["matchup"].split(" @ ", 1)
        try:
            conn.execute(
                """INSERT OR IGNORE INTO predictions
                (created_at, season, week, home_team, away_team, commence_time, market, selection,
                 player, stat_category, side, line, american_odds, decimal_odds, bookmaker,
                 model_prob, book_fair_prob, edge, section)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    now, season, week, home, away, leg.get("commence_time"), leg["market"], leg["selection"],
                    leg.get("player"), leg.get("stat_category"), leg.get("side"), leg.get("line"),
                    leg["american_odds"], leg["decimal_odds"], leg["bookmaker"],
                    leg["model_prob"], leg["book_fair_prob"], leg["edge"], section,
                ),
            )
        except sqlite3.Error:
            continue
    conn.commit()
    conn.close()


def _parse_commence(iso_str):
    try:
        return dt.datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


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


def _settle_player_prop(row, player_index):
    from app.player_props import MARKET_CONFIG  # local import: avoids a module import cycle

    field_info = MARKET_CONFIG.get(row["stat_category"])
    if not field_info:
        return None, None
    field = field_info[0]
    rows = player_index.get(row["player"])
    if not rows:
        return None, None

    target_teams = {row["home_team"], row["away_team"]}
    match = None
    for r, w in rows:
        if w < 1.0:  # only this-season games, not last year's blended-in rows
            continue
        team_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("team"))
        opp_full = nflverse_client.TEAM_ABBR_TO_NAME.get(r.get("opponent_team"))
        if {team_full, opp_full} == target_teams:
            match = r
            break
    if not match:
        return None, None

    try:
        actual = float(match.get(field) or 0)
    except (TypeError, ValueError):
        return None, None

    if actual == row["line"]:
        return "void", actual
    is_over = actual > row["line"]
    hit = is_over if row["side"] == "Over" else not is_over
    return ("hit" if hit else "miss"), actual


def settle_pending():
    """Attempts to settle every pending prediction whose game started more
    than SETTLEMENT_DELAY_HOURS ago. Safe to call on every page load --
    anything without final data available yet is simply left pending and
    retried next time. Returns how many were newly settled."""
    conn = _connect()
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=SETTLEMENT_DELAY_HOURS)
    pending = conn.execute("SELECT * FROM predictions WHERE status = 'pending'").fetchall()

    games_by_season = {}
    players_by_season = {}
    settled_count = 0

    for row in pending:
        commence = _parse_commence(row["commence_time"])
        if not commence or commence > cutoff:
            continue

        season = row["season"]
        if not season:
            continue
        if season not in games_by_season:
            try:
                games = espn_client.get_season_games_to_date(season)
            except Exception:
                games = []
            games_by_season[season] = {(g["home"], g["away"]): (g["home_score"], g["away_score"]) for g in games}
        if season not in players_by_season:
            try:
                players_by_season[season] = nflverse_client.build_player_index(season)
            except Exception:
                players_by_season[season] = {}

        if row["market"].startswith("Player Prop"):
            status, actual = _settle_player_prop(row, players_by_season[season])
        else:
            status, actual = _settle_game_market(row, games_by_season[season]), None

        if status is None:
            continue  # data not posted yet -- leave pending, retry on a later refresh

        conn.execute(
            "UPDATE predictions SET status=?, actual_value=?, settled_at=? WHERE id=?",
            (status, actual, dt.datetime.utcnow().isoformat(), row["id"]),
        )
        settled_count += 1

    conn.commit()
    conn.close()
    return settled_count


def get_track_record(stake=5.0):
    conn = _connect()
    settled = conn.execute("SELECT * FROM predictions WHERE status IN ('hit', 'miss')").fetchall()
    pending_count = conn.execute("SELECT COUNT(*) c FROM predictions WHERE status='pending'").fetchone()["c"]
    void_count = conn.execute("SELECT COUNT(*) c FROM predictions WHERE status='void'").fetchone()["c"]
    conn.close()

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

    return {
        "total_settled": total,
        "pending": pending_count,
        "void": void_count,
        "overall_hit_rate": (hits / total) if total else None,
        "buckets": buckets,
        "roi_pct": roi_pct,
        "stake": stake,
        "confident": total >= MIN_SAMPLE_FOR_CONFIDENCE,
        "recent": sorted(settled, key=lambda r: r["settled_at"] or "", reverse=True)[:25],
    }
