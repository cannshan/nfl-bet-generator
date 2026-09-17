"""Measures how correlated same-game bets really are, from real outcomes.

Why this exists: a parlay's combined probability is only the product of its
legs' probabilities when the legs are independent. Same-game legs aren't --
a QB's big passing day tends to come with his receivers' big days and a
higher game total; a run-heavy blowout depresses passing volume; etc.
Rather than guess these effects, this script measures them from every
regular-season player-game 2018-2025 (nflverse) plus each game's real
closing spread/total line (nflverse schedules), walk-forward: each player's
"expected" stat is a trailing weighted mean built only from games BEFORE
that one, mirroring the app's own projection weights, so there's no
lookahead.

For every pair of players/markets in the same game it records the Pearson
correlation of their standardized residuals (actual minus trailing
expectation, divided by trailing std). Those correlations feed a Gaussian
copula in app/correlations.py, which is how parlay_builder turns per-leg
probabilities into an honest joint probability for same-game combos.

Run:  .venv/Scripts/python.exe scripts/backtest_correlations.py
Output: prints the table and writes app/correlation_table.json.
"""
import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import requests

SEASONS = range(2018, 2026)
PLAYER_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
STATS = ["passing_yards", "attempts", "completions", "passing_tds", "rushing_yards", "receiving_yards", "receptions"]
# A player only contributes a residual for a stat they're actually used in --
# otherwise a WR's "passing yards" residual (0 vs 0) would flood the pairs.
USAGE_FLOOR = {
    "passing_yards": 120, "attempts": 18, "completions": 10, "passing_tds": 0.6,
    "rushing_yards": 20, "receiving_yards": 15, "receptions": 1.5,
}
MIN_PRIOR_WEIGHT = 3.0
MARKET_SIGMA = 13.0  # points; std of margin/total vs closing line
MIN_PAIRS = 300
Z_CLIP = 3.0


def fetch_csv(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def num(row, field):
    try:
        return float(row.get(field) or 0)
    except (TypeError, ValueError):
        return 0.0


def trailing_stats(history, season, week, stat):
    """Weighted mean/std of `stat` over prior games, weights mirroring
    player_props.py: same season 0.9^(weeks back), prior season 0.5, older 0.25."""
    vals, weights = [], []
    for (s, w, row) in history:
        if s == season:
            wt = 0.9 ** max(0, week - w)
        elif s == season - 1:
            wt = 0.5
        else:
            wt = 0.25
        vals.append(num(row, stat))
        weights.append(wt)
    tw = sum(weights)
    if tw < MIN_PRIOR_WEIGHT:
        return None, None
    mean = sum(v * w for v, w in zip(vals, weights)) / tw
    var = sum(w * (v - mean) ** 2 for v, w in zip(vals, weights)) / tw
    return mean, math.sqrt(var)


def main():
    print("downloading schedules...", flush=True)
    games = {}
    for g in fetch_csv(GAMES_URL):
        try:
            season = int(g["season"])
        except ValueError:
            continue
        if g.get("game_type") != "REG" or season not in SEASONS:
            continue
        if not g.get("home_score") or not g.get("away_score"):
            continue
        try:
            spread = float(g["spread_line"]) if g.get("spread_line") else None
            total_line = float(g["total_line"]) if g.get("total_line") else None
        except ValueError:
            spread = total_line = None
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        games[(season, int(g["week"]), g["home_team"], g["away_team"])] = {
            "margin_z": ((hs - as_) - spread) / MARKET_SIGMA if spread is not None else None,
            "total_z": ((hs + as_) - total_line) / MARKET_SIGMA if total_line is not None else None,
        }
    print(f"  {len(games)} games with results", flush=True)

    rows_by_season = {}
    for season in SEASONS:
        print(f"downloading player-week {season}...", flush=True)
        rows_by_season[season] = [r for r in fetch_csv(PLAYER_URL.format(season=season)) if r.get("season_type") == "REG"]

    # Walk forward in time, building residuals from prior-only history.
    history = defaultdict(list)  # player -> [(season, week, row)]
    residuals_by_game = defaultdict(list)  # game key -> [(team, stat, z)]
    for season in SEASONS:
        by_week = defaultdict(list)
        for r in rows_by_season[season]:
            try:
                by_week[int(r["week"])].append(r)
            except ValueError:
                pass
        for week in sorted(by_week):
            for r in by_week[week]:
                name = r.get("player_display_name")
                if not name:
                    continue
                team, opp = r.get("team"), r.get("opponent_team")
                hist = history[name]
                for stat in STATS:
                    mean, std = trailing_stats(hist, season, week, stat)
                    if mean is None or mean < USAGE_FLOOR[stat]:
                        continue
                    std = max(std, 0.2 * mean, 1.0)
                    z = max(-Z_CLIP, min(Z_CLIP, (num(r, stat) - mean) / std))
                    for key in ((season, week, team, opp), (season, week, opp, team)):
                        if key in games:
                            residuals_by_game[key].append((team, stat, z))
                            break
                hist.append((season, week, r))

    # Accumulate Pearson sums per (stat_a, stat_b, relation).
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def add(key, x, y):
        a = acc[key]
        a[0] += 1
        a[1] += x
        a[2] += y
        a[3] += x * x
        a[4] += y * y
        a[5] += x * y

    for gkey, res in residuals_by_game.items():
        _season, _week, home, _away = gkey
        g = games[gkey]
        for i in range(len(res)):
            ti, si, zi = res[i]
            if g["total_z"] is not None:
                add((si, "total", "same_game"), zi, g["total_z"])
            if g["margin_z"] is not None:
                own_margin = g["margin_z"] if ti == home else -g["margin_z"]
                add((si, "margin", "own_team"), zi, own_margin)
            for j in range(i + 1, len(res)):
                tj, sj, zj = res[j]
                rel = "same_team" if ti == tj else "opponent"
                add((si, sj, rel), zi, zj)
                add((sj, si, rel), zj, zi)
        if g["total_z"] is not None and g["margin_z"] is not None:
            add(("margin", "total", "own_team"), g["margin_z"], g["total_z"])

    table = {}
    print(f"\n{'stat A':16s} {'stat B':16s} {'relation':10s} {'rho':>7s} {'n':>7s}")
    for key in sorted(acc):
        n, sx, sy, sxx, syy, sxy = acc[key]
        if n < MIN_PAIRS:
            continue
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        if vx <= 0 or vy <= 0:
            continue
        rho = (sxy / n - (sx / n) * (sy / n)) / math.sqrt(vx * vy)
        table["|".join(key)] = {"rho": round(rho, 4), "n": n}
        print(f"{key[0]:16s} {key[1]:16s} {key[2]:10s} {rho:+7.3f} {n:7d}")

    out = Path(__file__).resolve().parent.parent / "app" / "correlation_table.json"
    out.write_text(json.dumps(table, indent=1, sort_keys=True))
    print(f"\nwrote {out} ({len(table)} entries)")


if __name__ == "__main__":
    sys.exit(main())
