"""Does this app's team model beat the closing moneyline? Walk-forward test.

Rebuilds the exact rating the app uses (scoring-margin power ratings
blended 55% with nflverse EPA, previous season blended in at half weight
until 48 current-season games exist, calibrated via WIN_PROB_CALIBRATION,
deferring to the market above WIN_PROB_CEILING) from only the games played
BEFORE each week, 2018-2025, then compares its home-win probability against
the real closing moneyline (nflverse schedules) for that game.

The question isn't "is the model calibrated?" -- it's "when the model
disagrees with the market, who's right?" So the output fits the weight k
in  p = market + k * (model - market)  that minimizes log-loss on real
outcomes, and shows what happened in the games where the model claimed a
big edge. That k is what value_finder.MODEL_WEIGHT_VS_MARKET_MONEYLINE
should be.

Run:  .venv/Scripts/python.exe scripts/backtest_moneyline.py
"""
import csv
import io
import math
import sys
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import ratings, odds_math  # noqa: E402

SEASONS = range(2018, 2026)
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TEAM_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
EPA_BLEND_WEIGHT = 0.55
MIN_CURRENT_GAMES = 48


def fetch_csv(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def net_epa(rows_weighted):
    offense = {}
    for row, _w in rows_weighted:
        try:
            offense[(row["game_id"], row["team"])] = float(row.get("passing_epa") or 0) + float(row.get("rushing_epa") or 0)
        except ValueError:
            pass
    off_sum, off_w, def_sum, def_w = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for row, w in rows_weighted:
        team = row["team"]
        off = offense.get((row["game_id"], team))
        allowed = offense.get((row["game_id"], row["opponent_team"]))
        if off is None or allowed is None:
            continue
        off_sum[team] += off * w
        off_w[team] += w
        def_sum[team] += allowed * w
        def_w[team] += w
    net = {t: off_sum[t] / off_w[t] - def_sum[t] / def_w[t] for t in off_w if def_w[t] > 0}
    if not net:
        return {}
    mean = sum(net.values()) / len(net)
    return {t: v - mean for t, v in net.items()}


def main():
    print("downloading schedules...", flush=True)
    games = []
    for g in fetch_csv(GAMES_URL):
        try:
            season, week = int(g["season"]), int(g["week"])
        except ValueError:
            continue
        if g.get("game_type") != "REG" or season < min(SEASONS) - 1 or season > max(SEASONS):
            continue
        if not g.get("home_score") or not g.get("away_score"):
            continue
        games.append({
            "season": season, "week": week, "home": g["home_team"], "away": g["away_team"],
            "hs": int(float(g["home_score"])), "as": int(float(g["away_score"])),
            "home_ml": float(g["home_moneyline"]) if g.get("home_moneyline") else None,
            "away_ml": float(g["away_moneyline"]) if g.get("away_moneyline") else None,
        })
    team_rows = {}
    for season in range(min(SEASONS) - 1, max(SEASONS) + 1):
        print(f"downloading team-week {season}...", flush=True)
        team_rows[season] = [r for r in fetch_csv(TEAM_URL.format(season=season)) if r.get("season_type") == "REG"]

    obs = []  # (p_model, p_market, home_won)
    for season in SEASONS:
        season_games = [g for g in games if g["season"] == season]
        prev_games = [g for g in games if g["season"] == season - 1]
        for week in range(1, 19):
            this_week = [g for g in season_games if g["week"] == week and g["home_ml"] and g["away_ml"]]
            if not this_week:
                continue
            current = [g for g in season_games if g["week"] < week]
            weighted = [(g["home"], g["away"], g["hs"], g["as"], 1.0) for g in current]
            epa_rows = [(r, 1.0) for r in team_rows[season] if int(r["week"]) < week]
            if len(current) < MIN_CURRENT_GAMES:
                weighted += [(g["home"], g["away"], g["hs"], g["as"], 0.5) for g in prev_games]
                epa_rows += [(r, 0.5) for r in team_rows[season - 1]]
            if not weighted:
                continue
            power = ratings.compute_power_ratings(weighted)
            epa = net_epa(epa_rows)
            blended = {t: (1 - EPA_BLEND_WEIGHT) * r + EPA_BLEND_WEIGHT * epa[t] if t in epa else r for t, r in power.items()}
            mean_r = sum(blended.values()) / len(blended)
            blended = {t: v - mean_r for t, v in blended.items()}
            for g in this_week:
                if g["home"] not in blended or g["away"] not in blended:
                    continue
                p_model = ratings.win_probability(blended[g["home"]], blended[g["away"]])
                fair_home, fair_away = odds_math.devig_two_way(
                    odds_math.american_to_implied_prob(g["home_ml"]), odds_math.american_to_implied_prob(g["away_ml"]),
                )
                if fair_home > ratings.WIN_PROB_CEILING or fair_away > ratings.WIN_PROB_CEILING:
                    p_model = fair_home  # the app defers to the market here
                if g["hs"] == g["as"]:
                    continue
                obs.append((p_model, fair_home, 1 if g["hs"] > g["as"] else 0))

    print(f"\n{len(obs)} games with a closing moneyline, 2018-2025\n")

    def logloss(k):
        total = 0.0
        for pm, pk, y in obs:
            p = min(max(pk + k * (pm - pk), 0.01), 0.99)
            total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return total / len(obs)

    print("weight on model (k)   log-loss   (lower is better; k=0 is pure market)")
    best_k, best_ll = None, None
    for k10 in range(0, 21):
        k = k10 / 20
        ll = logloss(k)
        if best_ll is None or ll < best_ll:
            best_k, best_ll = k, ll
        print(f"  {k:4.2f}               {ll:.5f}")
    print(f"\nbest k = {best_k:.2f}")

    print("\nWhen the model disagreed with the market (home side), who was right?")
    print(f"{'model - market':>16s} {'n':>6s} {'model avg':>10s} {'market avg':>11s} {'actual':>8s}")
    buckets = [(-1, -0.15), (-0.15, -0.08), (-0.08, -0.03), (-0.03, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 1)]
    for lo, hi in buckets:
        rows = [o for o in obs if lo <= o[0] - o[1] < hi]
        if not rows:
            continue
        print(f"{lo:+.2f} to {hi:+.2f}   {len(rows):6d} {sum(o[0] for o in rows)/len(rows):10.3f} "
              f"{sum(o[1] for o in rows)/len(rows):11.3f} {sum(o[2] for o in rows)/len(rows):8.3f}")

    # Flat-stake ROI of betting whichever side the model liked by >= 5 points, at the closing price.
    print("\nROI betting every side the model liked by >= 5 pts over the market (flat $1, closing price, devigged -> approx):")
    staked, returned = 0, 0.0
    for pm, pk, y in obs:
        for side_model, side_mkt, won in ((pm, pk, y), (1 - pm, 1 - pk, 1 - y)):
            if side_model - side_mkt >= 0.05:
                staked += 1
                returned += won / side_mkt * 0.955  # ~4.5% vig back onto the fair price
    if staked:
        print(f"  {staked} bets, ROI {(returned - staked) / staked * 100:+.1f}%")


if __name__ == "__main__":
    main()
