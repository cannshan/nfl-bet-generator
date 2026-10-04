"""How much should player-prop probabilities lean on this app's history model?
Unselected backtest against the full prop boards the app has cached.

The question: every prop leg is priced off the market's consensus center
and tilted toward the player's game-log projection by
MODEL_WEIGHT_VS_MARKET (0.10), shrunk as the two disagree and capped at
MAX_MODEL_TILT_SIGMAS (0.25) of sigma (see app/player_props.py). Those
constants were set from 19 settled bets. A later look at 544 settled
tracked legs put the optimal weight near 0.25 -- but those legs were
SELECTED (only what the app itself suggested, i.e. legs where the model
disagreed with the market in the direction it liked), so their outcomes
can't say what the weight should be across the whole board.

The data: each app_cache event_odds_* row is a FULL board -- every player,
every book -- not just what was suggested. For completed games, those
boards plus nflverse's player-week box scores are an unselected sample:
every player-market the offer book posted, priced exactly the way
production would have priced it at the time.

Method (all pricing math is imported from app/player_props.py -- the same
_collect_market, _history_projection, _market_consensus, _blend_center and
_prob_under production runs -- so nothing here can drift from the app):

  - Boards: only those cached strictly BEFORE kickoff (a key holds only
    its LAST fetch, and one fetched after kickoff is live in-play odds),
    with at least one bookmaker, for a game nflverse has a result for.
    Week and result come from nflverse's team-week game_ids
    ({season}_{week}_{away}_{home}).
  - Outcome: the player's nflverse row for that week on one of the two
    teams. No row = did not play (books void those) -> excluded.
  - As-of history, no look-ahead: current-season rows from weeks before
    the game (weight 1.0) + prior season (0.5), then production's own
    build_player_index / _apply_recency / role trend. The opponent matchup
    factor uses yards allowed from team-week rows before the game too.
    The injury report, QB depth chart, defensive-injury bonus and weather
    forecast as they stood before a past game can't be reconstructed, so
    those factors are neutral (1.0) here -- the projection tested is
    "game log + matchup + role trend", a bit less than production's.
  - Market center: production's consensus over EVERY book's devigged
    two-sided quote, with production's sigma (history std, floors,
    LOG_STD_CALIBRATION, scaled up when the market sits above history).
  - Scored lines: every distinct point the offer book (config.BOOKMAKER_KEY)
    posts for that player-market, P(Over) vs. the result, pushes skipped;
    each player-market counts once in total, so a player with several
    points doesn't outweigh one with a single line.
  - Compared by log loss and Brier: market-only, history-only, production,
    and a grid over weight 0..1 x cap {0.25, 0.5, 1, 2, none} sigmas x
    form {disagreement-shrunk (production), fixed weight}.
  - Continuous check: regress the realized residual (actual - market
    center, in model space: log(x+1) for receiving) on the gap (history
    center - market center), both in sigma units -- the slope is the
    squared-error-optimal center weight. OLS and least-absolute-deviation
    (the median, which is what a line's 50% point is about).
  - Sigma: a multiplier k on production's sigma (0.70-1.50), scored at the
    offered lines and on a ladder of off-center lines (market center
    +-0.5/1/1.5 sigma, snapped to half-points) -- an off-center line's
    price depends on sigma, the at-the-money line's barely at all. The
    same ladder also scores the blend weight (more lines per outcome =
    more power), and that ladder-fitted weight is then cross-validated
    at the real offered lines, the metric bets are actually made on.
  - Under-lean control: the weight is refit with a per-group constant
    center shift, so a generic lean to the Under (which the history
    projection, running below the market on average, would pick up for
    free) isn't mistaken for player-specific information.
  - Uncertainty: game-clustered bootstrap (resample whole games) for every
    parameter, 90% intervals. Overfitting guard: leave-one-game-out and
    leave-one-week-out cross-validation, refitting the weight inside each
    fold. The current constants were fitted on none of this data, so their
    plain score is already out of sample -- a re-tuned (form, cap) has to
    beat it in cross-validation to be recommended, and the weight is
    conservative: the current value unless the 90% interval excludes it,
    then only the interval edge nearest the current value.

Known limits of the reconstruction: a pass-catcher who played but recorded
no stat at all has no nflverse row and is treated as did-not-play; names
are matched exactly the way production matches them (Odds API description
== nflverse display name), so the "Jr."/nickname mismatches production
silently skips are skipped and counted here too; and the as-of cutoff is by
week, so a Thursday board fetched before the previous Monday night game
already sees that game's rows (other teams only -- it can only nudge the
league-average yards allowed).

Run:  .venv/Scripts/python.exe scripts/backtest_prop_blend.py [--boards-file PATH] [--out PATH]
      [--bootstrap N] [--seed N]
Reads app_cache (event_odds_% rows + cached nflverse releases) read-only,
in passive mode: no Odds API / ESPN / nflverse network fetch, no writes.
Every completed game nflverse ingests adds its cached pre-kickoff board to
the sample, so re-running after each week tightens every interval.
"""
import argparse
import datetime as dt
import json
import math
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import cache_utils  # noqa: E402

# Before anything else is imported or called: every read below must come
# from the cache (at any age), never a live fetch.
cache_utils.set_mode("passive")


def _refuse_write(*_args, **_kwargs):
    raise RuntimeError("backtest_prop_blend.py is read-only: an app_cache write was attempted")


# ...and nothing may write to app_cache. Passive mode already means no
# fetch-then-cache, but this makes a write impossible rather than merely
# unexpected: the app modules imported next bind these names at import.
cache_utils.cache_set = cache_utils.set_raw = _refuse_write

from app import nflverse_client, odds_math, player_props as pp  # noqa: E402

GROUP_OF_MARKET = {
    "player_pass_yds": "pass", "player_pass_attempts": "pass",
    "player_pass_completions": "pass", "player_pass_tds": "pass",
    "player_rush_yds": "rush",
    "player_reception_yds": "receiving", "player_receptions": "receiving",
}
GROUPS = ("pass", "rush", "receiving")
CURRENT_WEIGHT = pp.MODEL_WEIGHT_VS_MARKET
CURRENT_CAP = pp.MAX_MODEL_TILT_SIGMAS
CURRENT_SHRINK = True
CURRENT_K = 1.0

WEIGHT_GRID = [round(i * 0.02, 2) for i in range(51)]  # 0.00 .. 1.00
CAP_GRID = [0.25, 0.5, 1.0, 2.0, math.inf]
SHRINK_FORMS = (True, False)  # True = production's disagreement-shrunk weight, False = fixed weight
K_GRID = [round(0.70 + 0.05 * i, 2) for i in range(17)]  # 0.70 .. 1.50
LADDER_SIGMAS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
# A constant shift of the center, in sigma units, fitted per group. The
# history projection runs below the market on average and prop results
# also tend to land below the market center (the over-bias), so part of
# any "history helps" signal can be nothing but a generic lean to the
# Under. Refitting the weight WITH a per-group shift separates the two:
# whatever weight survives is player-specific information.
SHIFT_GRID = [round(-0.30 + 0.05 * i, 2) for i in range(13)]  # -0.30 .. +0.30 sigma
SHIFT_WEIGHTS = WEIGHT_GRID[::2]  # 0.00 .. 1.00 step 0.04 (keeps the 2-D bootstrap fast)
# History-only probabilities can be extreme (0.999 on a stale game log);
# clipping keeps one such miss from dominating the average log loss.
P_CLIP = 1e-3
CI_LO, CI_HI = 0.05, 0.95
# Switching away from production's (shrink form, cap) family needs its
# weight-refit procedure to beat the current constants in leave-one-game-out
# CV AND beat the current family in this share of bootstrap resamples -- one
# free parameter each, but only a 20-30 game sample to fit it on.
SWITCH_FAMILY_BOOT_SHARE = 0.80
NAME_TO_ABBR = {name: abbr for abbr, name in nflverse_client.TEAM_ABBR_TO_NAME.items()}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _game_week(team_row):
    parts = (team_row.get("game_id") or "").split("_")
    return _int(parts[1]) if len(parts) == 4 else 0


def _parse_iso(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _season_of(commence_ts):
    d = dt.datetime.fromtimestamp(commence_ts, dt.timezone.utc)
    return d.year if d.month >= 3 else d.year - 1


def _fmt_cap(cap):
    return "none" if cap == math.inf else f"{cap:g}"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_boards(boards_file):
    """[{key, cached_at, data}] -- from a JSON dump of the same shape, or
    straight from app_cache (read-only select, paged so no single response
    carries the whole multi-MB set)."""
    if boards_file:
        with open(boards_file, encoding="utf-8") as f:
            return json.load(f), f"file {boards_file}"
    rows, start, page = [], 0, 20
    while True:
        chunk = (
            cache_utils._sb().table(cache_utils.TABLE).select("key,data,cached_at")
            .like("key", "event_odds_%").order("key").range(start, start + page - 1).execute()
        ).data
        rows += chunk
        if len(chunk) < page:
            break
        start += page
    return rows, "app_cache (Supabase, read-only)"


def completed_games(team_rows):
    """{(away_abbr, home_abbr): week}. nflverse game_ids are
    {season}_{week}_{away}_{home}, and a team-week row only exists once a
    game has been played and ingested -- so this is both the week lookup
    and the "has a result" check."""
    games = {}
    for r in team_rows:
        parts = (r.get("game_id") or "").split("_")
        if len(parts) == 4:
            games[(parts[2], parts[3])] = _int(parts[1])
    return games


@contextmanager
def nflverse_as_of(season, week, player_rows, team_rows):
    """Production's own build_player_index / compute_allowed_yardage, fed
    only rows from BEFORE `week`. nflverse_client has no through_week
    parameter (and this script mustn't change that module), so its two
    row fetchers are swapped for filtered ones for the duration -- the
    aggregation code that runs is exactly production's."""
    orig = (nflverse_client._fetch_player_week_rows, nflverse_client._fetch_team_week_rows)

    def players(s):
        rows = player_rows.get(s, [])
        return [r for r in rows if _int(r.get("week")) < week] if s == season else rows

    def teams(s):
        rows = team_rows.get(s, [])
        return [r for r in rows if _game_week(r) < week] if s == season else rows

    nflverse_client._fetch_player_week_rows, nflverse_client._fetch_team_week_rows = players, teams
    try:
        yield
    finally:
        nflverse_client._fetch_player_week_rows, nflverse_client._fetch_team_week_rows = orig


def select_boards(raw_boards, games_by_season, counts):
    """The usable pre-kickoff board per completed game (latest one, if an
    event somehow has several keys). Every exclusion is counted."""
    chosen = {}
    for row in raw_boards:
        counts["boards_total"] += 1
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        commence = _parse_iso(data.get("commence_time"))
        cached_at = _num(row.get("cached_at"))
        if commence is None or cached_at is None:
            counts["boards_excl_unparseable"] += 1
            continue
        if cached_at >= commence:
            counts["boards_excl_cached_at_or_after_kickoff"] += 1
            if data.get("bookmakers"):
                counts["boards_excl_cached_at_or_after_kickoff__with_inplay_odds"] += 1
            continue
        if not data.get("bookmakers"):
            counts["boards_excl_no_bookmakers"] += 1
            continue
        season = _season_of(commence)
        away, home = NAME_TO_ABBR.get(data.get("away_team")), NAME_TO_ABBR.get(data.get("home_team"))
        week = games_by_season.get(season, {}).get((away, home))
        if week is None:
            counts["boards_excl_no_nflverse_result_yet"] += 1
            continue
        if not any(odds_math.is_offer_book(bm) for bm in data["bookmakers"]):
            counts["boards_excl_no_offer_book"] += 1
            continue
        eid = data.get("id") or row.get("key")
        if eid in chosen:
            counts["boards_excl_older_duplicate_of_same_event"] += 1
            if chosen[eid]["cached_at"] >= cached_at:
                continue
        chosen[eid] = {
            "event_id": eid, "board": data, "cached_at": cached_at, "commence": commence,
            "season": season, "week": week, "home": data["home_team"], "away": data["away_team"],
            "home_abbr": home, "away_abbr": away,
        }
    games = sorted(chosen.values(), key=lambda g: (g["commence"], g["home"]))
    counts["boards_used"] = len(games)
    return games


def build_records(games, player_rows, team_rows, counts):
    """One record per scored player-market, priced the production way with
    as-of data. Exclusions are counted in the order they're checked."""
    all_names = {r.get("player_display_name") for rows in player_rows.values() for r in rows}
    as_of_cache = {}
    records = []
    for gi, game in enumerate(games):
        season, week = game["season"], game["week"]
        if (season, week) not in as_of_cache:
            with nflverse_as_of(season, week, player_rows, team_rows):
                index = nflverse_client.build_player_index(season)
                allowed = nflverse_client.compute_allowed_yardage(season)
            outcome_rows = defaultdict(list)
            for r in player_rows.get(season, []):
                if _int(r.get("week")) == week:
                    outcome_rows[r.get("player_display_name")].append(r)
            as_of_cache[(season, week)] = (index, allowed, pp._league_average_allowed(allowed), outcome_rows)
        index, allowed, league_avg, outcome_rows = as_of_cache[(season, week)]

        best_prices, quotes = pp._collect_market(game["board"])
        offers = defaultdict(set)
        for (mkey, player, point, _side) in best_prices:
            offers[(mkey, player)].add(point)

        for (mkey, player), points in sorted(offers.items()):
            counts["pm_offered_by_offer_book"] += 1
            field, _label, category = pp.MARKET_CONFIG[mkey]
            player_quotes = quotes.get((mkey, player))
            if not player_quotes:
                counts["pm_excl_no_two_sided_quote_any_book"] += 1
                continue
            if player not in all_names:
                counts["pm_excl_name_not_in_nflverse"] += 1
                counts.setdefault("_unmatched_names", Counter())[player] += 1
                continue
            outcome = [r for r in outcome_rows.get(player, []) if r.get("team") in (game["home_abbr"], game["away_abbr"])]
            if not outcome:
                counts["pm_excl_no_row_this_game_did_not_play"] += 1
                continue
            if len(outcome) > 1:
                counts["pm_excl_ambiguous_name_two_rows"] += 1
                continue
            outcome = outcome[0]
            raw_rows = index.get(player)
            if not raw_rows:
                counts["pm_excl_no_history_before_game"] += 1
                continue
            team_full = nflverse_client.TEAM_ABBR_TO_NAME[outcome["team"]]
            opponent_full = game["away"] if team_full == game["home"] else game["home"]
            rows = pp._apply_recency(raw_rows)
            projection = pp._history_projection(
                rows, field, category,
                matchup_factor=pp._matchup_factor(field, category, opponent_full, allowed, league_avg),
            )
            if projection is None:
                counts["pm_excl_small_sample_production_skips"] += 1
                continue
            model_center, sigma0, log_space, mean = projection
            consensus = pp._market_consensus(player_quotes, sigma0, log_space, mean)
            if consensus is None:
                counts["pm_excl_no_market_center"] += 1
                continue
            market_center, sigma = consensus

            y = _num(outcome.get(field)) or 0.0
            scored = []
            for point in sorted(points):
                counts["line_evals_offered"] += 1
                if y == point:
                    counts["line_evals_excl_push"] += 1
                    continue
                scored.append((point, y > point))
            if not scored:
                counts["pm_excl_every_line_pushed"] += 1
                continue

            yt = math.log(max(y, 0) + 1) if log_space else y
            records.append({
                "game": gi, "week": week, "matchup": f"{game['away']} @ {game['home']}",
                "market": mkey, "group": GROUP_OF_MARKET[mkey], "player": player, "field": field,
                "log_space": log_space, "mean": mean, "sigma0": sigma0, "sigma": sigma,
                "model_center": model_center, "market_center": market_center,
                "actual": y, "actual_model_space": yt, "points": scored,
                "quotes": player_quotes, "n_quotes": len(player_quotes),
            })
            counts["pm_scored"] += 1
            counts["line_evals_scored"] += len(scored)
    return records


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def p_over(point, center, sigma, log_space):
    return 1 - pp._prob_under(point, center, sigma, log_space)


def score_points(points, center, sigma, log_space):
    """(log loss, Brier) averaged over one player-market's scored lines --
    so each player-market carries weight 1 however many lines it has."""
    ll = br = 0.0
    for point, over in points:
        p = min(max(p_over(point, center, sigma, log_space), P_CLIP), 1 - P_CLIP)
        ll -= math.log(p if over else 1 - p)
        br += ((1.0 if over else 0.0) - p) ** 2
    return ll / len(points), br / len(points)


def ladder_points(rec):
    """Off-center lines at market center +-LADDER_SIGMAS sigma, snapped to
    the half-point a book would actually post (every stat here is a whole
    number, so a half-point line can't push -- and pricing at x.5 is the
    same continuity treatment production gets at real lines). A real line
    sits near the center, where P is ~50% whatever sigma or a small center
    shift is; these off-center lines are what actually discriminate."""
    pts = set()
    for m in LADDER_SIGMAS:
        c = rec["market_center"] + m * rec["sigma"]
        raw = math.exp(c) - 1 if rec["log_space"] else c
        pt = math.floor(raw) + 0.5
        if pt >= 0.5:
            pts.add(pt)
    return [(pt, rec["actual"] > pt) for pt in sorted(pts)]


def build_configs():
    configs = [("market", None, None, None), ("history", None, None, None)]
    for shrink in SHRINK_FORMS:
        for cap in CAP_GRID:
            for w in WEIGHT_GRID:
                configs.append(("grid", w, cap, shrink))
    return configs


def config_center(cfg, rec):
    kind, w, cap, shrink = cfg
    if kind == "market":
        return rec["market_center"]
    if kind == "history":
        return rec["model_center"]
    return pp._blend_center(rec["model_center"], rec["market_center"], rec["sigma"],
                            weight=w, cap_sigmas=cap, disagreement_shrink=shrink)


def per_game_losses(records, configs, n_games):
    """Per config c and game g, summed over that game's player-markets:
    off_L/off_B = log loss/Brier at the offer book's lines, lad_L = log
    loss on the off-center ladder, grp_L = off_L split by group. N/N_lad/Ng
    count player-markets. Everything downstream (bootstrap, CV, splits)
    only needs these per-game sums."""
    zeros = lambda: [[0.0] * n_games for _ in configs]  # noqa: E731
    s = {"off_L": zeros(), "off_B": zeros(), "lad_L": zeros(),
         "grp_L": {grp: zeros() for grp in GROUPS},
         "N": [0] * n_games, "N_lad": [0] * n_games, "Ng": {grp: [0] * n_games for grp in GROUPS}}
    for rec in records:
        g, grp, ladder = rec["game"], rec["group"], rec["_ladder"]
        s["N"][g] += 1
        s["Ng"][grp][g] += 1
        if ladder:
            s["N_lad"][g] += 1
        for ci, cfg in enumerate(configs):
            center = config_center(cfg, rec)
            ll, br = score_points(rec["points"], center, rec["sigma"], rec["log_space"])
            s["off_L"][ci][g] += ll
            s["off_B"][ci][g] += br
            s["grp_L"][grp][ci][g] += ll
            if ladder:
                s["lad_L"][ci][g] += score_points(ladder, center, rec["sigma"], rec["log_space"])[0]
    return s


def sigma_losses(records, n_games):
    """Per k in K_GRID: per-game (offered-line, ladder) log loss for the
    market-only probability with production's sigma times k -- k applied
    BEFORE the market center is solved, exactly as changing the sigma
    constant in production would (off-center quotes translate through it)."""
    off = [[0.0] * n_games for _ in K_GRID]
    lad = [[0.0] * n_games for _ in K_GRID]
    off_g = {grp: [[0.0] * n_games for _ in K_GRID] for grp in GROUPS}
    lad_g = {grp: [[0.0] * n_games for _ in K_GRID] for grp in GROUPS}
    for rec in records:
        g, grp = rec["game"], rec["group"]
        for ki, k in enumerate(K_GRID):
            mc, s = pp._market_consensus(rec["quotes"], rec["sigma0"] * k, rec["log_space"], rec["mean"])
            ll_off = score_points(rec["points"], mc, s, rec["log_space"])[0]
            off[ki][g] += ll_off
            off_g[grp][ki][g] += ll_off
            if rec["_ladder"]:
                ll_lad = score_points(rec["_ladder"], mc, s, rec["log_space"])[0]
                lad[ki][g] += ll_lad
                lad_g[grp][ki][g] += ll_lad
    return off, lad, off_g, lad_g


def shift_losses(records, n_games, fams):
    """Weight x per-group-shift grid for the given (shrink, cap) families:
    off[grp][c][g] / lad[grp][c][g] are per-game summed offered-line and
    ladder log loss for config c = (family, weight, shift), where the
    center is the production blend moved by shift * sigma."""
    cfgs = [(fam, w, a) for fam in fams for w in SHIFT_WEIGHTS for a in SHIFT_GRID]
    off = {grp: [[0.0] * n_games for _ in cfgs] for grp in GROUPS}
    lad = {grp: [[0.0] * n_games for _ in cfgs] for grp in GROUPS}
    for rec in records:
        g, grp, s = rec["game"], rec["group"], rec["sigma"]
        blend = {}
        for ci, (fam, w, a) in enumerate(cfgs):
            if (fam, w) not in blend:
                blend[(fam, w)] = pp._blend_center(rec["model_center"], rec["market_center"], s,
                                                   weight=w, cap_sigmas=fam[1], disagreement_shrink=fam[0])
            center = blend[(fam, w)] + a * s
            off[grp][ci][g] += score_points(rec["points"], center, s, rec["log_space"])[0]
            if rec["_ladder"]:
                lad[grp][ci][g] += score_points(rec["_ladder"], center, s, rec["log_space"])[0]
    return cfgs, off, lad


def shift_analysis(cfgs, src, game_counts, fams, groups_present):
    """From per-game sums (src = off or lad), for one resample: the best
    per-group shift with no history (w=0), and per family the weight that
    minimizes loss when each group also gets its own best shift."""
    totals = {grp: [sum(row[g] * c for g, c in game_counts) for row in src[grp]] for grp in groups_present}
    index = {cfg: i for i, cfg in enumerate(cfgs)}
    out = {"shift_only": {}, "w_given_shift": {}, "loss_w0_shift": 0.0}
    fam0 = fams[0]
    for grp in groups_present:
        best_a = min(SHIFT_GRID, key=lambda a: totals[grp][index[(fam0, 0.0, a)]])
        out["shift_only"][grp] = best_a
        out["loss_w0_shift"] += totals[grp][index[(fam0, 0.0, best_a)]]
    pooled_a = min(SHIFT_GRID, key=lambda a: sum(totals[grp][index[(fam0, 0.0, a)]] for grp in groups_present))
    out["shift_only"]["pooled"] = pooled_a
    for fam in fams:
        curve = {w: sum(min(totals[grp][index[(fam, w, a)]] for a in SHIFT_GRID) for grp in groups_present)
                 for w in SHIFT_WEIGHTS}
        best_w = min(curve, key=curve.get)
        out["w_given_shift"][fam] = (best_w, curve[best_w])
    return out


def regression_stats(records, n_games):
    """Per group and game: sufficient statistics for regressing the residual
    r = (actual - market center)/sigma on the gap g = (history center -
    market center)/sigma and on h = g/(1+|g|) (the shrink form's effective
    gap), plus the market's over-rate bias at the offered lines."""
    stats = {grp: [[0.0] * 11 for _ in range(n_games)] for grp in GROUPS}
    for rec in records:
        s = rec["sigma"]
        r = (rec["actual_model_space"] - rec["market_center"]) / s
        g = (rec["model_center"] - rec["market_center"]) / s
        h = g / (1 + abs(g))
        rec["_r"], rec["_g"], rec["_h"] = r, g, h
        n_pts = len(rec["points"])
        p_mkt = sum(p_over(pt, rec["market_center"], s, rec["log_space"]) for pt, _ in rec["points"]) / n_pts
        over = sum(o for _, o in rec["points"]) / n_pts
        st = stats[rec["group"]][rec["game"]]
        for i, v in enumerate((1, g, r, g * g, g * r, h, h * h, h * r, p_mkt, over, rec["actual_model_space"] > rec["market_center"])):
            st[i] += v
    return stats


def ols_from_sums(n, sg, sr, sgg, sgr, sh, shh, shr, sp, so, sabove):
    if n < 3:
        return {}
    vg = sgg - sg * sg / n
    vh = shh - sh * sh / n
    slope = (sgr - sg * sr / n) / vg if vg else None
    return {
        "n": n,
        "slope_with_intercept": slope,
        "intercept": (sr - (slope or 0) * sg) / n,
        "slope_through_origin": sgr / sgg if sgg else None,
        "shrink_form_weight_with_intercept": (shr - sh * sr / n) / vh if vh else None,
        "mean_gap": sg / n, "mean_residual": sr / n,
        "mean_p_over_market": sp / n, "over_rate": so / n, "over_bias": (so - sp) / n,
        "share_above_market_center": sabove / n,
    }


def lad_slope(xs, ys, iters=30):
    """Least-absolute-deviation slope (with intercept) by iteratively
    reweighted least squares -- the median regression, robust to the
    occasional 88-yard game from a player the market had at 36 that pulls
    an OLS slope around, and the median is what a line's 50% point is about."""
    n = len(xs)
    if n < 3:
        return None
    w = [1.0] * n
    b = 0.0
    for _ in range(iters):
        sw = sum(w)
        mx = sum(wi * x for wi, x in zip(w, xs)) / sw
        my = sum(wi * y for wi, y in zip(w, ys)) / sw
        sxx = sum(wi * (x - mx) ** 2 for wi, x in zip(w, xs))
        if sxx <= 0:
            return None
        b = sum(wi * (x - mx) * (y - my) for wi, x, y in zip(w, xs, ys)) / sxx
        a = my - b * mx
        w = [1.0 / max(abs(y - a - b * x), 1e-4) for x, y in zip(xs, ys)]
    return b


def pct(values, q):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]


def ci90(values):
    return [pct(values, CI_LO), pct(values, CI_HI)]


def fmt_ci(ci, spec="+.4f"):
    if ci[0] is None:
        return "[n/a]"
    return f"[{ci[0]:{spec}}, {ci[1]:{spec}}]"


def conservative(current, lo, hi):
    """The value nearest the current setting that the data can't rule out:
    the current value if it's inside the 90% interval, otherwise the
    interval edge closest to it."""
    if lo is None or hi is None or lo <= current <= hi:
        return current
    return lo if current < lo else hi


def fam_label(fam):
    return f"{'shrink' if fam[0] else 'fixed'} cap={_fmt_cap(fam[1])}"


def cfg_label(cfg):
    kind, w, cap, shrink = cfg
    if kind != "grid":
        return f"{kind}-only"
    return f"w={w:.2f} {fam_label((shrink, cap))}"


def argmin(ids, totals):
    return min(ids, key=lambda i: totals[i])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boards-file", help="JSON dump of app_cache event_odds_* rows [{key, cached_at, data}] instead of reading Supabase")
    ap.add_argument("--out", help="also write the full results as JSON here")
    ap.add_argument("--bootstrap", type=int, default=2000, help="game-clustered bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---------------- data ----------------
    counts = Counter()
    raw_boards, source = load_boards(args.boards_file)
    print(f"boards: {len(raw_boards)} event_odds rows from {source}", flush=True)
    seasons = set()
    for row in raw_boards:
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        ts = _parse_iso(data.get("commence_time"))
        if ts:
            seasons.add(_season_of(ts))
    player_rows, team_rows, games_by_season = {}, {}, {}
    for season in sorted(seasons):
        for s in (season - 1, season):
            if s not in player_rows:
                player_rows[s] = nflverse_client._fetch_player_week_rows(s)
                team_rows[s] = nflverse_client._fetch_team_week_rows(s)
        games_by_season[season] = completed_games(team_rows[season])
        print(f"nflverse {season} (cached): {len(player_rows[season])} player-week rows, "
              f"{len(games_by_season[season])} completed games; {season - 1}: {len(player_rows[season - 1])} rows", flush=True)

    games = select_boards(raw_boards, games_by_season, counts)
    records = build_records(games, player_rows, team_rows, counts)
    unmatched = counts.pop("_unmatched_names", Counter())
    if not records:
        print("no scorable player-markets -- nothing to fit")
        return
    for rec in records:
        rec["_ladder"] = ladder_points(rec)
    n_games = len(games)
    print(f"scoring {len(records)} player-markets in {n_games} games...", flush=True)

    # ---------------- per-game sums ----------------
    configs = build_configs()
    S = per_game_losses(records, configs, n_games)
    L, Bm, LL, N, N_lad = S["off_L"], S["off_B"], S["lad_L"], S["N"], S["N_lad"]
    total_n, total_lad = sum(N), sum(N_lad)
    i_market, i_history = 0, 1
    idx = {cfg: i for i, cfg in enumerate(configs)}
    i_prod = idx[("grid", CURRENT_WEIGHT, CURRENT_CAP, CURRENT_SHRINK)]
    grid_ids = [i for i, c in enumerate(configs) if c[0] == "grid"]
    families = [(shrink, cap) for shrink in SHRINK_FORMS for cap in CAP_GRID]
    fam_ids = {fam: [idx[("grid", w, fam[1], fam[0])] for w in WEIGHT_GRID] for fam in families}
    current_family = (CURRENT_SHRINK, CURRENT_CAP)
    fixed_nocap = (False, math.inf)
    focus_fams = [current_family, fixed_nocap]  # the two the per-group / ladder fits report

    tot_L = [sum(r) / total_n for r in L]
    tot_B = [sum(r) / total_n for r in Bm]
    tot_LL = [sum(r) / total_lad for r in LL]
    best_overall = argmin(grid_ids, tot_L)
    best_brier = argmin(grid_ids, tot_B)
    fam_best = {fam: argmin(ids, tot_L) for fam, ids in fam_ids.items()}
    fam_best_ladder = {fam: argmin(fam_ids[fam], tot_LL) for fam in focus_fams}
    n_group = {grp: sum(S["Ng"][grp]) for grp in GROUPS}
    grp_tot = {grp: [sum(r) / n_group[grp] for r in S["grp_L"][grp]] if n_group[grp] else None for grp in GROUPS}

    s_off, s_lad, s_off_g, s_lad_g = sigma_losses(records, n_games)
    k_tot_off = [sum(r) / total_n for r in s_off]
    k_tot_lad = [sum(r) / total_lad for r in s_lad]

    groups_present = [grp for grp in GROUPS if n_group[grp]]
    sh_cfgs, sh_off, sh_lad = shift_losses(records, n_games, focus_fams)
    all_games = [(g, 1) for g in range(n_games)]
    sh_point = {"offered": shift_analysis(sh_cfgs, sh_off, all_games, focus_fams, groups_present),
                "ladder": shift_analysis(sh_cfgs, sh_lad, all_games, focus_fams, groups_present)}

    reg = regression_stats(records, n_games)

    def reg_summary(game_counts):
        out, pooled = {}, [0.0] * 11
        for grp in GROUPS:
            sums = [0.0] * 11
            for g, c in game_counts:
                for i, v in enumerate(reg[grp][g]):
                    sums[i] += c * v
            out[grp] = ols_from_sums(*sums)
            pooled = [a + b for a, b in zip(pooled, sums)]
        out["pooled"] = ols_from_sums(*pooled)
        return out

    def lad_summary(recs):
        return {grp: lad_slope([r["_g"] for r in recs if grp == "pooled" or r["group"] == grp],
                               [r["_r"] for r in recs if grp == "pooled" or r["group"] == grp])
                for grp in GROUPS + ("pooled",)}

    reg_point = reg_summary([(g, 1) for g in range(n_games)])
    lad_point = lad_summary(records)
    recs_by_game = defaultdict(list)
    for r in records:
        recs_by_game[r["game"]].append(r)

    # ---------------- bootstrap (resample whole games) ----------------
    print(f"bootstrapping {args.bootstrap} game-clustered resamples...", flush=True)
    boot = defaultdict(list)
    lad_reps = min(args.bootstrap, 500)
    shift_reps = min(args.bootstrap, 1000)
    k_range = range(len(K_GRID))
    for b in range(args.bootstrap):
        gc = list(Counter(rng.choices(range(n_games), k=n_games)).items())
        nb = sum(N[g] * c for g, c in gc)
        if nb == 0:
            continue
        T = [sum(row[g] * c for g, c in gc) for row in L]
        boot["d_prod"].append((T[i_prod] - T[i_market]) / nb)
        boot["d_hist"].append((T[i_history] - T[i_market]) / nb)
        bo = argmin(grid_ids, T)
        boot["best_overall"].append(configs[bo])
        boot["d_best"].append((T[bo] - T[i_market]) / nb)
        fam_loss = {}
        for fam, ids in fam_ids.items():
            bi = argmin(ids, T)
            fam_loss[fam] = T[bi]
            boot[("w", fam)].append(configs[bi][1])
        for fam in families:
            boot[("beats_current", fam)].append(fam_loss[fam] < fam_loss[current_family] - 1e-12)
        TL = [sum(row[g] * c for g, c in gc) for row in LL]
        for fam in focus_fams:
            boot[("w_ladder", fam)].append(configs[argmin(fam_ids[fam], TL)][1])
        for grp in GROUPS:
            if sum(S["Ng"][grp][g] * c for g, c in gc) == 0:
                continue
            for fam in focus_fams:
                TG = {i: sum(S["grp_L"][grp][i][g] * c for g, c in gc) for i in fam_ids[fam]}
                boot[("w_group", grp, fam)].append(configs[min(TG, key=TG.get)][1])
            for name, src in (("k_ladder", s_lad_g[grp]), ("k_offered", s_off_g[grp])):
                tk = [sum(row[g] * c for g, c in gc) for row in src]
                boot[(name, grp)].append(K_GRID[min(k_range, key=lambda i: tk[i])])
        for name, src in (("k_offered", s_off), ("k_ladder", s_lad)):
            tk = [sum(row[g] * c for g, c in gc) for row in src]
            boot[name].append(K_GRID[min(k_range, key=lambda i: tk[i])])
        for grp, d in reg_summary(gc).items():
            for key in ("slope_with_intercept", "slope_through_origin", "shrink_form_weight_with_intercept", "over_bias"):
                if d.get(key) is not None:
                    boot[("reg", grp, key)].append(d[key])
        if b < lad_reps:
            recs = [r for g, c in gc for r in recs_by_game[g] for _ in range(c)]
            for grp, v in lad_summary(recs).items():
                if v is not None:
                    boot[("lad", grp)].append(v)
        if b < shift_reps:
            present = [grp for grp in groups_present if sum(S["Ng"][grp][g] * c for g, c in gc)]
            for name, src in (("offered", sh_off), ("ladder", sh_lad)):
                res = shift_analysis(sh_cfgs, src, gc, focus_fams, present)
                for grp, a in res["shift_only"].items():
                    boot[("shift", name, grp)].append(a)
                for fam, (w, _loss) in res["w_given_shift"].items():
                    boot[("w_given_shift", name, fam)].append(w)

    def share(key):
        vals = boot[key]
        return sum(vals) / len(vals) if vals else 0.0

    # ---------------- cross-validation (weight refit inside each fold) ----------------
    weeks = sorted({g["week"] for g in games})
    week_of_game = [g["week"] for g in games]

    def cv(fold_of, procedure_ids, fit=None, evaluate=None, n_eval=None):
        """Refit (argmin over procedure_ids on the `fit` sums) on everything
        outside each fold, score the held-out fold on the `evaluate` sums
        (both default to offered-line log loss). A procedure with ONE
        candidate (the current constants, market-only) fits nothing, so its
        CV score is just its plain score -- already out of sample."""
        fit, evaluate, n_eval = fit or L, evaluate or L, n_eval or N
        folds = defaultdict(list)
        for g in range(n_games):
            folds[fold_of(g)].append(g)
        held, held_n, chosen = 0.0, 0, {}
        for fold, gs in sorted(folds.items()):
            if not any(n_eval[g] for g in gs):
                continue
            out = set(gs)
            train = {i: sum(fit[i][g] for g in range(n_games) if g not in out) for i in procedure_ids}
            pick = min(procedure_ids, key=lambda i: train[i])
            held += sum(evaluate[pick][g] for g in gs)
            held_n += sum(n_eval[g] for g in gs)
            chosen[fold] = configs[pick]
        return held / held_n, chosen

    procedures = {"market-only": [i_market], "history-only": [i_history],
                  "current constants (fixed)": [i_prod], "full grid (refit)": grid_ids}
    for fam, ids in fam_ids.items():
        procedures[f"{fam_label(fam)} (w refit)"] = ids
    cv_results = {}
    for name, ids in procedures.items():
        logo, logo_chosen = cv(lambda g: g, ids)
        lowo, lowo_chosen = cv(lambda g: week_of_game[g], ids)
        cv_results[name] = {
            "logo_ll": logo, "lowo_ll": lowo,
            "lowo_chosen_per_heldout_week": {wk: cfg_label(c) for wk, c in lowo_chosen.items()},
            "logo_weights_chosen": dict(Counter(c[1] for c in logo_chosen.values())),
        }
    # The ladder fit has more power (several lines per outcome), so the
    # sharper question is whether a weight fitted THERE also holds up at
    # the real offered lines on held-out games -- the decision metric.
    cv_ladder = {}
    for fam in focus_fams:
        for metric, ev, ne in (("offered", L, N), ("ladder", LL, N_lad)):
            logo, logo_chosen = cv(lambda g: g, fam_ids[fam], fit=LL, evaluate=ev, n_eval=ne)
            lowo, lowo_chosen = cv(lambda g: week_of_game[g], fam_ids[fam], fit=LL, evaluate=ev, n_eval=ne)
            cv_ladder[f"{fam_label(fam)}: fit on ladder, score {metric}"] = {
                "logo_ll": logo, "lowo_ll": lowo,
                "lowo_chosen_per_heldout_week": {wk: cfg_label(c) for wk, c in lowo_chosen.items()},
                "logo_weights_chosen": dict(Counter(c[1] for c in logo_chosen.values())),
            }

    # ---------------- splits (descriptive) ----------------
    def split_summary(game_set):
        n = sum(N[g] for g in game_set)
        if not n:
            return None
        t = lambda i: sum(L[i][g] for g in game_set) / n  # noqa: E731
        out = {"games": len(game_set), "player_markets": n, "market": t(i_market),
               "production": t(i_prod), "history": t(i_history)}
        for fam in focus_fams:
            bi = min(fam_ids[fam], key=t)
            out[f"best_w_{fam_label(fam)}"] = configs[bi][1]
            out[f"best_ll_{fam_label(fam)}"] = t(bi)
        return out

    splits = {}
    for wk in weeks:
        splits[f"week {wk}"] = split_summary([g for g in range(n_games) if week_of_game[g] == wk])
    hours = [(g["commence"] - g["cached_at"]) / 3600 for g in games]
    splits["board <= 6h before kickoff"] = split_summary([g for g in range(n_games) if hours[g] <= 6])
    splits["board > 6h before kickoff"] = split_summary([g for g in range(n_games) if hours[g] > 6])

    # ---------------- recommendation ----------------
    # The current constants were fitted on none of this data, so their plain
    # score is already out of sample. A re-tune earns a change only if
    # re-fitting it (inside each held-out fold) beats that.
    current_score = tot_L[i_prod]
    fam_cv = {fam: cv_results[f"{fam_label(fam)} (w refit)"]["logo_ll"] for fam in families}
    cv_best_fam = min(families, key=lambda fam: fam_cv[fam])
    gate = fam_cv[cv_best_fam] < current_score
    beat = share(("beats_current", cv_best_fam))
    if gate and cv_best_fam != current_family and beat >= SWITCH_FAMILY_BOOT_SHARE:
        rec_family = cv_best_fam
        family_reason = (f"switch: {fam_label(cv_best_fam)} refit beats the current constants out of sample "
                         f"(LOGO {fam_cv[cv_best_fam]:.5f} < {current_score:.5f}) and the current family in {beat:.0%} of resamples")
    else:
        rec_family = current_family
        family_reason = (f"keep: best family by LOGO CV is {fam_label(cv_best_fam)} at {fam_cv[cv_best_fam]:.5f} vs the "
                         f"current constants' {current_score:.5f} ({'beats' if gate else 'does NOT beat'} them out of sample) "
                         f"and beats the current family in {beat:.0%} of resamples (switch needs both, >= {SWITCH_FAMILY_BOOT_SHARE:.0%})")
    w_point = configs[fam_best[rec_family]][1]
    w_ci = ci90(boot[("w", rec_family)])
    w_rec = conservative(CURRENT_WEIGHT, *w_ci)
    cap_binding = w_rec > rec_family[1] if rec_family[0] else None
    k_point = K_GRID[min(k_range, key=lambda i: k_tot_lad[i])]
    k_ci = ci90(boot["k_ladder"])
    k_rec = conservative(CURRENT_K, *k_ci)
    k_group = {}
    for grp in GROUPS:
        if n_group[grp]:
            kp = K_GRID[min(k_range, key=lambda i: sum(s_lad_g[grp][i]))]
            kc = ci90(boot[("k_ladder", grp)])
            k_group[grp] = {"ladder_best": kp, "ci90": kc, "recommended": conservative(CURRENT_K, *kc),
                            "offered_ci90": ci90(boot[("k_offered", grp)])}
    w_group = {}
    for grp in GROUPS:
        if n_group[grp]:
            w_group[grp] = {}
            for fam in focus_fams:
                bi = min(fam_ids[fam], key=lambda i: grp_tot[grp][i])
                wc = ci90(boot[("w_group", grp, fam)])
                w_group[grp][fam_label(fam)] = {"best_w": configs[bi][1], "ci90": wc, "ll": grp_tot[grp][bi],
                                                "recommended": conservative(CURRENT_WEIGHT, *wc)}

    # ---------------- print ----------------
    print("\n=== sample ===")
    for k_, v in sorted(counts.items()):
        print(f"  {k_:62s} {v}")
    print(f"  => games: {n_games} (weeks {weeks}), player-markets: {len(records)}, "
          f"line evaluations: {sum(len(r['points']) for r in records)}, off-center ladder lines: "
          f"{sum(len(r['_ladder']) for r in records)}")
    print(f"  by group: {n_group};  by market: {dict(Counter(r['market'] for r in records))}")
    print(f"  books quoting both sides per player-market: median {sorted(r['n_quotes'] for r in records)[len(records) // 2]}; "
          f"board age at fetch: median {sorted(hours)[len(hours) // 2]:.1f}h before kickoff (range {min(hours):.1f}-{max(hours):.1f}h)")
    if unmatched:
        print(f"  names on the board with no nflverse match (production skips these too): {unmatched.most_common(10)}")

    def line(label, i):
        return f"  {label:40s} {tot_L[i]:.5f}  {tot_B[i]:.5f}  {tot_L[i] - tot_L[i_market]:+.5f}  {tot_LL[i]:.5f}"

    print("\n=== at the offer book's lines (each player-market weighted once); last column = off-center ladder log loss ===")
    print(f"  {'model':40s} {'logloss':>7s}  {'Brier':>7s}  {'dLL mkt':>8s}  {'ladderLL':>8s}")
    print(line("market-only", i_market))
    print(line("history-only", i_history))
    print(line(f"current: {cfg_label(configs[i_prod])}", i_prod))
    print(line(f"in-sample best: {cfg_label(configs[best_overall])}", best_overall))
    print(f"  Brier-best grid point: {cfg_label(configs[best_brier])};  in-sample best is chosen on this data (optimistic)")
    print(f"  dLL current - market  {tot_L[i_prod] - tot_L[i_market]:+.5f}  90% CI {fmt_ci(ci90(boot['d_prod']))}")
    print(f"  dLL history - market  {tot_L[i_history] - tot_L[i_market]:+.5f}  90% CI {fmt_ci(ci90(boot['d_hist']))}")
    print(f"  dLL best - market     {tot_L[best_overall] - tot_L[i_market]:+.5f}  90% CI {fmt_ci(ci90(boot['d_best']))}")
    print(f"  bootstrap picks of the best grid point: "
          f"{[(cfg_label(c), n) for c, n in Counter(boot['best_overall']).most_common(5)]}")

    print("\n=== weight per (form, cap) family: in-sample best, bootstrap 90% CI, cross-validated (w refit per fold) ===")
    print(f"  {'family':18s} {'best w':>6s} {'logloss':>8s} {'w 90% CI':>14s} {'LOGO CV':>8s} {'LOWO CV':>8s} {'beats current fam':>17s}")
    for fam in families:
        bi = fam_best[fam]
        name = f"{fam_label(fam)} (w refit)"
        print(f"  {fam_label(fam):18s} {configs[bi][1]:6.2f} {tot_L[bi]:8.5f} {fmt_ci(ci90(boot[('w', fam)]), '.2f'):>14s} "
              f"{cv_results[name]['logo_ll']:8.5f} {cv_results[name]['lowo_ll']:8.5f} {share(('beats_current', fam)):16.0%}")
    for name in ("market-only", "history-only", "current constants (fixed)", "full grid (refit)"):
        print(f"  {name:47s} {cv_results[name]['logo_ll']:8.5f} {cv_results[name]['lowo_ll']:8.5f}")
    print("  LOWO pick per held-out week (fit on the other weeks):")
    for name in (f"{fam_label(current_family)} (w refit)", f"{fam_label(fixed_nocap)} (w refit)", "full grid (refit)"):
        print(f"    {name}: {cv_results[name]['lowo_chosen_per_heldout_week']}")

    print(f"  weight fitted on the ladder, scored on held-out games (current constants: offered {tot_L[i_prod]:.5f}, "
          f"ladder {tot_LL[i_prod]:.5f}; market-only: offered {tot_L[i_market]:.5f}, ladder {tot_LL[i_market]:.5f}):")
    for name, d in cv_ladder.items():
        print(f"    {name:44s} LOGO {d['logo_ll']:.5f}  LOWO {d['lowo_ll']:.5f}  LOWO picks {d['lowo_chosen_per_heldout_week']}")

    print("\n=== weight curve (offered-line log loss | ladder log loss) ===")
    print(f"  {'w':>5s}  {'shrink cap .25':>17s}  {'shrink no cap':>17s}  {'fixed no cap':>17s}")
    for w in (0.0, 0.04, 0.1, 0.16, 0.2, 0.26, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        cells = []
        for fam in ((True, CURRENT_CAP), (True, math.inf), fixed_nocap):
            i = idx[("grid", w, fam[1], fam[0])]
            cells.append(f"{tot_L[i]:.5f}|{tot_LL[i]:.5f}")
        print(f"  {w:5.2f}  {cells[0]:>17s}  {cells[1]:>17s}  {cells[2]:>17s}")
    for fam in focus_fams:
        bi = fam_best_ladder[fam]
        print(f"  ladder-scored best w, {fam_label(fam)}: {configs[bi][1]:.2f}  90% CI {fmt_ci(ci90(boot[('w_ladder', fam)]), '.2f')}")

    print("\n=== by group (offered-line log loss) ===")
    for grp in GROUPS:
        if not n_group[grp]:
            continue
        t = grp_tot[grp]
        print(f"  {grp:10s} n={n_group[grp]:4d}  market {t[i_market]:.5f}  current {t[i_prod]:.5f}  history {t[i_history]:.5f}")
        for fam in focus_fams:
            d = w_group[grp][fam_label(fam)]
            print(f"             best w ({fam_label(fam)}) {d['best_w']:.2f} -> {d['ll']:.5f}, 90% CI {fmt_ci(d['ci90'], '.2f')}")

    print("\n=== splits (offered-line log loss; descriptive, no CI) ===")
    for name, d in splits.items():
        if d:
            print(f"  {name:28s} games={d['games']:2d} n={d['player_markets']:4d}  market {d['market']:.5f}  current {d['production']:.5f}  "
                  f"history {d['history']:.5f}  best w shrink/.25 {d[f'best_w_{fam_label(current_family)}']:.2f}  "
                  f"fixed/none {d[f'best_w_{fam_label(fixed_nocap)}']:.2f}")

    print("\n=== when the history model disagreed with the market at the offered line ===")
    print(f"  {'P(hist)-P(mkt), over':>22s} {'n':>6s} {'market':>7s} {'history':>8s} {'current':>8s} {'actual':>7s}")
    disagreement = []
    for lo, hi in ((-1, -0.15), (-0.15, -0.08), (-0.08, -0.03), (-0.03, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 1.01)):
        acc = [0.0] * 5
        for r in records:
            prod_c = config_center(configs[i_prod], r)
            for point, over in r["points"]:
                pm = p_over(point, r["market_center"], r["sigma"], r["log_space"])
                ph = p_over(point, r["model_center"], r["sigma"], r["log_space"])
                if lo <= ph - pm < hi:
                    wt = 1 / len(r["points"])
                    for i, v in enumerate((1, pm, ph, p_over(point, prod_c, r["sigma"], r["log_space"]), over)):
                        acc[i] += wt * v
        if acc[0]:
            d = {"bucket": [lo, hi], "n": acc[0], "market": acc[1] / acc[0], "history": acc[2] / acc[0],
                 "current": acc[3] / acc[0], "actual_over_rate": acc[4] / acc[0]}
            disagreement.append(d)
            print(f"  {lo:+.2f} to {min(hi, 1):+.2f}        {acc[0]:6.0f} {d['market']:7.3f} {d['history']:8.3f} "
                  f"{d['current']:8.3f} {d['actual_over_rate']:7.3f}")

    print("\n=== market calibration in the large (offered lines) ===")
    for grp in GROUPS + ("pooled",):
        d = reg_point.get(grp)
        if d:
            print(f"  {grp:10s} mean P(over) {d['mean_p_over_market']:.3f}  over rate {d['over_rate']:.3f}  "
                  f"bias {d['over_bias']:+.3f} 90% CI {fmt_ci(ci90(boot[('reg', grp, 'over_bias')]), '+.3f')}  "
                  f"results above market center {d['share_above_market_center']:.3f}")

    print("\n=== history weight after a per-group center shift (separates player-specific info from a generic Under lean) ===")
    shift_json = {}
    for name, denom, base in (("offered", total_n, tot_L[i_market]), ("ladder", total_lad, tot_LL[i_market])):
        sp = sh_point[name]
        shifts = {grp: {"best_shift_sigmas": a, "ci90": ci90(boot[("shift", name, grp)])} for grp, a in sp["shift_only"].items()}
        print(f"  [{name} lines] market-only {base:.5f} -> market + per-group shift {sp['loss_w0_shift'] / denom:.5f}; best shift (sigma): "
              + ", ".join(f"{grp} {d['best_shift_sigmas']:+.2f} {fmt_ci(d['ci90'], '+.2f')}" for grp, d in shifts.items()))
        fams_json = {}
        for fam in focus_fams:
            w, loss = sp["w_given_shift"][fam]
            wc = ci90(boot[("w_given_shift", name, fam)])
            fams_json[fam_label(fam)] = {"best_w": w, "logloss": loss / denom, "w_ci90": wc}
            print(f"     + history, {fam_label(fam):16s}: best w {w:.2f} {fmt_ci(wc, '.2f')} -> {loss / denom:.5f}")
        shift_json[name] = {"market_only": base, "market_plus_shift": sp["loss_w0_shift"] / denom,
                            "shift_only": shifts, "weight_given_shift": fams_json}

    print("\n=== continuous fit: residual (actual - market center) on gap (history - market), sigma units ===")
    print(f"  {'group':10s} {'n':>4s} {'OLS slope':>9s} {'90% CI':>16s} {'thru 0':>7s} {'shrink w':>8s} "
          f"{'LAD slope':>9s} {'LAD 90% CI':>16s} {'mean gap':>8s} {'mean resid':>10s}")
    for grp in GROUPS + ("pooled",):
        d = reg_point.get(grp)
        if not d:
            continue
        print(f"  {grp:10s} {d['n']:4.0f} {d['slope_with_intercept']:9.3f} {fmt_ci(ci90(boot[('reg', grp, 'slope_with_intercept')]), '+.3f'):>16s} "
              f"{d['slope_through_origin']:7.3f} {d['shrink_form_weight_with_intercept']:8.3f} "
              f"{(lad_point[grp] if lad_point[grp] is not None else float('nan')):9.3f} {fmt_ci(ci90(boot[('lad', grp)]), '+.3f'):>16s} "
              f"{d['mean_gap']:8.3f} {d['mean_residual']:10.3f}")

    print("\n=== sigma multiplier k on production's sigma (market-only probabilities) ===")
    print(f"  {'k':>5s} {'offered LL':>10s} {'ladder LL':>10s}")
    for ki, k in enumerate(K_GRID):
        print(f"  {k:5.2f} {k_tot_off[ki]:10.5f} {k_tot_lad[ki]:10.5f}")
    k_off_point = K_GRID[min(k_range, key=lambda i: k_tot_off[i])]
    print(f"  best k at offered lines {k_off_point:.2f}, 90% CI {fmt_ci(ci90(boot['k_offered']), '.2f')} (at-the-money lines barely depend on sigma)")
    print(f"  best k on off-center ladder {k_point:.2f}, 90% CI {fmt_ci(k_ci, '.2f')}")
    for grp, d in k_group.items():
        print(f"    {grp:10s} ladder best {d['ladder_best']:.2f}, 90% CI {fmt_ci(d['ci90'], '.2f')}")

    print("\n=== recommendation (conservative: keep the current value unless the 90% CI excludes it) ===")
    print(f"  form / MAX_MODEL_TILT_SIGMAS: {fam_label(rec_family)} -- {family_reason}")
    print(f"  MODEL_WEIGHT_VS_MARKET: in-sample best {w_point:.2f}, 90% CI {fmt_ci(w_ci, '.2f')} -> {w_rec:.2f}"
          + (f"  (cap {'binds' if cap_binding else 'does not bind'} at this weight)" if cap_binding is not None else ""))
    print(f"  sigma multiplier: {k_point:.2f}, 90% CI {fmt_ci(k_ci, '.2f')} -> {k_rec:.2f}; per group "
          f"{ {g: d['recommended'] for g, d in k_group.items()} }")
    for fam in focus_fams:
        d = cv_ladder[f"{fam_label(fam)}: fit on ladder, score offered"]
        print(f"  ladder-fitted weight ({fam_label(fam)}, in-sample {configs[fam_best_ladder[fam]][1]:.2f}) at held-out offered "
              f"lines: LOGO {d['logo_ll']:.5f} vs current constants {current_score:.5f} -> "
              f"{'transfers' if d['logo_ll'] < current_score else 'does NOT transfer'} to the lines bets are made at")

    if args.out:
        rnd = lambda v: round(v, 6) if isinstance(v, float) else v  # noqa: E731
        results = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": source, "bootstrap_resamples": args.bootstrap, "lad_bootstrap_resamples": lad_reps,
            "seed": args.seed, "p_clip": P_CLIP,
            "current_constants": {"MODEL_WEIGHT_VS_MARKET": CURRENT_WEIGHT, "MAX_MODEL_TILT_SIGMAS": CURRENT_CAP,
                                  "form": "shrink", "sigma_multiplier": CURRENT_K},
            "counts": dict(counts), "unmatched_names": dict(unmatched),
            "n_games": n_games, "weeks": weeks, "n_player_markets": len(records),
            "n_line_evaluations": sum(len(r["points"]) for r in records),
            "n_ladder_lines": sum(len(r["_ladder"]) for r in records), "n_by_group": n_group,
            "games": [{"matchup": f"{g['away']} @ {g['home']}", "week": g["week"],
                       "cached_at": dt.datetime.fromtimestamp(g["cached_at"], dt.timezone.utc).isoformat(),
                       "hours_before_kickoff": round(hours[i], 1), "player_markets": N[i]} for i, g in enumerate(games)],
            "headline": {
                "market_only": {"logloss": tot_L[i_market], "brier": tot_B[i_market], "ladder_logloss": tot_LL[i_market]},
                "history_only": {"logloss": tot_L[i_history], "brier": tot_B[i_history], "ladder_logloss": tot_LL[i_history],
                                 "d_ll_vs_market_ci90": ci90(boot["d_hist"])},
                "current": {"logloss": tot_L[i_prod], "brier": tot_B[i_prod], "ladder_logloss": tot_LL[i_prod],
                            "d_ll_vs_market_ci90": ci90(boot["d_prod"])},
                "in_sample_best": {"config": cfg_label(configs[best_overall]), "logloss": tot_L[best_overall],
                                   "brier": tot_B[best_overall], "d_ll_vs_market_ci90": ci90(boot["d_best"]),
                                   "bootstrap_picks": [[cfg_label(c), n] for c, n in Counter(boot["best_overall"]).most_common(10)]},
                "brier_best": {"config": cfg_label(configs[best_brier]), "brier": tot_B[best_brier]},
            },
            "families": {fam_label(fam): {
                "best_w": configs[fam_best[fam]][1], "logloss": tot_L[fam_best[fam]], "brier": tot_B[fam_best[fam]],
                "w_ci90": ci90(boot[("w", fam)]), "beats_current_family_share": share(("beats_current", fam)),
            } for fam in families},
            "ladder_weight_fit": {fam_label(fam): {"best_w": configs[fam_best_ladder[fam]][1],
                                                   "ladder_logloss": tot_LL[fam_best_ladder[fam]],
                                                   "w_ci90": ci90(boot[("w_ladder", fam)])} for fam in focus_fams},
            "by_group_weight": w_group,
            "by_group_logloss": {grp: {"market": grp_tot[grp][i_market], "current": grp_tot[grp][i_prod],
                                       "history": grp_tot[grp][i_history]} for grp in GROUPS if n_group[grp]},
            "cv": cv_results, "cv_ladder_fit": cv_ladder, "splits": splits,
            "grid": [{"config": cfg_label(c), "weight": c[1], "cap": _fmt_cap(c[2]), "form": "shrink" if c[3] else "fixed",
                      "logloss": tot_L[i], "brier": tot_B[i], "ladder_logloss": tot_LL[i]}
                     for i, c in enumerate(configs) if c[0] == "grid"],
            "disagreement_buckets": disagreement,
            "shift_control": shift_json, "shift_bootstrap_resamples": shift_reps,
            "continuous_fit": {grp: {**reg_point[grp],
                                     **{f"{key}_ci90": ci90(boot[("reg", grp, key)]) for key in
                                        ("slope_with_intercept", "slope_through_origin",
                                         "shrink_form_weight_with_intercept", "over_bias")},
                                     "lad_slope": lad_point.get(grp), "lad_slope_ci90": ci90(boot[("lad", grp)])}
                               for grp in GROUPS + ("pooled",) if reg_point.get(grp)},
            "sigma_multiplier": {"k_grid": K_GRID, "offered_logloss": k_tot_off, "ladder_logloss": k_tot_lad,
                                 "offered_best": k_off_point, "offered_ci90": ci90(boot["k_offered"]),
                                 "ladder_best": k_point, "ladder_ci90": k_ci, "by_group": k_group,
                                 "ladder_sigmas": list(LADDER_SIGMAS)},
            "recommendation": {
                "form": "shrink" if rec_family[0] else "fixed", "MAX_MODEL_TILT_SIGMAS": _fmt_cap(rec_family[1]),
                "family_reason": family_reason, "cap_binds_at_recommended_weight": cap_binding,
                "MODEL_WEIGHT_VS_MARKET": {"in_sample_best": w_point, "ci90": w_ci, "recommended": w_rec},
                "sigma_multiplier": {"ladder_best": k_point, "ci90": k_ci, "recommended": k_rec,
                                     "by_group": {g: d["recommended"] for g, d in k_group.items()}},
            },
            "records": [{k_: rnd(v) for k_, v in r.items() if not k_.startswith("_") and k_ != "quotes"} for r in records],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
