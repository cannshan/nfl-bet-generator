"""Player prop candidates: live sportsbook prop lines (via The Odds API's
per-event endpoint), priced by anchoring on the MARKET's own consensus and
tilting it only slightly toward this app's player-history projection.

Why market-anchored (the single most important design decision here): the
sportsbook line for a player prop is, by a wide margin, the best available
predictor of that stat. It already reflects everything a public stats feed
can see (matchups, weather, recency) plus things it can't (depth-chart
changes, snap-count plans, practice reports, sharp money). A projection
built from a player's own recent box scores has NO demonstrated edge over
it -- and this app's own track record proved that the hard way: the first
19 settled prop suggestions were shown at an average 75% model confidence
(vs. ~50% market-implied) and hit 37% of the time. Fitting a shrinkage
weight to those outcomes put the optimal weight on the history model at
zero. The mechanism was plain to see in the legs themselves: a backup RB
with a near-zero 2025 game log got "92% Under 7.5 rush yards" while the
book -- who knew his role had changed -- priced it at 52%.

So the probability for every leg is now:

  1. a consensus market center for the player's stat, solved from every
     book's devigged two-sided quote (a book quoting Over 271.5 at 45% is
     saying the median is a bit under 271.5; every quote is one estimate of
     the center, and the median across books is the consensus);
  2. tilted a small, capped amount (MODEL_WEIGHT_VS_MARKET, MAX_MODEL_TILT_SIGMAS)
     toward the history projection -- enough for genuine, fresh signals
     (injury-return workload, a backup QB, weather) to nudge the number,
     never enough to let a stale game log overrule the book;
  3. priced at the specific line/side being offered, using the player's own
     historical dispersion (the one thing the game log estimates well).

"Edge" is then model probability minus the BREAKEVEN probability at the
offered price (1 / decimal odds), i.e. actual expected value after vig --
so most of the surviving edge is line/price shopping across books, which is
the one edge a retail bettor reliably has.

The history projection itself still adjusts for the things listed below,
but note each is a small tilt on top of the market, not the whole answer:

- opponent's run/pass defense (yardage fields only, regressed toward the
  league average -- a 20% swing off a few games is noise, not signal)
- recent role trend (target share vs. own baseline)
- recency (a player's last few games count more than early-season/last-year)
- weather at the stadium (wind/precipitation hurt the passing game)
- the opponent defense's fresh injuries
- coming back from a significant injury (workload-managed early games)
- the player's own starting QB being out

Small-sample players (fewer than MIN_GAMES_WEIGHT effective games) are
skipped rather than guessed at.
"""
import math
import re
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist, median
from app import odds_client, nflverse_client, ratings, odds_math, espn_client, injury_client, weather_client, roster_client

# ESPN's comments are written like news blurbs ("X said Wednesday that...,
# Reporter Name of Outlet reports.") -- strip the trailing attribution clause
# and hard-cap length so notes stay skimmable instead of reading like an article.
_TRAILING_ATTRIBUTION_RE = re.compile(r",\s*[^,]+ reports?\.?\s*$", re.IGNORECASE)
COMMENT_MAX_LEN = 140
_NORMAL = NormalDist()


def _simplify_comment(text):
    if not text:
        return text
    text = _TRAILING_ATTRIBUTION_RE.sub("", text).strip().rstrip(",").strip()
    if len(text) > COMMENT_MAX_LEN:
        text = text[:COMMENT_MAX_LEN].rsplit(" ", 1)[0] + "…"
    return text


def _extract_keyword_sentence(text, keywords):
    """Pulls out just the sentence containing a trigger keyword, rather than
    the start of a long comment -- the relevant detail (e.g. "tore his ACL")
    is often buried well past where a flat character-length truncation would
    cut off."""
    if not text:
        return text
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if any(kw in sentence.lower() for kw in keywords):
            return sentence.strip()
    return text


MIN_GAMES_WEIGHT = 3.0

# How far the final probability moves from the market consensus toward the
# history projection. 0 = pure market, 1 = pure history model. Fitting this
# to the app's own settled suggestions (see module docstring) put the
# optimum at 0; published work on prop markets puts a decent independent
# model in the 0.05-0.15 range. 0.10 keeps real fresh signals (injury
# returns, backup QBs, weather) alive as small nudges while making it
# impossible for a stale game log to manufacture a 40-point "edge." The
# Track Record page is the arbiter: if bets keep hitting below their shown
# probability, this should go DOWN, not up.
MODEL_WEIGHT_VS_MARKET = 0.10
# Hard cap on that tilt, in units of the player's own game-to-game std --
# a second guard for the exact failure mode above (a projection several std
# away from the market is the model missing information, not finding it).
MAX_MODEL_TILT_SIGMAS = 0.25
# The tilt weight also shrinks as the history projection and the market
# disagree more: weight = MODEL_WEIGHT_VS_MARKET / (1 + |gap| / sigma). A
# projection a full std away from the book's number is far more likely to
# be a stale game log (a role that changed) than a real insight, so the
# further apart they are, the LESS the history counts -- the opposite of
# what a fixed weight does. A projection right next to the market keeps
# the full 10%.
# A player's game-to-game std from his own history describes the ROLE he
# had then. When the market center sits well above his history mean (a
# backup promoted, a rookie's role growing), that std is far too small for
# the role he has now -- and a too-small std makes even the capped 10%
# tilt swing the probability by several points. So for the raw-normal
# fields the std is scaled up in proportion to how far the market center
# exceeds the history mean (dispersion of yardage/volume stats scales with
# level). It is never scaled DOWN: a market center below history keeps the
# wider history std. Log-space fields are already scale-free and skip this.
SIGMA_SCALES_WITH_MARKET = True

# Opponent-defense matchup: applied only to YARDAGE fields (a defense's
# yards-allowed says something about efficiency against it, nothing about
# how many times a QB will drop back), and regressed toward league average
# since a handful of games of yards-allowed is mostly noise. Previously a
# raw, unshrunk ratio clamped at +-25% was applied to attempts/completions
# too -- that single factor turned Jared Goff's honest 55% Under 35.5
# attempts into an 85% "lock" off one game of Buffalo's pass defense.
MATCHUP_SHRINK = 0.35
MATCHUP_ADJUSTMENT_CLAMP = (0.90, 1.10)
YARDAGE_FIELDS = {"passing_yards", "rushing_yards", "receiving_yards"}
ROLE_TREND_CLAMP = (0.85, 1.15)
RECENT_GAMES_FOR_TREND = 3
RECENCY_DECAY_PER_WEEK = 0.90  # within the current season, each week further back counts ~10% less
# Log-scale std, fit walk-forward against ~31,000 real player-games
# (2018-2025) including the matchup+role-trend adjustments, ran slightly
# overconfident (~65% of outcomes landed within +-1 std, vs. the ~68% a
# well-calibrated model should show). This constant widens it back to
# match: the empirical z capturing 68.27% of real outcomes was ~1.06-1.11
# across receiving_yards/receptions, so 1.1 corrects both without needing
# a separate constant per field.
LOG_STD_CALIBRATION = 1.1

DEFENSIVE_INJURY_NEW_ABSENCE_STATUSES = {"Out", "Doubtful"}
PER_MISSING_DEFENDER_BONUS = 0.03
MAX_DEFENSIVE_INJURY_BONUS = 0.15
MIN_DEFENSIVE_INVOLVEMENT = 2.5  # rough "actually plays meaningful defensive snaps" floor

# Injury-report comments mentioning these terms (regardless of current game
# status -- this includes fully-cleared "Active" players) suggest a workload
# management situation a season-average stat line can't see coming.
MAJOR_INJURY_RETURN_KEYWORDS = (
    "acl", "achilles", "hamstring", "torn", "tore", "surgery",
    "recovery", "recovering", "return from", "returning from",
)
# A lower-body injury mainly threatens mobility/rushing; passing/arm-based
# categories get a much lighter haircut since there's less reason to expect
# a throwing-motion impact from, say, a torn ACL.
MAJOR_INJURY_RETURN_DISCOUNT = {"rush": 0.80, "pass": 0.95}

# A backup QB is a real, if modest, drag on every pass-catcher's numbers.
QB_OUT_RECEIVING_DISCOUNT = 0.90

# market key -> (stat field in nflverse player rows, display label, weather/defense category)
MARKET_CONFIG = {
    "player_pass_yds": ("passing_yards", "Pass Yds", "pass"),
    "player_pass_attempts": ("attempts", "Pass Attempts", "pass"),
    "player_pass_completions": ("completions", "Completions", "pass"),
    "player_pass_tds": ("passing_tds", "Pass TDs", "pass"),
    "player_rush_yds": ("rushing_yards", "Rush Yds", "rush"),
    "player_reception_yds": ("receiving_yards", "Rec Yds", "pass"),
    "player_receptions": ("receptions", "Receptions", "pass"),
}
RECEIVING_FIELDS = {"receiving_yards", "receptions"}

DEFAULT_MARKETS = ",".join(MARKET_CONFIG.keys())


def _mentions_major_injury_return(injury):
    """Checks ESPN's longer comment, not just the (often terse) short one --
    seen in practice: shortComment said only '(knee) does not have an injury
    designation', while longComment specified a torn ACL/LCL and named the
    exact mobility limitation."""
    if not injury:
        return False
    text = (injury.get("long_comment") or injury.get("comment") or "").lower()
    return any(kw in text for kw in MAJOR_INJURY_RETURN_KEYWORDS)


def _weighted_mean_std(rows_with_weight, field):
    values, weights = [], []
    for row, w in rows_with_weight:
        try:
            v = float(row.get(field) or 0)
        except (TypeError, ValueError):
            continue
        values.append(v)
        weights.append(w)
    total_w = sum(weights)
    if total_w < MIN_GAMES_WEIGHT:
        return None, None
    mean = sum(v * w for v, w in zip(values, weights)) / total_w
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return mean, variance ** 0.5


def _weighted_mean_std_log(rows_with_weight, field):
    """Same as _weighted_mean_std, but on log(value + 1) -- receiving yards
    and receptions are real-world right-skewed (most games land BELOW a
    player's average, with occasional big-game outliers pulling the average
    up), not the symmetric normal distribution _weighted_mean_std assumes.

    Backtested walk-forward against ~31,000 real player-games (2018-2025,
    WITH the matchup + role-trend adjustments applied, not just the bare
    projection): the raw-normal model put only 37-41% of actual outcomes
    above its own projected mean (should be ~50% if the shape is right, not
    just the average); the log-transformed version corrected that to
    50-52%. Passing and rushing yards did NOT show the same bias under this
    transform (still 35-37% above mean, a different-shaped skew, likely
    game-script driven) -- so this is only used for RECEIVING_FIELDS, not
    applied blanket to every category."""
    values, weights = [], []
    for row, w in rows_with_weight:
        try:
            v = float(row.get(field) or 0)
        except (TypeError, ValueError):
            continue
        values.append(math.log(max(v, 0) + 1))
        weights.append(w)
    total_w = sum(weights)
    if total_w < MIN_GAMES_WEIGHT:
        return None, None
    mean = sum(v * w for v, w in zip(values, weights)) / total_w
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return mean, (variance ** 0.5) * LOG_STD_CALIBRATION


def _apply_recency(rows_with_weight):
    """Within the current season, weight recent games more than early ones
    (a proxy for "current form" / role change independent of injury reports).
    Prior-season rows keep their flat 0.5 weight -- the season-level discount
    already handles that decay."""
    current_weeks = []
    for row, w in rows_with_weight:
        if w >= 1.0:
            try:
                current_weeks.append(int(row.get("week")))
            except (TypeError, ValueError):
                pass
    if not current_weeks:
        return rows_with_weight
    max_week = max(current_weeks)

    adjusted = []
    for row, w in rows_with_weight:
        if w >= 1.0:
            try:
                weeks_back = max(0, max_week - int(row.get("week")))
            except (TypeError, ValueError):
                weeks_back = 0
            w = w * (RECENCY_DECAY_PER_WEEK ** weeks_back)
        adjusted.append((row, w))
    return adjusted


def _role_trend_factor(rows_with_weight):
    """Compares a player's target share over their last few current-season
    games to their own overall baseline -- a real, data-driven "bigger or
    smaller role" signal, independent of what any injury report says."""
    def _target_share(row):
        try:
            return float(row.get("target_share") or 0)
        except (TypeError, ValueError):
            return None

    baseline_vals = [(ts, w) for (row, w) in rows_with_weight if (ts := _target_share(row)) is not None and ts > 0]
    if not baseline_vals:
        return 1.0
    baseline_w = sum(w for _, w in baseline_vals)
    baseline = sum(ts * w for ts, w in baseline_vals) / baseline_w

    current_rows = sorted(
        [(row, w) for row, w in rows_with_weight if w >= 1.0],
        key=lambda rw: int(rw[0].get("week") or 0),
        reverse=True,
    )[:RECENT_GAMES_FOR_TREND]
    recent_vals = [ts for row, _w in current_rows if (ts := _target_share(row)) is not None and ts > 0]
    if not recent_vals or baseline <= 0:
        return 1.0

    recent = sum(recent_vals) / len(recent_vals)
    factor = recent / baseline
    return max(ROLE_TREND_CLAMP[0], min(ROLE_TREND_CLAMP[1], factor))


def _compute_category_reliability(player_index):
    """Average, across all players with enough games, of each stat category's
    own week-to-week coefficient of variation (std/mean). Lower = more
    consistent = a more statistically reliable category to bet on at the
    same edge. Computed fresh each refresh (cheap: pure in-memory math over
    data already fetched)."""
    cv_lists = {field: [] for field, _label, _cat in MARKET_CONFIG.values()}
    for rows in player_index.values():
        for field in cv_lists:
            mean, std = _weighted_mean_std(rows, field)
            if mean and mean > 3 and std is not None:
                cv_lists[field].append(std / mean)
    return {field: (sum(vals) / len(vals) if vals else 0.5) for field, vals in cv_lists.items()}


def _defensive_involvement(rows):
    """Rough per-game defensive-activity score from real tackle/sack/INT stats
    -- used only to distinguish an actual contributor from a healthy scratch
    or special-teamer who happens to share a position group."""
    total, total_w = 0.0, 0.0
    for row, w in rows:
        try:
            score = (
                float(row.get("def_tackles_solo") or 0)
                + 0.5 * float(row.get("def_tackles_with_assist") or 0)
                + 2 * float(row.get("def_sacks") or 0)
                + 3 * float(row.get("def_interceptions") or 0)
            )
        except (TypeError, ValueError):
            continue
        total += score * w
        total_w += w
    return total / total_w if total_w else 0.0


def _defensive_injury_bonus_by_team(player_index, injuries, current_rosters):
    """{team_full_name: multiplier} boosting offensive projections against a
    team currently missing an actual defensive contributor THIS week (Out/
    Doubtful only -- Injured Reserve absences are typically long-standing and
    already reflected in that defense's recent allowed-yardage average, so
    counting them again would double-count the same signal). Players with
    negligible defensive stats (healthy scratches, special-teamers) are
    excluded so a routine inactive doesn't trigger a bonus."""
    counts = {}
    for rows in player_index.values():
        if not rows:
            continue
        row = rows[0][0]
        if row.get("position_group") not in ("DB", "DL", "LB"):
            continue
        if _defensive_involvement(rows) < MIN_DEFENSIVE_INVOLVEMENT:
            continue
        name = row.get("player_display_name")
        team_full = current_rosters.get(name) or nflverse_client.TEAM_ABBR_TO_NAME.get(row.get("team"))
        if not team_full:
            continue
        injury = injuries.get(name)
        if injury and injury["status"] in DEFENSIVE_INJURY_NEW_ABSENCE_STATUSES:
            counts[team_full] = counts.get(team_full, 0) + 1

    return {team: 1 + min(MAX_DEFENSIVE_INJURY_BONUS, n * PER_MISSING_DEFENDER_BONUS) for team, n in counts.items()}


def _collect_market(event_odds):
    """Two views of the same prop board:

    best_prices: {(market, player, point, side): best price across books}
      -- the price a leg is actually offered at (shop every book).
    quotes: {(market, player): [(point, devigged_p_over), ...]}
      -- one entry per book that posts BOTH sides at a point, devigged
      within that single book. This is what the market consensus is built
      from. Devigging the best Over from one book against the best Under
      from another (what an earlier version did) mixes two books' opinions
      and understates the vig, so it's never done here."""
    best = {}
    quotes = {}
    for bm in event_odds.get("bookmakers", []):
        for market in bm.get("markets", []):
            mkey = market.get("key")
            if mkey not in MARKET_CONFIG:
                continue
            per_point = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                point = outcome.get("point")
                side = outcome.get("name")
                if not player or point is None or side not in ("Over", "Under"):
                    continue
                dec = odds_math.american_to_decimal(outcome["price"])
                key = (mkey, player, point, side)
                if key not in best or dec > best[key]["decimal"]:
                    best[key] = {"american": outcome["price"], "decimal": dec, "bookmaker": bm.get("title")}
                per_point.setdefault((player, point), {})[side] = outcome["price"]
            for (player, point), sides in per_point.items():
                if "Over" in sides and "Under" in sides:
                    p_over, _p_under = odds_math.devig_two_way(
                        odds_math.american_to_implied_prob(sides["Over"]),
                        odds_math.american_to_implied_prob(sides["Under"]),
                    )
                    quotes.setdefault((mkey, player), []).append((point, p_over))
    return best, quotes


def _market_center(quotes, sigma, log_space):
    """Consensus center of the market's distribution for one player-stat,
    solved from every book's devigged quote. A book quoting P(Over point) =
    p is saying the center sits at point + sigma * z(p) (in log space for
    the receiving fields); the median across all quotes is the consensus.
    Quotes right at 50% pin the center exactly; off-center quotes are
    translated using the player's own historical dispersion."""
    centers = []
    for point, p_over in quotes:
        p_over = min(max(p_over, 0.02), 0.98)
        z = _NORMAL.inv_cdf(p_over)
        if log_space:
            centers.append(math.log(point + 1) + sigma * z)
        else:
            centers.append(point + sigma * z)
    return median(centers) if centers else None


def _prob_under(point, center, sigma, log_space):
    x = math.log(point + 1) if log_space else point
    return ratings.normal_cdf((x - center) / sigma)


def _fetch_event_odds_and_weather(event, markets):
    """One event's props + weather -- independent of every other event, so
    get_player_prop_candidates runs this concurrently across all events
    rather than one at a time (each call is a network round trip even on a
    cache hit)."""
    try:
        event_odds = odds_client.get_event_odds(
            event["id"], markets,
            home_team=event.get("home_team"), away_team=event.get("away_team"),
        )
    except odds_client.OddsApiError:
        return None
    home = event_odds.get("home_team")
    commence = event_odds.get("commence_time")
    weather = weather_client.get_game_weather(home, commence) if home else None
    odds_age = odds_client.get_event_odds_age(event["id"], markets)
    return event_odds, weather, odds_age


def get_player_prop_candidates(markets=DEFAULT_MARKETS, max_events=None, event_filter=None):
    """`event_filter`, when given, is a set of (home_team, away_team) tuples
    -- only those games get a per-event props call. Each event costs one
    live API request (The Odds API's per-event endpoint isn't bulk like its
    game-odds endpoint), and get_events() can return several weeks' worth of
    upcoming games uncapped, so fetching props for every one of them by
    default can cost 30+ requests per refresh. Narrowing to specific games
    the user actually cares about is the main lever for controlling that."""
    season, week, _season_type = espn_client.get_current_season_and_week()

    # These 5 lookups are all independent of each other (only depend on
    # `season`, already known) -- running them concurrently rather than one
    # at a time was the other big chunk of a slow page load, alongside the
    # per-event props loop below.
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_player_index = pool.submit(nflverse_client.build_player_index, season)
        f_rosters = pool.submit(roster_client.get_current_rosters)
        f_allowed = pool.submit(nflverse_client.compute_allowed_yardage, season)
        f_injuries = pool.submit(injury_client.get_injury_lookup)
        f_qb_depth = pool.submit(roster_client.get_qb_depth_charts)
        player_index = f_player_index.result()
        current_rosters = f_rosters.result()
        allowed = f_allowed.result()
        injuries = f_injuries.result()
        qb_depth_charts = f_qb_depth.result()

    reliability = _compute_category_reliability(player_index)
    defensive_bonus = _defensive_injury_bonus_by_team(player_index, injuries, current_rosters)
    qb_out_teams = {
        team for team, qb_list in qb_depth_charts.items()
        if qb_list and injuries.get(qb_list[0], {}).get("status") in injury_client.EXCLUDE_STATUSES
    }
    league_avg = {}
    if allowed:
        league_avg["pass"] = sum(v["pass"] for v in allowed.values()) / len(allowed)
        league_avg["rush"] = sum(v["rush"] for v in allowed.values()) / len(allowed)

    events = odds_client.get_events()
    if event_filter:
        events = [e for e in events if (e.get("home_team"), e.get("away_team")) in event_filter]
    if max_events:
        events = events[:max_events]

    # Each event's props + weather are fetched independently of every other
    # event -- with 15-30 games in a normal week, doing this one at a time
    # was the single biggest contributor to a slow page load (each fetch is
    # a network round trip, even when it's just a cache lookup). None of
    # this touches shared state, so it's safe to run concurrently.
    fetched = list(ThreadPoolExecutor(max_workers=20).map(
        lambda e: _fetch_event_odds_and_weather(e, markets), events,
    ))

    candidates = []
    for event, fetched_result in zip(events, fetched):
        if fetched_result is None:
            continue
        event_odds, weather, odds_age = fetched_result

        home = event_odds.get("home_team")
        away = event_odds.get("away_team")
        matchup = f"{away} @ {home}"
        commence = event_odds.get("commence_time")

        best_prices, quotes = _collect_market(event_odds)
        offers = {}
        for (mkey, player, point, side), price in best_prices.items():
            offers.setdefault((mkey, player), {})[(point, side)] = price

        for (mkey, player), sides_by_point in offers.items():
            field, label, category = MARKET_CONFIG[mkey]
            raw_rows = player_index.get(player)
            if not raw_rows:
                continue
            player_quotes = quotes.get((mkey, player))
            if not player_quotes:
                continue  # no book posts both sides -> no way to know the market's real opinion

            injury = injuries.get(player)
            if injury and injury["status"] in injury_client.EXCLUDE_STATUSES:
                # Out / Doubtful / Injured Reserve: don't recommend a bet on a
                # player who likely won't play at all.
                continue

            rows = _apply_recency(raw_rows)
            log_space = field in RECEIVING_FIELDS
            if log_space:
                mean, std = _weighted_mean_std_log(rows, field)
                if mean is None or not std:
                    continue
                sigma = std
            else:
                mean, std = _weighted_mean_std(rows, field)
                if mean is None:
                    continue
                sigma = max(std, mean * 0.2, 1.0)  # floor: avoid overconfident small-sample variance

            # Prefer the current-roster team over the stat row's own 'team'
            # field: that field only reflects games actually played, so a
            # player traded/signed this offseason who hasn't played yet would
            # otherwise still show their old team (seen in practice).
            player_team_full = current_rosters.get(player) or nflverse_client.TEAM_ABBR_TO_NAME.get(
                raw_rows[0][0].get("team")
            )
            if player_team_full not in (home, away):
                # Extra safety net for the rare case a prop feed genuinely
                # lists a player under the wrong event.
                continue
            opponent_full = away if player_team_full == home else home

            # --- history projection: every adjustment is a multiplicative
            # factor on the projected stat, collected into one total_factor
            # (applied as an additive log shift for the log-space fields).
            total_factor = 1.0
            if injury and injury["status"] == "Questionable":
                total_factor *= injury_client.QUESTIONABLE_DISCOUNT

            return_risk = _mentions_major_injury_return(injury)
            if return_risk:
                total_factor *= MAJOR_INJURY_RETURN_DISCOUNT[category]

            matchup_factor = 1.0
            if field in YARDAGE_FIELDS and opponent_full in allowed and league_avg.get(category):
                raw_ratio = allowed[opponent_full][category] / league_avg[category]
                matchup_factor = 1 + MATCHUP_SHRINK * (raw_ratio - 1)
                matchup_factor = max(MATCHUP_ADJUSTMENT_CLAMP[0], min(MATCHUP_ADJUSTMENT_CLAMP[1], matchup_factor))
            total_factor *= matchup_factor

            if log_space:
                total_factor *= _role_trend_factor(rows)
                if player_team_full in qb_out_teams:
                    total_factor *= QB_OUT_RECEIVING_DISCOUNT

            if opponent_full and defensive_bonus.get(opponent_full):
                total_factor *= defensive_bonus[opponent_full]

            if weather:
                total_factor *= weather_client.adjustment_factor(weather, category)

            total_factor = max(total_factor, 0.01)
            model_center = mean + math.log(total_factor) if log_space else mean * total_factor

            # --- market consensus, then the small capped tilt toward the model.
            market_center = _market_center(player_quotes, sigma, log_space)
            if market_center is None:
                continue
            if SIGMA_SCALES_WITH_MARKET and not log_space and mean > 0 and market_center > mean:
                sigma *= market_center / mean
                market_center = _market_center(player_quotes, sigma, log_space)
            gap = model_center - market_center
            tilt_weight = MODEL_WEIGHT_VS_MARKET / (1 + abs(gap) / sigma)
            tilt = tilt_weight * gap
            tilt = max(-MAX_MODEL_TILT_SIGMAS * sigma, min(MAX_MODEL_TILT_SIGMAS * sigma, tilt))
            blended_center = market_center + tilt

            injury_note, injury_flag = None, None
            if injury and injury["status"] == "Questionable":
                injury_flag = "warning"
                comment = _simplify_comment(injury["comment"])
                injury_note = f"Questionable: {comment}" if comment else "Questionable"
            elif injury and injury["status"] == "Active" and injury["comment"] and not return_risk:
                # Not a game-status concern, but the comment can still carry real
                # context that our stats-only model has no way to weigh -- surface
                # it so the user can factor it in themselves. (When return_risk is
                # true, the more specific return_risk_note below covers this same
                # comment with an actual number attached, so skip the duplicate.)
                injury_flag = "info"
                injury_note = _simplify_comment(injury["comment"])

            no_current_season_data = not any(w >= 1.0 for _row, w in rows)
            # Only worth flagging once it's a real outlier -- in week 1, every
            # player has zero current-season games by definition, so the note
            # would be true for the whole pool and tell you nothing.
            stale_data_note = f"No {season} games yet" if no_current_season_data and week and week > 1 else None

            return_risk_note = (
                f"Coming back from injury: {round((1 - MAJOR_INJURY_RETURN_DISCOUNT[category]) * 100)}% "
                f"{label.lower()} discount ("
                f"{_simplify_comment(_extract_keyword_sentence(injury.get('long_comment') or injury['comment'], MAJOR_INJURY_RETURN_KEYWORDS))})"
                if return_risk else None
            )

            weather_note = None
            if weather and weather["condition"] not in ("clear", "dome"):
                pct = round((weather_client.adjustment_factor(weather, category) - 1) * 100)
                detail = f"{weather['precip_in']:.2f}\" precip" if weather["condition"] == "precipitation" else f"{weather['wind_mph']:.0f} mph wind"
                label_map = {"windy": "Windy", "heavy_wind": "Very windy", "precipitation": "Rain/snow"}
                weather_note = f"{label_map.get(weather['condition'], 'Weather')} ({detail}): {pct:+d}% to this projection."

            qb_out_note = (
                f"{player_team_full}'s starting QB is out — receiving projection discounted "
                f"{round((1 - QB_OUT_RECEIVING_DISCOUNT) * 100)}% for a backup under center."
                if log_space and player_team_full in qb_out_teams else None
            )

            # Shown on the leg so the disagreement is visible, never hidden:
            # what the player's game log alone would project vs. what the
            # market consensus says, both as the stat itself.
            model_median = round(math.exp(model_center) - 1, 1) if log_space else round(model_center, 1)
            market_median = round(math.exp(market_center) - 1, 1) if log_space else round(market_center, 1)

            common = dict(
                player=player, team=player_team_full, stat_category=mkey, stat_label=label,
                stat_field=field,
                injury_note=injury_note, injury_flag=injury_flag,
                stale_data_note=stale_data_note, weather_note=weather_note, qb_out_note=qb_out_note,
                return_risk_note=return_risk_note,
                category_cv=reliability.get(field, 0.5),
                odds_age_seconds=odds_age,
                model_median=model_median, market_median=market_median,
            )
            market_label = f"Player Prop: {label}"

            for (point, side), price in sides_by_point.items():
                p_under_model = _prob_under(point, blended_center, sigma, log_space)
                p_under_market = _prob_under(point, market_center, sigma, log_space)
                p_under_history = _prob_under(point, model_center, sigma, log_space)
                if side == "Under":
                    p_model, p_market, p_history = p_under_model, p_under_market, p_under_history
                else:
                    p_model, p_market, p_history = 1 - p_under_model, 1 - p_under_market, 1 - p_under_history
                selection = f"{player} {side} {point} {label}"
                leg = odds_math.make_leg(matchup, commence, market_label, selection, price, p_model, p_market)
                leg.update(common, side=side, line=point, history_prob=round(p_history, 4))
                candidates.append(leg)

    return candidates
