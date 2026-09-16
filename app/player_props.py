"""Player prop candidates: live sportsbook prop lines (via The Odds API's
per-event endpoint) matched against each player's own recent-game stat
distribution (free, from nflverse), adjusted for:

- the strength of the opponent's run/pass defense (allowed yardage vs. league average)
- recent role trend (target share climbing/falling vs. their own baseline)
- recency (a player's last few games count more than early-season/last-year ones)
- weather at the stadium (wind/precipitation hurt the passing game; free, no-key)
- the opponent defense's own fresh injuries (missing starters this week, not
  already reflected in their season-to-date allowed-yardage numbers)
- how statistically predictable this stat category tends to be league-wide
  (a lower-variance category is a more reliable bet at the same edge)
- coming back from a significant injury (ACL, Achilles, hamstring, etc.) --
  even once a player is fully cleared ("Active"), a coach often manages their
  workload down for the first several games back, which a season-average
  stat line won't reflect. We can't measure "how limited," so we apply a
  flat, conservative discount rather than pretend precision we don't have.
- the player's OWN team's starting QB being out -- a backup center-fielding
  the passing game tends to drag down every pass-catcher's numbers with him,
  not just the QB's own stats (which are already handled by excluding him
  from the pool entirely if he's the one who's hurt).

This is deliberately simpler than the team-game model: we estimate a
player's expected stat as roughly-normal around their adjusted recent
average, and compare the resulting probability to the sportsbook's devigged
price. Small-sample players (fewer than MIN_GAMES_WEIGHT effective games)
are skipped rather than guessed at.
"""
import re
from app import odds_client, nflverse_client, ratings, odds_math, espn_client, injury_client, weather_client, roster_client

# ESPN's comments are written like news blurbs ("X said Wednesday that...,
# Reporter Name of Outlet reports.") -- strip the trailing attribution clause
# and hard-cap length so notes stay skimmable instead of reading like an article.
_TRAILING_ATTRIBUTION_RE = re.compile(r",\s*[^,]+ reports?\.?\s*$", re.IGNORECASE)
COMMENT_MAX_LEN = 140


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
MATCHUP_ADJUSTMENT_CLAMP = (0.75, 1.25)
ROLE_TREND_CLAMP = (0.8, 1.3)
RECENT_GAMES_FOR_TREND = 3
RECENCY_DECAY_PER_WEEK = 0.90  # within the current season, each week further back counts ~10% less

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


def _consolidate_best_prices(event_odds):
    """Collapse all bookmakers down to the single best (highest decimal) price
    per (market, player, point, side), so shopping across books doesn't
    produce duplicate near-identical legs."""
    best = {}
    for bm in event_odds.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") not in MARKET_CONFIG:
                continue
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                point = outcome.get("point")
                side = outcome.get("name")
                if not player or point is None or side not in ("Over", "Under"):
                    continue
                dec = odds_math.american_to_decimal(outcome["price"])
                key = (market["key"], player, point, side)
                if key not in best or dec > best[key]["decimal"]:
                    best[key] = {"american": outcome["price"], "decimal": dec, "bookmaker": bm.get("title")}
    return best


def get_player_prop_candidates(markets=DEFAULT_MARKETS, max_events=None):
    season, week, _season_type = espn_client.get_current_season_and_week()
    player_index = nflverse_client.build_player_index(season)
    current_rosters = roster_client.get_current_rosters()
    allowed = nflverse_client.compute_allowed_yardage(season)
    injuries = injury_client.get_injury_lookup()
    reliability = _compute_category_reliability(player_index)
    defensive_bonus = _defensive_injury_bonus_by_team(player_index, injuries, current_rosters)
    qb_out_teams = {
        team for team, qb_list in roster_client.get_qb_depth_charts().items()
        if qb_list and injuries.get(qb_list[0], {}).get("status") in injury_client.EXCLUDE_STATUSES
    }
    league_avg = {}
    if allowed:
        league_avg["pass"] = sum(v["pass"] for v in allowed.values()) / len(allowed)
        league_avg["rush"] = sum(v["rush"] for v in allowed.values()) / len(allowed)

    events = odds_client.get_events()
    if max_events:
        events = events[:max_events]

    candidates = []
    for event in events:
        try:
            event_odds = odds_client.get_event_odds(
                event["id"], markets,
                home_team=event.get("home_team"), away_team=event.get("away_team"),
            )
        except odds_client.OddsApiError:
            continue

        home = event_odds.get("home_team")
        away = event_odds.get("away_team")
        matchup = f"{away} @ {home}"
        commence = event_odds.get("commence_time")
        weather = weather_client.get_game_weather(home, commence)

        best_prices = _consolidate_best_prices(event_odds)
        pairs = {}
        for (mkey, player, point, side), price in best_prices.items():
            pairs.setdefault((mkey, player, point), {})[side] = price

        for (mkey, player, point), sides in pairs.items():
            field, label, category = MARKET_CONFIG[mkey]
            raw_rows = player_index.get(player)
            if not raw_rows:
                continue

            injury = injuries.get(player)
            if injury and injury["status"] in injury_client.EXCLUDE_STATUSES:
                # Out / Doubtful / Injured Reserve: don't recommend a bet on a
                # player who likely won't play at all.
                continue

            rows = _apply_recency(raw_rows)
            mean, std = _weighted_mean_std(rows, field)
            if mean is None:
                continue
            std = max(std, mean * 0.2, 1.0)  # floor: avoid overconfident small-sample variance

            if injury and injury["status"] == "Questionable":
                mean *= injury_client.QUESTIONABLE_DISCOUNT

            return_risk = _mentions_major_injury_return(injury)
            if return_risk:
                mean *= MAJOR_INJURY_RETURN_DISCOUNT[category]

            # Prefer the current-roster team over the stat row's own 'team'
            # field: that field only reflects games actually played, so a
            # player traded/signed this offseason who hasn't played yet would
            # otherwise still show their old team (seen in practice).
            player_team_full = current_rosters.get(player) or nflverse_client.TEAM_ABBR_TO_NAME.get(
                raw_rows[0][0].get("team")
            )
            if player_team_full not in (home, away):
                # Extra safety net for the rare case a prop feed genuinely
                # lists a player under the wrong event. Note this used to
                # trigger on legitimate trades too (e.g. a receiver who moved
                # teams) back when it only checked nflverse's stale
                # last-played-team field -- current_rosters above fixes that
                # for the common case, so this should now only catch real
                # feed errors.
                continue
            opponent_full = away if player_team_full == home else home

            adjusted_mean = mean
            if opponent_full and opponent_full in allowed and league_avg.get(category):
                factor = allowed[opponent_full][category] / league_avg[category]
                factor = max(MATCHUP_ADJUSTMENT_CLAMP[0], min(MATCHUP_ADJUSTMENT_CLAMP[1], factor))
                adjusted_mean *= factor

            if field in RECEIVING_FIELDS:
                adjusted_mean *= _role_trend_factor(rows)
                if player_team_full in qb_out_teams:
                    adjusted_mean *= QB_OUT_RECEIVING_DISCOUNT

            if opponent_full and defensive_bonus.get(opponent_full):
                adjusted_mean *= defensive_bonus[opponent_full]

            if weather:
                adjusted_mean *= weather_client.adjustment_factor(weather, category)

            p_under = ratings.normal_cdf((point - adjusted_mean) / std)
            p_over = 1 - p_under

            over_price, under_price = sides.get("Over"), sides.get("Under")
            book_over_p = odds_math.american_to_implied_prob(over_price["american"]) if over_price else None
            book_under_p = odds_math.american_to_implied_prob(under_price["american"]) if under_price else None
            if book_over_p is not None and book_under_p is not None:
                book_over_p, book_under_p = odds_math.devig_two_way(book_over_p, book_under_p)

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
            stale_data_note = "No 2026 games yet" if no_current_season_data and week and week > 1 else None

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
                if field in RECEIVING_FIELDS and player_team_full in qb_out_teams else None
            )

            common = dict(
                player=player, stat_category=mkey, stat_label=label,
                line=point, injury_note=injury_note, injury_flag=injury_flag,
                stale_data_note=stale_data_note, weather_note=weather_note, qb_out_note=qb_out_note,
                return_risk_note=return_risk_note,
                category_cv=reliability.get(field, 0.5),
            )

            market_label = f"Player Prop: {label}"
            if over_price and book_over_p is not None:
                selection = f"{player} Over {point} {label}"
                leg = odds_math.make_leg(matchup, commence, market_label, selection, over_price, p_over, book_over_p)
                leg.update(common, side="Over")
                candidates.append(leg)
            if under_price and book_under_p is not None:
                selection = f"{player} Under {point} {label}"
                leg = odds_math.make_leg(matchup, commence, market_label, selection, under_price, p_under, book_under_p)
                leg.update(common, side="Under")
                candidates.append(leg)

    return candidates
