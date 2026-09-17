"""Combine live sportsbook odds with our power-rating model to surface bets
where the model's win probability is meaningfully higher than what the
book's price implies (an "edge"), plus a broader candidate pool for parlay
construction.
"""
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist, median
from app.config import SHARP_BOOK_KEY, SHARP_BOOK_WEIGHT
from app import espn_client, ratings, odds_math, nflverse_client, injury_client, roster_client, odds_client
from app.team_names import build_lookup, match

MIN_EDGE_FOR_VALUE_BET = 0.02   # 2 percentage points of model probability over the breakeven price (i.e. +EV after vig)
# Legs below this probability are kept out of the parlay pool. Was 0.50
# ("no coinflip-or-worse legs"), which quietly worked against the target:
# every leg carries the book's vig, so the way to reach a 200x payout with
# the HIGHEST hit chance is the fewest legs, i.e. plus-money legs -- an
# underdog at +150 with an honest 42% is a better building block than two
# -110 legs at 52% each. The floor now only keeps out deep longshots, where
# the model tilt and correlation estimates are least trustworthy.
MIN_PROB_FOR_PARLAY_LEG = 0.25
# How far a Moneyline probability moves from the devigged market price
# toward the power-rating model. Backtested walk-forward over 2,118 games
# with a real closing moneyline (2018-2025, scripts/backtest_moneyline.py):
# the log-loss-minimizing weight was 0.00, every step toward the model
# made predictions worse, and in the games where the model disagreed with
# the market by 15+ points the market was right (model avg 60%, market 39%,
# actual 35%). Flat-betting every side the model liked by 5+ points lost
# 7.6% at closing prices. Kept as a constant (rather than deleting the
# model) so the machinery stays testable if the rating model ever improves
# -- but the evidence says leave it at 0.
MODEL_WEIGHT_VS_MARKET_MONEYLINE = 0.0
EPA_BLEND_WEIGHT = 0.55  # how much of the final rating comes from EPA vs. raw scoring margin

# Rough, widely-cited handicapping heuristic: losing the starting QB costs a
# team several points against the spread. Not precise -- backup QB quality
# varies a lot -- but far better than ignoring a season-ending injury outright.
STARTING_QB_OUT_PENALTY = 6.0
STARTING_QB_QUESTIONABLE_PENALTY = 2.0
# Additional penalty (on top of the above) when the backup is ALSO out --
# losing your top two QBs is much worse than losing just the starter, since
# whoever's left is often a rarely-used emergency arm.
BACKUP_QB_ALSO_OUT_PENALTY = 4.0


def _gather_weighted_games():
    season, week, season_type = espn_client.get_current_season_and_week()
    current_games = espn_client.get_season_games_to_date(season)
    weighted = [(g["home"], g["away"], g["home_score"], g["away_score"], 1.0) for g in current_games]

    # Early in the season there isn't enough current-year data for stable ratings,
    # so blend in last season at reduced weight.
    if len(current_games) < 48:
        prev_games = espn_client.get_full_season_games(season - 1)
        weighted += [(g["home"], g["away"], g["home_score"], g["away_score"], 0.5) for g in prev_games]

    return weighted, season, week


def _blend_with_epa(power, season):
    """Blend the scoring-margin power ratings with nflverse's free per-team EPA
    data, when available. EPA is denominated in expected points, so it lives on
    roughly the same scale as the point-based power ratings without needing an
    arbitrary conversion factor. Falls back to pure scoring-margin ratings for
    any team missing EPA data (or if the nflverse fetch fails outright)."""
    try:
        net_epa = nflverse_client.compute_net_epa_ratings(season)
    except Exception:
        net_epa = {}

    if not net_epa:
        return power, False

    blended = {}
    for team, rating in power.items():
        if team in net_epa:
            blended[team] = (1 - EPA_BLEND_WEIGHT) * rating + EPA_BLEND_WEIGHT * net_epa[team]
        else:
            blended[team] = rating

    mean_r = sum(blended.values()) / len(blended)
    return {t: v - mean_r for t, v in blended.items()}, True


def _apply_starting_qb_injuries(power):
    """Docks a team's rating when its ACTUAL starting QB (ESPN's real depth
    chart, not a historical-usage proxy) is out/doubtful/questionable, per
    ESPN's live injury report. Returns (adjusted_power, notes) where notes is
    [{"team", "player", "status"}] for display.

    Depth-chart QB1 is used rather than "most recent pass attempts" because
    that proxy gets it wrong whenever the starter changes for any reason
    other than a trade -- seen in practice: a team's real, injured starter
    was missed while a healthy 3rd-string emergency QB got wrongly flagged
    instead, because he'd started more games historically."""
    try:
        starters = roster_client.get_qb_depth_charts()
        injuries = injury_client.get_injury_lookup()
    except Exception:
        return power, []

    adjusted = dict(power)
    notes = []
    for team_full, qb_list in starters.items():
        if team_full not in adjusted or not qb_list:
            continue
        qb_name = qb_list[0]
        injury = injuries.get(qb_name)
        if not injury:
            continue
        status = injury["status"]
        if status in injury_client.EXCLUDE_STATUSES:
            adjusted[team_full] -= STARTING_QB_OUT_PENALTY
            notes.append({"team": team_full, "player": qb_name, "status": status})

            if len(qb_list) > 1:
                backup_name = qb_list[1]
                backup_injury = injuries.get(backup_name)
                if backup_injury and backup_injury["status"] in injury_client.EXCLUDE_STATUSES:
                    adjusted[team_full] -= BACKUP_QB_ALSO_OUT_PENALTY
                    notes.append({
                        "team": team_full, "player": backup_name,
                        "status": backup_injury["status"], "is_backup": True,
                    })
        elif status == "Questionable":
            adjusted[team_full] -= STARTING_QB_QUESTIONABLE_PENALTY
            notes.append({"team": team_full, "player": qb_name, "status": status})

    return adjusted, notes


def _build_model():
    weighted_games, season, week = _gather_weighted_games()
    if not weighted_games:
        return None
    power = ratings.compute_power_ratings(weighted_games)
    power, epa_used = _blend_with_epa(power, season)
    power, qb_injury_notes = _apply_starting_qb_injuries(power)
    return {
        "power": power,
        "season": season,
        "week": week,
        "epa_used": epa_used,
        "qb_injury_notes": qb_injury_notes,
    }


def _offer_books(bookmakers):
    """Just the configured sportsbook's entry (see config.BOOKMAKER_KEY) --
    the only place a leg's line and price may come from."""
    return [bm for bm in bookmakers if odds_math.is_offer_book(bm)]


def _best_price(bookmakers, market_key, outcome_name, point=None):
    """The offer book's price for a given outcome (the best across its
    entries, which is normally just one)."""
    best = None
    for bm in _offer_books(bookmakers):
        for market in bm.get("markets", []):
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") != outcome_name:
                    continue
                if point is not None and outcome.get("point") != point:
                    continue
                dec = odds_math.american_to_decimal(outcome["price"])
                if best is None or dec > best["decimal"]:
                    best = {
                        "american": outcome["price"],
                        "decimal": dec,
                        "point": outcome.get("point"),
                        "bookmaker": bm.get("title"),
                    }
    return best


_NORMAL = NormalDist()


def _sharp_fair(bookmakers, market_key, name_a, name_b, point_a, threshold_sign, sigma):
    """The sharp book's devigged P(side a), translated to the offer book's
    point when the two books post different numbers. A side wins when the
    underlying margin/total X exceeds a threshold t (t = -point for a
    spread side, t = +point for a total's Over, -point for its Under);
    the sharp book's P(X > t_sharp) becomes P(X > t_offer) via the normal
    margin model: Phi(z + (t_sharp - t_offer) / sigma). Returns None when
    the sharp book has no price for this market."""
    for bm in bookmakers:
        if (bm.get("key") or "").lower() != SHARP_BOOK_KEY:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != market_key:
                continue
            outs = {o.get("name"): o for o in market.get("outcomes", [])}
            if name_a not in outs or name_b not in outs:
                continue
            fair_a, _fair_b = odds_math.devig_two_way(
                odds_math.american_to_implied_prob(outs[name_a]["price"]),
                odds_math.american_to_implied_prob(outs[name_b]["price"]),
            )
            sharp_point = outs[name_a].get("point")
            if threshold_sign and point_a is not None and sharp_point is not None and sharp_point != point_a:
                t_sharp, t_offer = threshold_sign * sharp_point, threshold_sign * point_a
                z = _NORMAL.inv_cdf(min(max(fair_a, 0.01), 0.99))
                fair_a = _NORMAL.cdf(z + (t_sharp - t_offer) / sigma)
            return fair_a
    return None


def _consensus_two_way(bookmakers, market_key, name_a, name_b, point_a=None, point_b=None,
                       threshold_sign=None, sigma=ratings.MARGIN_SIGMA):
    """Fair (fair_a, fair_b) for a two-way market at the offer book's
    point: each book's own pair of prices is devigged WITHIN that book, the
    median across books is the consensus -- and when the sharp book
    (config.SHARP_BOOK_KEY) has a price, the answer is SHARP_BOOK_WEIGHT
    on its number (translated to this point, see _sharp_fair) and the rest
    on that consensus. Devigging the best price for side A (from one book)
    against the best price for side B (from another) -- what an earlier
    version did -- mixes two books' opinions and understates the vig, so
    both sides come out a little too likely. Books that only post one
    side, or a different point, don't enter the consensus."""
    sharp = _sharp_fair(bookmakers, market_key, name_a, name_b, point_a, threshold_sign, sigma)
    fair_as = []
    for bm in bookmakers:
        if (bm.get("key") or "").lower() == SHARP_BOOK_KEY:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != market_key:
                continue
            prices = {}
            for outcome in market.get("outcomes", []):
                name, point = outcome.get("name"), outcome.get("point")
                if name == name_a and (point_a is None or point == point_a):
                    prices["a"] = outcome["price"]
                elif name == name_b and (point_b is None or point == point_b):
                    prices["b"] = outcome["price"]
            if "a" in prices and "b" in prices:
                fair_a, _fair_b = odds_math.devig_two_way(
                    odds_math.american_to_implied_prob(prices["a"]),
                    odds_math.american_to_implied_prob(prices["b"]),
                )
                fair_as.append(fair_a)
    if not fair_as and sharp is None:
        return None, None
    if sharp is None:
        fair_a = median(fair_as)
    elif not fair_as:
        fair_a = sharp
    else:
        fair_a = SHARP_BOOK_WEIGHT * sharp + (1 - SHARP_BOOK_WEIGHT) * median(fair_as)
    return fair_a, 1 - fair_a


def _first_spread_points(bookmakers, home_name, away_name):
    """Returns (home_point, away_point) as posted by the offer book."""
    for bm in _offer_books(bookmakers):
        for market in bm.get("markets", []):
            if market.get("key") != "spreads":
                continue
            outcomes = market.get("outcomes", [])
            home_out = next((o for o in outcomes if o["name"] == home_name), None)
            away_out = next((o for o in outcomes if o["name"] == away_name), None)
            if home_out and away_out and home_out.get("point") is not None and away_out.get("point") is not None:
                return home_out["point"], away_out["point"]
    return None


def _first_total_line(bookmakers):
    """The offer book's total line."""
    for bm in _offer_books(bookmakers):
        for market in bm.get("markets", []):
            if market.get("key") != "totals":
                continue
            outcomes = market.get("outcomes", [])
            over_out = next((o for o in outcomes if o["name"] == "Over"), None)
            if over_out and over_out.get("point") is not None:
                return over_out["point"]
    return None


def analyze_games(odds_data, model):
    """Returns a list of candidate legs across all games/markets with model probabilities."""
    lookup = build_lookup(model["power"].keys())
    candidates = []

    for game in odds_data:
        home_name = game.get("home_team")
        away_name = game.get("away_team")
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        home_key = match(home_name, lookup)
        away_key = match(away_name, lookup)
        if not home_key or not away_key:
            continue

        home_rating = model["power"][home_key]
        away_rating = model["power"][away_key]
        home_win_prob = ratings.win_probability(home_rating, away_rating)
        away_win_prob = 1 - home_win_prob

        matchup = f"{away_name} @ {home_name}"
        commence = game.get("commence_time")

        # --- Moneyline (h2h) ---
        home_price = _best_price(bookmakers, "h2h", home_name)
        away_price = _best_price(bookmakers, "h2h", away_name)
        fair_home_p, fair_away_p = _consensus_two_way(bookmakers, "h2h", home_name, away_name)
        if home_price and away_price and fair_home_p is not None:

            # Market-anchored, like every other market here: the devigged
            # closing-style price is the estimate, and the power-rating model
            # only tilts it by MODEL_WEIGHT_VS_MARKET_MONEYLINE (currently 0
            # -- see that constant for the backtest that put it there). An
            # earlier version used the model's own probability outright
            # inside its calibrated range and deferred to the market only
            # above WIN_PROB_CEILING; that still let the model claim 15-20
            # point edges on underdogs that the backtest shows were pure
            # noise. The ceiling deferral is kept as a second guard: the
            # model structurally can't express more confidence than ~75%,
            # so above it the "disagreement" is mechanical, not real.
            if fair_home_p > ratings.WIN_PROB_CEILING or fair_away_p > ratings.WIN_PROB_CEILING:
                home_win_prob, away_win_prob = fair_home_p, fair_away_p
            else:
                home_win_prob = fair_home_p + MODEL_WEIGHT_VS_MARKET_MONEYLINE * (home_win_prob - fair_home_p)
                away_win_prob = 1 - home_win_prob

            candidates.append(_make_leg(
                matchup, commence, "Moneyline", home_name, home_price,
                home_win_prob, fair_home_p, team=home_name,
            ))
            candidates.append(_make_leg(
                matchup, commence, "Moneyline", away_name, away_price,
                away_win_prob, fair_away_p, team=away_name,
            ))

        # --- Spread --- (find the posted point from the first book that has one; then
        # shop all books for the best price at that point)
        # NOTE (found in a full model audit, backtested walk-forward against
        # 5,142 real (game, side) spread observations, 2016-2025 closing
        # lines): cover_probability showed NO demonstrated skill over the
        # real market -- every edge bucket landed at ~48-52% actual cover
        # rate regardless of claimed edge size (a flat line, not the
        # monotonic separation a real signal would show), consistent with
        # pure vig loss. Same treatment as Totals: price Spread legs at the
        # devigged MARKET probability (edge = 0) rather than manufacture a
        # false edge from a signal proven not to beat the market. This also
        # fixes a smaller inconsistency where Spread previously compared
        # against the raw (vig-inflated) price instead of the devigged fair
        # price like Moneyline/Totals do.
        spread_point = _first_spread_points(bookmakers, home_name, away_name)
        if spread_point:
            home_point, away_point = spread_point
            home_price_s = _best_price(bookmakers, "spreads", home_name, home_point)
            away_price_s = _best_price(bookmakers, "spreads", away_name, away_point)
            fair_home_p, fair_away_p = _consensus_two_way(
                bookmakers, "spreads", home_name, away_name, home_point, away_point,
                threshold_sign=-1, sigma=ratings.MARGIN_SIGMA,
            )
            if home_price_s and away_price_s and fair_home_p is not None:
                home_label = f"{home_name} {home_point:+g}"
                away_label = f"{away_name} {away_point:+g}"
                candidates.append(_make_leg(matchup, commence, "Spread", home_label, home_price_s, fair_home_p, fair_home_p, team=home_name))
                candidates.append(_make_leg(matchup, commence, "Spread", away_label, away_price_s, fair_away_p, fair_away_p, team=away_name))

        # --- Totals ---
        # NOTE (found in a full model audit, backtested walk-forward against
        # 2016-2025 real closing lines): predicted_total here comes from
        # compute_scoring_averages(), a raw, NOT opponent-adjusted average --
        # unlike the power ratings used for Moneyline/Spread. Measured
        # residual std against real outcomes was 13.69, actually WORSE than
        # just using the book's own total_line directly (13.17), and hit
        # rate by confidence bucket was flat/non-monotonic (a "72% confident"
        # bucket hit only 42% of the time). In plain terms: this signal has
        # no demonstrated edge over the market. Rather than keep shipping a
        # model_prob that can manufacture a false "value bet" out of noise,
        # Total legs are priced at the devigged MARKET probability (edge
        # ~= 0 by construction) until a real opponent-adjusted total model
        # replaces this. They still show up in the pool/parlays at their
        # honest fair price -- just never as a false "edge."
        total_line = _first_total_line(bookmakers)
        if total_line is not None:
            over_price = _best_price(bookmakers, "totals", "Over", total_line)
            under_price = _best_price(bookmakers, "totals", "Under", total_line)
            fair_over_p, fair_under_p = _consensus_two_way(
                bookmakers, "totals", "Over", "Under", total_line, total_line,
                threshold_sign=+1, sigma=ratings.TOTAL_SIGMA,
            )
            if over_price and under_price and fair_over_p is not None:
                candidates.append(_make_leg(matchup, commence, "Total", f"Over {total_line}", over_price, fair_over_p, fair_over_p, side="Over"))
                candidates.append(_make_leg(matchup, commence, "Total", f"Under {total_line}", under_price, fair_under_p, fair_under_p, side="Under"))

    # All game-level legs share one bulk odds fetch, so they share one age --
    # unlike props, which are fetched (and can go stale) one game at a time.
    odds_age = odds_client.get_odds_age()
    for leg in candidates:
        leg["odds_age_seconds"] = odds_age

    return candidates


def _make_leg(matchup, commence, market, selection, price, model_prob, fair_prob, team=None, side=None):
    """odds_math.make_leg plus the team/side fields app/correlations.py
    needs to place a game-level leg relative to the player props around it."""
    leg = odds_math.make_leg(matchup, commence, market, selection, price, model_prob, fair_prob)
    leg["team"] = team
    leg["side"] = side
    return leg


def get_value_bets_and_pool(markets="h2h,spreads,totals", include_props=True, event_filter=None):
    """Top-level entry point used by the web app.
    Returns (value_bets, parlay_pool, meta) or raises on hard failure.
    `event_filter`: see player_props.get_player_prop_candidates -- narrows
    which games get a (costly, one-call-per-event) props fetch.
    """
    # _build_model() (season games, ratings, injuries -- all cache reads) and
    # get_odds() don't depend on each other at all, but used to run one after
    # the other; fetching both concurrently overlaps their cache round trips
    # instead of paying for them back to back.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_model = pool.submit(_build_model)
        f_odds = pool.submit(odds_client.get_odds, markets=markets)
        model = f_model.result()
        odds_data = f_odds.result()

    if model is None:
        return [], [], {"error": "No historical game data available to build ratings yet."}

    candidates = analyze_games(odds_data, model)

    props_used = False
    if include_props:
        try:
            from app import player_props
            prop_candidates = player_props.get_player_prop_candidates(event_filter=event_filter)
            candidates += prop_candidates
            props_used = True
        except odds_client.OddsApiError:
            pass  # props are a bonus -- don't break the page if the props call fails/quota runs out

    value_bets = sorted(
        [c for c in candidates if c["edge"] >= MIN_EDGE_FOR_VALUE_BET],
        key=lambda c: c["edge"],
        reverse=True,
    )
    parlay_pool = sorted(
        [c for c in candidates if c["model_prob"] >= MIN_PROB_FOR_PARLAY_LEG],
        key=lambda c: c["edge"],
        reverse=True,
    )
    meta = {
        "season": model["season"],
        "week": model["week"],
        "games_analyzed": len(odds_data),
        "epa_used": model["epa_used"],
        "props_used": props_used,
        "qb_injury_notes": model["qb_injury_notes"],
    }
    return value_bets, parlay_pool, meta
