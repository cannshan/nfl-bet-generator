"""Reusable combo-search core: given a pool of candidate legs, search
combinations for a payout near a target, ranked by the highest combined
hit probability achievable at that payout level. Starts with a tight band
around the target and widens it, but only actually prefers a wider band's
combo when it's a meaningfully better bet (see PROB_IMPROVEMENT_FACTOR) --
hitting the exact target payout is never allowed to force a near-zero-
probability combo when a somewhat-lower-payout, much-more-likely one exists.

This is deliberately leg-pool-agnostic -- see game_cards.py, which calls it
per matchup with that game's own legs (mostly player props). Multiplying
probabilities together is only a strictly fair calculation when legs are
independent; for same-game legs that's a modeling simplification made
explicit to the user (see the disclaimer in index.html), not something this
module pretends is exact.
"""
import itertools
from app import odds_math, formatting

MAX_LEGS_SEARCHED = 8
CROSS_GAME_CANDIDATE_POOL_SIZE = 14
# Backtested against 3,089 real (QB, game) observations, 2018-2025,
# walk-forward: a QB beating their own projected passing yards and that
# SAME GAME's total beating its real closing line are positively
# correlated in reality (a big passing day usually means more total
# points) -- the naive independence multiplication combo_stats() otherwise
# uses understates the true joint probability by ~23-26% when both go the
# same direction (both Over or both Under), and overstates it by ~23-26%
# when they go opposite directions. This is the one same-game correlation
# actually measured and corrected here; other same-game pairs (e.g. two
# receivers on the same team) remain treated as independent -- not because
# they're assumed uncorrelated, but because they haven't been backtested.
QB_PASS_TOTAL_SAME_DIRECTION_LIFT = 1.24
QB_PASS_TOTAL_OPPOSITE_DIRECTION_LIFT = 0.76
# A wider-tolerance combo only replaces a tighter-tolerance one if it's at
# least this many times more likely to hit. Reaching an ambitious target
# (e.g. 200x on a $5 stake) can force 7-8 legs, which multiplies down to a
# ~1-2% combined probability even when each leg is individually solid --
# that combo would otherwise "win" the tightest-tolerance band every time
# despite being a much worse bet than a smaller combo paying somewhat less.
# This is a genuine trade-off, not a bug: a lower target payout will always
# get you higher win-probability combos, since parlay math doesn't allow both
# a huge payout and a high hit rate off realistic odds.
PROB_IMPROVEMENT_FACTOR = 1.5


def _qb_pass_total_correlation(combo):
    """See QB_PASS_TOTAL_*_LIFT above. Applies once per (QB passing leg,
    same-game Total leg) pair found in the combo -- there's normally at
    most one of each per game, so this rarely compounds more than once."""
    factor = 1.0
    qb_legs = [l for l in combo if l.get("stat_category") == "player_pass_yds"]
    total_legs = [l for l in combo if l["market"] == "Total"]
    for qb_leg in qb_legs:
        for total_leg in total_legs:
            if qb_leg["matchup"] != total_leg["matchup"]:
                continue
            same_direction = (qb_leg["side"] == "Over") == total_leg["selection"].startswith("Over")
            factor *= QB_PASS_TOTAL_SAME_DIRECTION_LIFT if same_direction else QB_PASS_TOTAL_OPPOSITE_DIRECTION_LIFT
    return factor


def combo_stats(combo, stake):
    dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = 1.0
    for leg in combo:
        combined_prob *= leg["model_prob"]
    combined_prob = min(combined_prob * _qb_pass_total_correlation(combo), 1.0)
    return dec_odds, payout, combined_prob


def _build_result(combo, stake):
    dec_odds, payout, combined_prob = combo_stats(combo, stake)
    return {
        "legs": list(combo),
        "decimal_odds": dec_odds,
        "american_equivalent": odds_math.decimal_to_american(dec_odds),
        "payout": round(payout, 2),
        "stake": stake,
        "combined_prob": combined_prob,
        "num_legs": len(combo),
    }


def search_near_target(legs, stake, target_payout, max_legs=MAX_LEGS_SEARCHED,
                        tolerances=(0.15, 0.3, 0.5, 0.75, 0.95)):
    """Returns matches (payout within tolerance of target) sorted by highest
    combined probability. Walks tolerance bands tightest to widest, but only
    adopts a wider band's best combo when it's meaningfully more likely to
    hit (see PROB_IMPROVEMENT_FACTOR) -- so a lower-payout combo with a real
    shot at winning beats a target-hitting one that's basically a lottery
    ticket. Combo stats are computed once against the widest band, then just
    filtered per tolerance level rather than recomputed each time."""
    widest = tolerances[-1]
    low_widest = target_payout * (1 - widest)
    high_widest = target_payout * (1 + widest)
    all_combos = []
    for size in range(2, min(max_legs, len(legs)) + 1):
        for combo in itertools.combinations(legs, size):
            dec_odds, payout, combined_prob = combo_stats(combo, stake)
            if low_widest <= payout <= high_widest:
                all_combos.append((payout, combined_prob, combo))

    best_matches, best_prob = None, -1.0
    for tol in tolerances:
        low = target_payout * (1 - tol)
        high = target_payout * (1 + tol)
        matches = [c for c in all_combos if low <= c[0] <= high]
        if not matches:
            continue
        matches.sort(key=lambda c: c[1], reverse=True)
        top_prob = matches[0][1]
        if best_matches is None or top_prob >= best_prob * PROB_IMPROVEMENT_FACTOR:
            best_matches, best_prob = matches, top_prob

    if best_matches is None:
        return []
    return [_build_result(combo, stake) for _payout, _prob, combo in best_matches]


def dedupe_best_per_bet(pool):
    """Collapse every variant of the same underlying bet -- different
    Over/Under sides, and different books posting different lines for the
    same player+stat (e.g. Over 39.5 / 44.5 / 49.5 rush yards are all really
    "will this player rush for a lot," not independent bets) -- down to
    whichever single variant has the best edge. Deliberately does NOT key on
    `line`: keeping every line a book happens to offer would let the search
    stack several near-duplicate bets on the same outcome and count them as
    independent, which is a much worse double-count than ordinary same-game
    correlation. Unlike dedupe_best_per_game, this does NOT restrict to one
    leg per game: different props/markets from the same game can both survive."""
    best = {}
    for leg in pool:
        key = (leg["matchup"], leg["market"], leg.get("player"), leg.get("stat_category"))
        if key not in best or leg["edge"] > best[key]["edge"]:
            best[key] = leg
    return list(best.values())


def _drop_redundant_favorite_bets(legs):
    """A team's own spread and their moneyline are nearly the same bet
    (covering the spread almost always means winning outright too); when
    both survive dedupe for the same team, keep only whichever has the
    better edge instead of letting the search double-count them as if
    independent. Drops whichever of the pair is weaker, not always the spread."""
    moneyline_by_matchup_team = {
        (leg["matchup"], leg["selection"]): leg for leg in legs if leg["market"] == "Moneyline"
    }
    dropped_ids = set()
    for leg in legs:
        if leg["market"] != "Spread":
            continue
        matching_ml = next(
            (ml for (matchup, team), ml in moneyline_by_matchup_team.items()
             if matchup == leg["matchup"] and leg["selection"].startswith(team)),
            None,
        )
        if matching_ml:
            loser = leg if leg["edge"] <= matching_ml["edge"] else matching_ml
            dropped_ids.add(id(loser))
    return [leg for leg in legs if id(leg) not in dropped_ids]


def find_best_odds_parlays(pool, stake, target_payout, num_results=3, pool_size=CROSS_GAME_CANDIDATE_POOL_SIZE):
    """The 'statistically best bets' parlay: no restriction on which games a
    leg can come from -- purely chases the highest achievable hit probability
    at the target payout using every signal the model has (edge, discounted
    for how historically predictable that stat category is). This CAN select
    multiple legs from the same game; when it does, the combined probability
    is an approximation for the same reason a Same Game Parlay's is (legs
    aren't actually independent) -- each result says whether that happened
    (`has_same_game_legs`) so that's never hidden."""
    legs = _drop_redundant_favorite_bets(dedupe_best_per_bet(pool))
    legs = sorted(legs, key=lambda l: l["edge"] - 0.05 * l.get("category_cv", 0.5), reverse=True)[:pool_size]
    matches = search_near_target(legs, stake, target_payout)
    results = matches[:num_results]
    for r in results:
        matchups = [leg["matchup"] for leg in r["legs"]]
        r["has_same_game_legs"] = len(set(matchups)) < len(matchups)
        ages = [leg["odds_age_seconds"] for leg in r["legs"] if leg.get("odds_age_seconds") is not None]
        r["odds_age_seconds"] = max(ages) if ages else None
        r["odds_age_display"] = formatting.format_odds_age(r["odds_age_seconds"])
    return results


def best_effort_combo(legs, stake, max_legs=MAX_LEGS_SEARCHED):
    """Fallback for when nothing lands near the target even at the widest
    tolerance (common for a game with few/low-odds legs): just take the
    highest-payout combination available, so there's still something to show
    rather than nothing."""
    if len(legs) < 2:
        return None
    best = None
    for size in range(2, min(max_legs, len(legs)) + 1):
        for combo in itertools.combinations(legs, size):
            result = _build_result(combo, stake)
            if best is None or result["payout"] > best["payout"]:
                best = result
    return best
