"""Reusable combo-search core: given a pool of candidate legs, search
combinations for a payout near a target, ranked by the highest combined
hit probability achievable at that payout level (searches a tight band
around the target first, widening only if nothing is found there).

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


def combo_stats(combo, stake):
    dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = 1.0
    for leg in combo:
        combined_prob *= leg["model_prob"]
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
    combined probability, using the tightest tolerance band that finds any."""
    for tol in tolerances:
        low = target_payout * (1 - tol)
        high = target_payout * (1 + tol)
        matches = []
        for size in range(2, min(max_legs, len(legs)) + 1):
            for combo in itertools.combinations(legs, size):
                dec_odds, payout, combined_prob = combo_stats(combo, stake)
                if low <= payout <= high:
                    matches.append(_build_result(combo, stake))
        if matches:
            matches.sort(key=lambda m: m["combined_prob"], reverse=True)
            return matches
    return []


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
