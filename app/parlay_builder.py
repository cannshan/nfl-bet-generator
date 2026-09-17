"""Reusable combo-search core: given a pool of candidate legs, search
combinations for a payout near a target, ranked by the highest combined
hit probability achievable at that payout level. Starts with a tight band
around the target and widens it only when the target band can't produce
anything better than a hopeless lottery ticket (see MIN_PROB_VS_FAIR).

Combined probability is NOT the naive product of the legs' probabilities:
same-game legs are correlated, and app/correlations.py turns per-leg
probabilities into a joint probability using correlations measured from
real outcomes (2018-2025). Legs from different games are independent and
multiply as usual. A parlay also never contains two legs on the same
player (a QB's attempts, completions and passing yards are one bet in
three disguises), or a team's spread alongside its own moneyline.
"""
import itertools
from app import odds_math, formatting, correlations

MAX_LEGS_SEARCHED = 8
CROSS_GAME_CANDIDATE_POOL_SIZE = 16
# A combo landing near the target payout is only accepted if its hit
# probability is at least this fraction of the FAIR probability for that
# payout (stake / payout -- the probability at which the bet breaks even).
# A $5 -> $1000 ticket is fair at 0.5%; anything below 0.25% is a worse
# deal than the payout justifies, so the search widens the payout band
# (accepting a lower payout) until it finds something that clears the bar.
# Scaling the floor to the target -- rather than a fixed 10% -- is what lets
# an ambitious target actually be met: with honest probabilities a 200x
# parlay is never going to be a 10% shot, and a fixed floor would quietly
# collapse every search down to a cheap 3-legger regardless of the target.
MIN_PROB_VS_FAIR = 0.5


def combo_stats(combo, stake, pair_lifts=None):
    dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = correlations.joint_probability(combo, pair_lifts=pair_lifts)
    return dec_odds, payout, combined_prob


def build_result(combo, stake):
    """Full result for display: the joint probability here comes from the
    exact (Monte Carlo) copula evaluation rather than the pairwise
    approximation the search ranks with, plus the fair/breakeven
    probability for the payout and the resulting expected value."""
    dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = correlations.joint_probability_mc(combo)
    breakeven_prob = 1.0 / dec_odds
    matchups = [leg["matchup"] for leg in combo]
    ages = [leg["odds_age_seconds"] for leg in combo if leg.get("odds_age_seconds") is not None]
    return {
        "legs": list(combo),
        "decimal_odds": dec_odds,
        "american_equivalent": odds_math.decimal_to_american(dec_odds),
        "payout": round(payout, 2),
        "stake": stake,
        "combined_prob": combined_prob,
        "breakeven_prob": breakeven_prob,
        "ev_pct": (combined_prob * dec_odds - 1) * 100,
        "num_legs": len(combo),
        "has_same_game_legs": len(set(matchups)) < len(matchups),
        "odds_age_seconds": max(ages) if ages else None,
        "odds_age_display": formatting.format_odds_age(max(ages) if ages else None),
    }


def _combo_allowed(combo):
    """No two legs on the same player, and never both sides/teams of the
    same game-level market (a hedge, not a parlay)."""
    players = [leg["player"] for leg in combo if leg.get("player")]
    if len(players) != len(set(players)):
        return False
    game_markets = [(leg["matchup"], leg["market"]) for leg in combo if not leg.get("player")]
    return len(game_markets) == len(set(game_markets))


def search_near_target(legs, stake, target_payout, max_legs=MAX_LEGS_SEARCHED,
                        tolerances=(0.15, 0.3, 0.5, 0.75, 0.95)):
    """Returns matches (payout within tolerance of target) sorted by highest
    combined probability. Walks tolerance bands tightest to widest and stops
    at the FIRST band whose best combo clears the MIN_PROB_VS_FAIR floor --
    so the target payout is respected as closely as possible, and only gets
    pulled down toward a lower, safer payout when hitting it closely would
    mean a combo priced worse than a lottery ticket. Combo stats are
    computed once against the widest band, then just filtered per tolerance
    level rather than recomputed each time."""
    widest = tolerances[-1]
    low_widest = target_payout * (1 - widest)
    high_widest = target_payout * (1 + widest)
    pair_lifts = {}
    all_combos = []
    for size in range(2, min(max_legs, len(legs)) + 1):
        for combo in itertools.combinations(legs, size):
            if not _combo_allowed(combo):
                continue
            dec_odds, payout, combined_prob = combo_stats(combo, stake, pair_lifts)
            if low_widest <= payout <= high_widest:
                all_combos.append((payout, combined_prob, combo))

    best_matches = None
    for tol in tolerances:
        low = target_payout * (1 - tol)
        high = target_payout * (1 + tol)
        matches = [c for c in all_combos if low <= c[0] <= high]
        if not matches:
            continue
        matches.sort(key=lambda c: c[1], reverse=True)
        best_matches = matches
        best_payout, best_prob, _combo = matches[0]
        if best_prob >= MIN_PROB_VS_FAIR * stake / best_payout:
            break  # close enough to target AND a reasonable shot -- stop here

    if best_matches is None:
        return []
    return [combo for _payout, _prob, combo in best_matches]


def dedupe_best_per_bet(pool):
    """Collapse every variant of the same underlying bet down to whichever
    single variant has the best edge. For player props that's one leg PER
    PLAYER: different Over/Under sides, different books' lines for the same
    stat, and different stats on the same player (a QB's attempts,
    completions and passing yards are ~0.6-0.7 correlated in real outcomes
    -- one bet in three disguises). For game-level markets it's one leg per
    (matchup, market). Different players from the same game can all
    survive; their real correlation is handled by app/correlations.py."""
    best = {}
    for leg in pool:
        key = (leg["matchup"], leg["player"]) if leg.get("player") else (leg["matchup"], leg["market"])
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
    for how historically predictable that stat category is). Same-game legs
    are allowed; their combined probability is correlation-adjusted (see
    app/correlations.py) and each result says whether that happened
    (`has_same_game_legs`) since a sportsbook will reprice such a ticket in
    its own Same Game Parlay builder rather than honor the naive product of
    the individual prices."""
    legs = _drop_redundant_favorite_bets(dedupe_best_per_bet(pool))
    legs = sorted(legs, key=lambda l: l["edge"] - 0.05 * l.get("category_cv", 0.5), reverse=True)[:pool_size]
    matches = search_near_target(legs, stake, target_payout)
    return [build_result(combo, stake) for combo in matches[:num_results]]


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
            if not _combo_allowed(combo):
                continue
            dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
            if best is None or dec_odds > best[0]:
                best = (dec_odds, combo)
    return build_result(best[1], stake) if best else None
