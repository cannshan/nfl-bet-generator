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
import math
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
# The three tickets shown should be genuinely different bets, not the same
# ticket with one leg swapped (which is what "top 3 by probability" gives,
# since the best combo's neighbors are always next). Each later ticket may
# share at most this fraction of its legs with any earlier one; if the
# band can't supply enough that different, the limit relaxes one leg at a
# time rather than showing near-duplicates or nothing.
MAX_SHARED_LEG_FRACTION = 0.5
# A leg only goes into a parlay if it is at least break-even at the price
# offered (edge = model probability - 1/decimal odds, i.e. EV after vig).
# The tracked record showed why: 98 of the first 151 settled v2 legs were
# NEGATIVE-edge -- picked because they were likely to hit, not because the
# price was right -- so every ticket stacked the book's vig leg after leg.
MIN_EDGE_FOR_PARLAY_LEG = 0.0
# Every game must still get a ticket. When break-even-or-better legs alone
# can't make one (common for a single focus game, or late in a slate when
# most games have kicked off), the floor relaxes one step at a time and the
# ticket is flagged (relaxed_edge) so the card can say so -- its EV is shown
# honestly, negative or not. None = any pregame, non-contradicting leg.
EDGE_FALLBACK_TIERS = (MIN_EDGE_FOR_PARLAY_LEG, -0.025, -0.05, None)
# "Positive" tickets: prefer legs where a player OVERachieves (Overs) over
# Unders. A preference, never a filter -- every leg still has to clear
# MIN_EDGE_FOR_PARLAY_LEG. Kept small on purpose: the prop backtest found
# Overs hit slightly LESS often than priced (47.5% vs 49.3%; rush-yard Overs
# 41.7%), so a big tilt would trade real hit chance for vibes.
OVER_RANK_BONUS = 0.02    # added to a leg's rank_score (edge units) when it's an Over
OVER_TICKET_BOOST = 1.03  # in ticket RANKING only, each Over leg counts its joint prob 3% higher


def parlay_eligible(leg, min_edge=MIN_EDGE_FOR_PARLAY_LEG):
    """Whether a pool leg may go on a ticket: priced at `min_edge` or better
    (None = no price floor; see EDGE_FALLBACK_TIERS), not contradicting the
    research note it displays, and pregame. The odds feed keeps a game
    listed while it's being played, with in-play prices; the model is
    pregame-only, so a started game's prices are never legs (seen in
    practice: a live Over 43.5 at +188 topped the edge list)."""
    return (
        (min_edge is None or leg.get("edge", -1.0) >= min_edge)
        and not leg.get("contradicts_expert")
        and not formatting.has_kicked_off(leg.get("commence_time"))
    )


def sgp_lift(combo, exact=False, book_lifts=None):
    """How much a sportsbook's same-game parlay builder shrinks the payout
    below the plain product of the legs' prices. DraftKings prices legs
    from the same game as ONE correlated bet: it estimates their joint
    probability from its own leg prices plus their correlation, so a
    positively correlated stack pays less than the multiplied price. The
    estimate here applies the same correlation table this app uses for its
    own joint probability to the BOOK's implied probabilities (1/decimal),
    per same-game group: lift = joint / product. Never below 1 -- a book
    doesn't pay MORE than the product for a negatively correlated group --
    and books add extra SGP hold on top, so the estimated payout is, if
    anything, still a little generous. Without this the search treated the
    correlation as free money: tickets showed +79% EV at prices no book
    would actually pay."""
    groups = {}
    for leg in combo:
        groups.setdefault(leg["matchup"], []).append(leg)
    lift = 1.0
    for group in groups.values():
        if len(group) < 2:
            continue
        book_probs = [1.0 / leg["decimal_odds"] for leg in group]
        if exact:
            joint = correlations.joint_probability_mc(group, probs=book_probs)
        else:
            joint = correlations.joint_probability(group, probs=book_probs, pair_lifts=book_lifts)
        lift *= max(1.0, joint / math.prod(book_probs))
    return lift


def combo_stats(combo, stake, pair_lifts=None, book_lifts=None):
    """(decimal odds, payout, joint probability) for ranking in the search --
    the odds already reflect the estimated same-game repricing (sgp_lift),
    so the search aims at payouts a book would actually pay. `book_lifts`
    caches pairwise lifts at the book's probabilities, like `pair_lifts`
    does at the model's (keys are leg-object id pairs, so the two caches
    must stay separate)."""
    naive = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    dec_odds = naive / sgp_lift(combo, book_lifts=book_lifts)
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = correlations.joint_probability(combo, pair_lifts=pair_lifts)
    return dec_odds, payout, combined_prob


def build_result(combo, stake):
    """Full result for display: the joint probability here comes from the
    exact (Monte Carlo) copula evaluation rather than the pairwise
    approximation the search ranks with, plus the fair/breakeven
    probability for the payout and the resulting expected value."""
    naive_dec_odds = odds_math.parlay_decimal_odds([leg["decimal_odds"] for leg in combo])
    lift = sgp_lift(combo, exact=True)
    dec_odds = naive_dec_odds / lift
    payout = odds_math.payout_for_stake(dec_odds, stake)
    combined_prob = correlations.joint_probability_mc(combo)
    breakeven_prob = 1.0 / dec_odds
    matchups = [leg["matchup"] for leg in combo]
    ages = [leg["odds_age_seconds"] for leg in combo if leg.get("odds_age_seconds") is not None]
    kickoffs = sorted(leg["commence_time"] for leg in combo if leg.get("commence_time"))
    return {
        "legs": list(combo),
        "decimal_odds": dec_odds,
        "naive_decimal_odds": naive_dec_odds,
        # True when same-game legs made the estimated SGP price lower than
        # the multiplied price (see sgp_lift) -- the payout is an estimate.
        "sgp_repriced": lift > 1.0001,
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
        # Earliest kickoff among the legs, Eastern time -- the moment the
        # ticket has to be placed by (and the first leg starts settling).
        "kickoff_et": formatting.format_kickoff_et(kickoffs[0]) if kickoffs else None,
        "spans_multiple_kickoffs": len(set(kickoffs)) > 1,
    }


def _combo_allowed(combo):
    """No two legs on the same player, and never both sides/teams of the
    same game-level market (a hedge, not a parlay)."""
    players = [leg["player"] for leg in combo if leg.get("player")]
    if len(players) != len(set(players)):
        return False
    game_markets = [(leg["matchup"], leg["market"]) for leg in combo if not leg.get("player")]
    if len(game_markets) != len(set(game_markets)):
        return False
    # One team's moneyline with the OTHER team's spread is a hedge too (it
    # only wins on a narrow margin) -- seen as a -49% EV fallback ticket.
    margin_teams = {}
    for leg in combo:
        if not leg.get("player") and leg["market"] in ("Spread", "Moneyline"):
            teams = margin_teams.setdefault(leg["matchup"], set())
            teams.add(leg.get("team") or leg["selection"])
            if len(teams) > 1:
                return False
    return True


def search_near_target(legs, stake, target_payout, max_legs=MAX_LEGS_SEARCHED,
                        tolerances=(0.15, 0.3, 0.5, 0.75, 0.95), required=()):
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
    pair_lifts, book_lifts = {}, {}
    all_combos = []
    for combo in _combos(legs, max_legs, required):
            if not _combo_allowed(combo):
                continue
            dec_odds, payout, combined_prob = combo_stats(combo, stake, pair_lifts, book_lifts)
            if low_widest <= payout <= high_widest:
                all_combos.append((payout, combined_prob, combo))

    best_matches = None
    for tol in tolerances:
        low = target_payout * (1 - tol)
        high = target_payout * (1 + tol)
        matches = [c for c in all_combos if low <= c[0] <= high]
        if not matches:
            continue
        matches.sort(key=lambda c: c[1] * over_weight(c[2]), reverse=True)
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
        if key not in best or rank_score(leg) > rank_score(best[key]):
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


def _leg_key(leg):
    return (leg["matchup"], leg["player"]) if leg.get("player") else (leg["matchup"], leg["market"], leg["selection"])


def pick_diverse(combos, num_results, max_shared_fraction=MAX_SHARED_LEG_FRACTION):
    """Greedy: walk the combos in rank order, keeping one only if it shares
    no more than the allowed number of legs with every combo already kept.
    Relaxes the allowance a leg at a time if that yields fewer than
    num_results, so the result is always the most-different set the band
    can offer, in probability order within the constraint."""
    if not combos:
        return []
    size = max(len(c) for c in combos)
    allowance = int(size * max_shared_fraction)
    while True:
        chosen, chosen_keys = [], []
        for combo in combos:
            keys = {_leg_key(leg) for leg in combo}
            if all(len(keys & prior) <= allowance for prior in chosen_keys):
                chosen.append(combo)
                chosen_keys.append(keys)
                if len(chosen) == num_results:
                    return chosen
        if allowance >= size or len(chosen) == len(combos):
            return chosen
        allowance += 1


def over_weight(combo):
    """Ranking multiplier that tilts ticket choice toward Overs (see
    OVER_TICKET_BOOST). Never changes the displayed probability."""
    return OVER_TICKET_BOOST ** sum(1 for leg in combo if leg.get("side") == "Over")


def rank_score(leg):
    """How strongly a leg deserves a place in the candidate pool: its edge
    (EV after vig), discounted for how erratic its stat category is, plus
    the priority credible research gives a leg it agrees with (or takes
    from one it disagrees with) -- see game_cards.RESEARCH_PRIORITY -- plus
    the small preference for Overs (OVER_RANK_BONUS)."""
    over_bonus = OVER_RANK_BONUS if leg.get("side") == "Over" else 0.0
    return leg["edge"] - 0.05 * leg.get("category_cv", 0.5) + leg.get("research_priority", 0.0) + over_bonus


def find_best_odds_parlays(pool, stake, target_payout, num_results=3, pool_size=CROSS_GAME_CANDIDATE_POOL_SIZE):
    """The 'statistically best bets' parlay: no restriction on which games a
    leg can come from -- purely chases the highest achievable hit probability
    at the target payout using every signal the model has (edge, discounted
    for how historically predictable that stat category is). Same-game legs
    are allowed; their combined probability is correlation-adjusted (see
    app/correlations.py) and their payout is the estimated same-game price
    (sgp_lift).

    Tries break-even-or-better legs first and relaxes the price floor a step
    at a time (EDGE_FALLBACK_TIERS) only when that can't produce a ticket, so
    any game with at least two pregame legs always gets one. If nothing lands
    near the target even then, falls back to the highest-payout combo.
    Every result carries `edge_floor` and `relaxed_edge`."""
    legs = []
    for min_edge in EDGE_FALLBACK_TIERS:
        legs = _drop_redundant_favorite_bets(dedupe_best_per_bet([l for l in pool if parlay_eligible(l, min_edge)]))
        legs = sorted(legs, key=rank_score, reverse=True)[:pool_size]
        matches = search_near_target(legs, stake, target_payout)
        if matches:
            return [_with_edge_floor(build_result(combo, stake), min_edge)
                    for combo in pick_diverse(matches, num_results)]
    fallback = best_effort_combo(legs, stake)
    return [_with_edge_floor(fallback, None)] if fallback else []


def _with_edge_floor(result, min_edge):
    result["edge_floor"] = min_edge
    result["relaxed_edge"] = min_edge != MIN_EDGE_FOR_PARLAY_LEG
    return result


def _combos(legs, max_legs, required=()):
    """Every combination of 2..max_legs legs; with `required`, every
    combination that CONTAINS those legs (e.g. Leah's hair players), the
    rest drawn from `legs`."""
    required = tuple(required)
    low = max(0, 2 - len(required))
    high = min(max_legs - len(required), len(legs))
    for size in range(low, high + 1):
        for extra in itertools.combinations(legs, size):
            yield required + extra


def best_effort_combo(legs, stake, max_legs=MAX_LEGS_SEARCHED, required=()):
    """Fallback for when nothing lands near the target even at the widest
    tolerance (a game with only a few legs, e.g. next week's games before
    their props are posted): the highest-payout combination that is still a
    fair shot -- hit chance at least MIN_PROB_VS_FAIR of its break-even
    chance, the same floor the main search uses. Plain "highest payout"
    used to pick self-contradicting hedges (one team's moneyline with the
    other team's spread: -96% EV) because they pay the most. If no combo
    clears the floor, the best-EV combo instead."""
    if len(legs) + len(required) < 2:
        return None
    pair_lifts, book_lifts = {}, {}
    best, best_ev = None, None
    for combo in _combos(legs, max_legs, required):
            if not _combo_allowed(combo):
                continue
            dec_odds, _payout, prob = combo_stats(combo, stake, pair_lifts, book_lifts)
            if prob >= MIN_PROB_VS_FAIR / dec_odds and (best is None or dec_odds > best[0]):
                best = (dec_odds, combo)
            if best_ev is None or prob * dec_odds > best_ev[0]:
                best_ev = (prob * dec_odds, combo)
    chosen = best or best_ev
    return build_result(chosen[1], stake) if chosen else None
