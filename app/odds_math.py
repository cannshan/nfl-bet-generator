def american_to_decimal(american):
    american = float(american)
    if american > 0:
        return 1 + american / 100.0
    return 1 + 100.0 / abs(american)


def american_to_implied_prob(american):
    american = float(american)
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def devig_two_way(prob_a, prob_b):
    """Normalize two implied probabilities that sum to >1 (due to vig) back to 100%."""
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def parlay_decimal_odds(leg_decimal_odds):
    result = 1.0
    for d in leg_decimal_odds:
        result *= d
    return result


def payout_for_stake(decimal_odds, stake):
    return stake * decimal_odds


def make_leg(matchup, commence_time, market, selection, price_info, model_prob, book_fair_prob):
    """Shared shape for anything the parlay builder can pick a leg from --
    game-level bets and player props alike.

    `edge` is model probability minus the BREAKEVEN probability at the
    offered price (1 / decimal odds), i.e. the bet's expected value after
    the book's vig. An earlier definition compared against the devigged
    fair price instead, which called a bet "+2 points of edge" when it was
    actually -2.5% EV at -110 -- the vig was simply never counted."""
    breakeven_prob = 1.0 / price_info["decimal"]
    edge = model_prob - breakeven_prob
    return {
        "matchup": matchup,
        "commence_time": commence_time,
        "market": market,
        "selection": selection,
        "american_odds": price_info["american"],
        "decimal_odds": price_info["decimal"],
        "bookmaker": price_info["bookmaker"],
        "model_prob": round(model_prob, 4),
        "book_fair_prob": round(book_fair_prob, 4),
        "breakeven_prob": round(breakeven_prob, 4),
        "edge": round(edge, 4),
    }
