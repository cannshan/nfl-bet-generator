from app.config import BOOKMAKER_KEY


def is_offer_book(bookmaker):
    """Whether a feed bookmaker entry is the one book legs are offered at
    (config.BOOKMAKER_KEY). Matches the Odds API key, and falls back to the
    title for the fallback feed (which has titles only), so e.g. "DraftKings"
    still matches "draftkings". Everything matches when no book is set."""
    if not BOOKMAKER_KEY:
        return True
    key = (bookmaker.get("key") or "").lower()
    title = (bookmaker.get("title") or "").lower().replace(" ", "")
    return key == BOOKMAKER_KEY or title == BOOKMAKER_KEY


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
    """Strip the vig from two implied probabilities using the POWER method:
    find k so that prob_a**k + prob_b**k = 1. At even prices this is
    identical to plain proportional scaling, but at lopsided prices it
    assigns more of the vig to the longshot -- which is where books
    actually load it (the well-documented favorite-longshot bias).
    Proportional devig gives a +206 underdog vs a -234 favorite 31.8%;
    power gives 31.1%. Small per leg, but the tickets built here lean on
    plus-money legs, and it compounds."""
    if prob_a <= 0 or prob_b <= 0:
        total = prob_a + prob_b
        return (prob_a / total, prob_b / total) if total > 0 else (prob_a, prob_b)
    lo, hi = 0.5, 4.0
    for _ in range(60):
        k = (lo + hi) / 2
        if prob_a ** k + prob_b ** k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    fair_a, fair_b = prob_a ** k, prob_b ** k
    total = fair_a + fair_b
    return fair_a / total, fair_b / total


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
