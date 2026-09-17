"""Per-game ("same-game") parlay suggestions: for each matchup, search that
game's own legs -- mostly player props -- for the combo landing closest to a
target payout with the highest hit probability achievable at that payout
level.

Selection itself (which players/props make the card, not just how they're
described afterward) is influenced by three things beyond raw model edge:
- live expert research (see expert_insight.py) nudges a prop's probability
  before anything is ranked, so a bullish/bearish read can actually change
  which legs get picked, not just decorate the final choice
- each stat category's real historical predictability (lower variance =
  preferred at the margin, see player_props.py's category_cv)
- the moneyline is only included when it clears its own bar, never as filler

Legs within one game are correlated (a team winning big tends to move its
skill players' stat lines with it), so the payout and "combined hit
probability" shown here are illustrative, not a real sportsbook quote --
the actual, correlation-adjusted price only exists in your sportsbook's own
Same Game Parlay builder once you enter the same legs there.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from app import parlay_builder, formatting, expert_insight

PROPS_POOL_SIZE = 12
MONEYLINE_MIN_MODEL_PROB = 0.60
MONEYLINE_MIN_EDGE = 0.03
MAX_OTHER_LEGS = 2  # at most this many non-prop legs (spread/total) added as extra options

# Research nudges probability a small, fixed amount rather than trying to be
# precise -- it's a tiebreaker/confirmation signal on top of the stats model,
# not a replacement for it.
SENTIMENT_PROB_NUDGE = {"bullish": 0.03, "bearish": -0.03}
RELIABILITY_PENALTY_WEIGHT = 0.05  # how much a category's own historical variance discounts its edge for ranking


def _dedupe_best_per_prop(props):
    """Different books sometimes post different lines for the same player/stat;
    keep only the single best-edge line per (player, stat_category)."""
    best = {}
    for leg in props:
        key = (leg.get("player"), leg.get("stat_category"))
        if key not in best or leg["edge"] > best[key]["edge"]:
            best[key] = leg
    return list(best.values())


def _apply_expert_insight(matchup, props, kickoff_et):
    """Fetches insight for every candidate player in this game (before any
    top-N cut) and folds sentiment into model_prob/edge directly, so research
    can change which props are ranked highly enough to be selected -- not
    just annotate whatever the stats model already picked."""
    players = sorted({p["player"] for p in props if p.get("player")})
    if not players:
        return
    try:
        insights = expert_insight.get_expert_insight(matchup, players, kickoff_et)
    except Exception:
        return
    by_player = {i["player"]: i for i in insights}

    for leg in props:
        info = by_player.get(leg.get("player"))
        if not info:
            continue
        leg["expert_note"] = info.get("note")
        leg["expert_sentiment"] = info.get("sentiment")
        nudge = SENTIMENT_PROB_NUDGE.get(info.get("sentiment"))
        if nudge is None:
            continue
        direction = 1 if leg["side"] == "Over" else -1
        new_prob = max(0.01, min(0.99, leg["model_prob"] + nudge * direction))
        leg["model_prob"] = round(new_prob, 4)
        leg["edge"] = round(new_prob - leg["book_fair_prob"], 4)


def _selection_score(leg):
    """Ranking key for which props make the card: edge, discounted a little
    for how historically unpredictable this stat category tends to be."""
    return leg["edge"] - RELIABILITY_PENALTY_WEIGHT * leg.get("category_cv", 0.5)


def _candidate_legs_for_game(legs, matchup, kickoff_et, include_insight):
    """Player props dominate the candidate pool; the moneyline is added only
    when it's a strong pick on its own (not just filler), and a team's own
    spread is excluded when its moneyline is already in since covering a
    spread and winning outright are nearly the same bet."""
    props = _dedupe_best_per_prop([l for l in legs if l["market"].startswith("Player Prop")])

    if include_insight and expert_insight.is_configured():
        _apply_expert_insight(matchup, props, kickoff_et)

    props.sort(key=_selection_score, reverse=True)
    props = props[:PROPS_POOL_SIZE]

    moneylines = [l for l in legs if l["market"] == "Moneyline"]
    best_ml = max(moneylines, key=lambda l: l["model_prob"]) if moneylines else None

    others = []
    if best_ml and best_ml["model_prob"] >= MONEYLINE_MIN_MODEL_PROB and best_ml["edge"] >= MONEYLINE_MIN_EDGE:
        others.append(best_ml)

    def _redundant_with_ml(leg):
        return (
            best_ml is not None
            and leg["market"] == "Spread"
            and leg["selection"].startswith(best_ml["selection"])
        )

    fillers = sorted(
        [l for l in legs if l["market"] in ("Spread", "Total") and not _redundant_with_ml(l)],
        key=lambda l: l["edge"],
        reverse=True,
    )
    others += fillers[: MAX_OTHER_LEGS - len(others)]

    return props + others


def _prefetch_expert_insight(by_game):
    """Each game's expert-insight call is a slow, independent LLM+web-search
    round trip (multiple seconds each) that get_expert_insight caches for 4
    hours -- but the per-game loop in build_game_cards calls it one game at a
    time, so a cold cache (a fresh deployment, or just past the 4h TTL) turns
    into N sequential multi-second calls in a row, dominating the whole
    refresh. Warming them all concurrently first means that same cold-cache
    cost is paid once, in parallel, rather than serially; the loop below then
    always hits a warm cache."""
    if not expert_insight.is_configured():
        return

    def _warm(item):
        matchup, legs = item
        players = sorted({l["player"] for l in legs if l.get("player") and l["market"].startswith("Player Prop")})
        if not players:
            return
        kickoff_et = formatting.format_kickoff_et(legs[0].get("commence_time"))
        try:
            expert_insight.get_expert_insight(matchup, players, kickoff_et)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_warm, by_game.items()))


def build_game_cards(pool, stake, target_payout, include_insight=True):
    by_game = defaultdict(list)
    for leg in pool:
        by_game[leg["matchup"]].append(leg)

    if include_insight:
        _prefetch_expert_insight(by_game)

    cards = []
    for matchup, legs in by_game.items():
        kickoff_et = formatting.format_kickoff_et(legs[0].get("commence_time"))
        candidates = _candidate_legs_for_game(legs, matchup, kickoff_et, include_insight)
        if len(candidates) < 2:
            continue

        matches = parlay_builder.search_near_target(candidates, stake, target_payout)
        best = matches[0] if matches else parlay_builder.best_effort_combo(candidates, stake)
        if not best:
            continue

        best["matchup"] = matchup
        best["commence_time"] = legs[0].get("commence_time")
        best["kickoff_et"] = kickoff_et
        best["hit_target"] = bool(matches)
        best["has_props"] = any(leg.get("player") for leg in best["legs"])
        cards.append(best)

    # Closest to the target payout first, tie-broken by higher hit probability
    cards.sort(key=lambda c: (abs(c["payout"] - target_payout), -c["combined_prob"]))
    return cards
