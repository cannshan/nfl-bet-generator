"""Applies live expert research to player-prop legs before parlay search
runs, so a bullish/bearish read can actually change which props look best
-- not just decorate whatever the stats model already picked.

Used to also build "Same Game Parlay" cards (one best combo per game); that
feature was removed in favor of a single cross-game "Statistically Best
Bets" list, but the research step itself is still worth keeping since it's
a real signal on top of the stats model.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from app import formatting, expert_insight

# Research nudges probability a small, fixed amount rather than trying to be
# precise -- it's a tiebreaker/confirmation signal on top of the stats model,
# not a replacement for it. Was +-3 points back when the stats model itself
# could swing a leg by 40; now that every prop is anchored on the market
# (player_props.py) and the model's own tilt is typically 1-3 points, a
# 3-point sentiment nudge would have been the LARGEST single component of
# most edges -- for an unvalidated, LLM-summarized signal. Scaled to match.
SENTIMENT_PROB_NUDGE = {"bullish": 0.01, "bearish": -0.01}


def _apply_expert_insight(matchup, props, kickoff_et):
    """Fetches insight for every candidate player in this game and folds
    sentiment into model_prob/edge directly, so research can change which
    props rank highest -- not just annotate whatever the stats model
    already picked."""
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


def _prefetch_expert_insight(by_game):
    """Each game's expert-insight call is a slow, independent LLM+web-search
    round trip (multiple seconds each) that get_expert_insight caches for 4
    hours -- so warming them all concurrently first means a cold cache pays
    that cost once, in parallel, rather than serially, one game at a time."""
    if not expert_insight.is_configured():
        return

    def _warm(item):
        matchup, legs = item
        players = sorted({l["player"] for l in legs if l.get("player")})
        if not players:
            return
        kickoff_et = formatting.format_kickoff_et(legs[0].get("commence_time"))
        try:
            expert_insight.get_expert_insight(matchup, players, kickoff_et)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_warm, by_game.items()))


def apply_expert_insight(pool, include_insight=True):
    """Mutates every player-prop leg in `pool` in place with a research-based
    probability nudge, grouped and fetched per game. Safe to call even when
    insight is disabled/unconfigured (no-op) or the pool has no props."""
    if not include_insight or not expert_insight.is_configured():
        return

    by_game = defaultdict(list)
    for leg in pool:
        if leg.get("player") and leg["market"].startswith("Player Prop"):
            by_game[leg["matchup"]].append(leg)
    if not by_game:
        return

    _prefetch_expert_insight(by_game)
    for matchup, props in by_game.items():
        kickoff_et = formatting.format_kickoff_et(props[0].get("commence_time"))
        _apply_expert_insight(matchup, props, kickoff_et)
