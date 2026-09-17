from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, redirect, url_for
from app.config import DEFAULT_STAKE, DEFAULT_TARGET_PAYOUT, BOOKMAKER_KEY
from app import value_finder, odds_client, game_cards, parlay_builder, expert_insight, injury_client, tracking, cache_utils, formatting


def _coherent(pool):
    """Legs whose side doesn't contradict the expert read shown on them
    (see game_cards._apply_expert_insight)."""
    return [leg for leg in pool if not leg.get("contradicts_expert")]

# static_folder points at the repo-root public/ directory (Vercel's
# convention -- it serves public/** from its CDN and Flask's own
# app.static_folder is explicitly unsupported there), with static_url_path=""
# so both local dev (`python run.py`) and Vercel resolve the same file at the
# same URL (e.g. /style.css) from the same single copy on disk.
app = Flask(__name__, static_folder="../public", static_url_path="")
# Kickoff times come from the odds feed as UTC; every place the UI shows
# one goes through this so they're always Eastern.
app.jinja_env.filters["kickoff_et"] = formatting.format_kickoff_et


@app.before_request
def _reset_request_cache():
    cache_utils.reset_request_cache()


POSITION_ORDER = {"QB": 0, "RB": 1, "FB": 2, "WR": 3, "TE": 4}
# Out/Doubtful/Questionable are fresh, this-week game-time decisions; IR is a
# longer-standing, already-known absence -- rank it last so the top of the
# default view surfaces new/actionable news, not old news everyone already knows.
STATUS_SEVERITY = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Injured Reserve": 3}
INJURED_STATUSES = set(STATUS_SEVERITY.keys())


def _offensive_injuries():
    """Actually-injured players only (excludes "Active" -- most of those are
    routine post-game recap notes, not injury news), most severe/impactful
    first. Collapsed by default in the UI; the full list renders when
    expanded, so no truncation here."""
    injuries = [i for i in injury_client.get_offensive_injuries() if i["status"] in INJURED_STATUSES]
    injuries.sort(key=lambda i: (
        STATUS_SEVERITY.get(i["status"], 9), POSITION_ORDER.get(i["position"], 9), i["team"], i["player"]
    ))
    return injuries


def _record_suggestions(best_odds_parlays, season, week):
    """Logs every leg actually shown to the user -- the real suggestions,
    not the full candidate pool -- so the track record reflects what this
    app told you before any outcome was known. Best-effort: a logging
    failure should never break the page. One log_suggestions call per
    parlay is independent of every other, so they run concurrently rather
    than one at a time (this used to be a real chunk of page load time with
    a full slate of games)."""
    calls = [(parlay["legs"], f"best_odds_parlay_{tracking.MODEL_VERSION}") for parlay in best_odds_parlays]
    if not calls:
        return

    def _log(item):
        legs, section = item
        try:
            tracking.log_suggestions(legs, season, week, section=section)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_log, calls))


def _dashboard_params():
    return (
        request.args.get("stake", type=float) or DEFAULT_STAKE,
        request.args.get("target", type=float) or DEFAULT_TARGET_PAYOUT,
        request.args.get("props", "1") != "0",
        request.args.get("insight", "1") != "0",
        request.args.get("focus_game", ""),
    )


def _parse_event_filter(focus_game):
    """focus_game is an "Away Team @ Home Team" string from the dropdown, or
    "" for no filter (all games). Returns a {(home, away)} set for
    player_props.get_player_prop_candidates, or None for no filter."""
    if not focus_game or " @ " not in focus_game:
        return None
    away, home = focus_game.split(" @ ", 1)
    return {(home, away)}


def _available_games():
    """Games list for the focus-game dropdown, read from whatever's already
    cached for get_events() (a free call -- doesn't cost odds-quota either
    way, but this still respects the current cache_utils mode like
    everything else, so a plain page view doesn't force a fetch just to
    populate a dropdown). Empty until the first refresh on a new deployment."""
    try:
        events = odds_client.get_events()
    except Exception:
        return []
    seen = set()
    games = []
    for e in events:
        home, away = e.get("home_team"), e.get("away_team")
        if not home or not away or (home, away) in seen or formatting.has_kicked_off(e.get("commence_time")):
            continue
        seen.add((home, away))
        games.append({
            "value": f"{away} @ {home}",  # what the filter keys on -- unchanged
            "kickoff_iso": e.get("commence_time") or "",
            "kickoff_et": formatting.format_kickoff_et(e.get("commence_time")),
        })
    # Kickoff order, so the dropdown reads like the week's schedule.
    games.sort(key=lambda g: g["kickoff_iso"])
    return games


def _run_bets_pipeline(stake, target, include_props, include_insight, event_filter=None):
    """The live-data pipeline: settle old predictions, pull odds/model data,
    build the best-odds parlay pool, log new suggestions, capture closing
    lines. Every external fetch inside this call chain checks
    cache_utils.live_fetch_allowed() before ever touching a network -- so
    this is safe to call from a plain page view (cache_utils mode 'passive':
    nothing live happens, whatever's cached gets reused regardless of age)
    as well as from an explicit refresh action (mode 'active': stale/missing
    cache entries actually refetch). Callers are responsible for setting the
    mode first. `event_filter`: see player_props.get_player_prop_candidates.

    The tracking writes below (settling old predictions, logging new
    suggestions, capturing closing lines) only matter when the pool actually
    might have changed, i.e. during an explicit refresh -- on a plain page
    view the pool is whatever was already cached, so these would just be
    idempotent no-op writes hitting Supabase for no reason (measured: ~240
    redundant round trips on a single passive page load). Gated on the same
    active/passive mode as the live fetches for that reason."""
    error = None
    pool, meta, best_odds_parlays = [], {}, []
    if cache_utils.live_fetch_allowed():
        try:
            tracking.settle_pending()
        except Exception:
            pass
    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool(
            include_props=include_props, event_filter=event_filter,
        )
        if meta.get("error"):
            error = meta["error"]
        elif pool:
            # Expert research still nudges prop probabilities in place (same
            # signal that used to only matter for Same Game Parlay
            # selection) so Statistically Best Bets benefits from it too.
            game_cards.apply_expert_insight(pool, include_insight=include_insight)
            if event_filter:
                # A chosen focus game means the ticket is built from THAT
                # game only -- its props and its spread/total/moneyline --
                # not just "props from that game plus whatever else is
                # cheapest across the slate." (The props fetch was already
                # limited to it; game-level odds come from one bulk call
                # covering every game, so they're filtered here.)
                focus_matchups = {f"{away} @ {home}" for home, away in event_filter}
                pool = [leg for leg in pool if leg["matchup"] in focus_matchups]
            best_odds_parlays = parlay_builder.find_best_odds_parlays(_coherent(pool), stake, target)
            if cache_utils.live_fetch_allowed():
                _record_suggestions(best_odds_parlays, meta.get("season"), meta.get("week"))
                try:
                    tracking.capture_closing_lines(pool)
                except Exception:
                    pass
    except odds_client.OddsApiError as e:
        error = str(e)
    except Exception as e:  # surface any other failure plainly rather than a blank page
        error = f"Something went wrong pulling live data: {e}"
    return error, best_odds_parlays, meta


@app.route("/")
def dashboard():
    """Plain page view -- NEVER makes a live API call, no matter how old the
    cache is. Only the /refresh-bets and /refresh-injuries routes are allowed
    to actually fetch anything; this just renders whatever's already cached
    (or an empty "click refresh" state on a brand new deployment)."""
    stake, target, include_props, include_insight, focus_game = _dashboard_params()
    cache_utils.set_mode("passive")

    error, best_odds_parlays, meta = _run_bets_pipeline(
        stake, target, include_props, include_insight, _parse_event_filter(focus_game),
    )
    available_games = _available_games()
    odds_api_quota = odds_client.get_quota_usage()

    # In passive mode a missing-data "error" almost always just means nothing
    # has ever been refreshed yet on this deployment, not a real failure --
    # a genuine API/network failure can only happen during an active refresh.
    never_refreshed = bool(error) and "No historical game data" in (error or "")
    if never_refreshed:
        error = None

    return render_template(
        "index.html",
        error=error,
        never_refreshed=never_refreshed,
        best_odds_parlays=best_odds_parlays,
        meta=meta,
        stake=stake,
        target=target,
        include_props=include_props,
        include_insight=include_insight,
        insight_configured=expert_insight.is_configured(),
        offensive_injuries=_offensive_injuries(),
        focus_game=focus_game,
        available_games=available_games,
        odds_api_quota=odds_api_quota,
        offer_book=_offer_book_title(best_odds_parlays),
    )


def _offer_book_title(parlays):
    """Display name of the configured offer book, taken from a leg actually
    shown (the feed's own title, e.g. "DraftKings") so the heading and the
    legs agree; None when every book is in play."""
    if not BOOKMAKER_KEY:
        return None
    for parlay in parlays:
        for leg in parlay["legs"]:
            if leg.get("bookmaker"):
                return leg["bookmaker"]
    return BOOKMAKER_KEY.capitalize()


@app.route("/refresh-bets")
def refresh_bets():
    """The ONLY thing that triggers live odds/model/expert-insight calls.
    Runs the fetch pipeline in 'active' mode (refetches whatever's actually
    stale, reuses whatever's still fresh) purely to warm the cache, then
    redirects back to / so the URL and any later reload stay on the cheap,
    cache-only path. `focus_game`, when set, is the main lever for keeping
    this cheap: player-props cost one live request PER game, and
    get_events() can list several weeks' worth uncapped, so narrowing to a
    single game the user actually cares about can cut a refresh from 30+
    quota-consuming requests down to 2."""
    stake, target, include_props, include_insight, focus_game = _dashboard_params()
    cache_utils.set_mode("active")
    try:
        _run_bets_pipeline(
            stake, target, include_props, include_insight, _parse_event_filter(focus_game),
        )
    finally:
        cache_utils.set_mode("passive")
    return redirect(url_for(
        "dashboard", stake=stake, target=target,
        props=1 if include_props else 0, insight=1 if include_insight else 0,
        focus_game=focus_game,
    ))


@app.route("/refresh-injuries")
def refresh_injuries():
    """Separate, lightweight refresh for just the injury report -- free
    (ESPN, no quota), so this lets you check for injury news without
    spending any odds-quota re-fetching everything else. Note: the model's
    own injury-based adjustments (QB-out penalty, prop discounts) won't
    reflect this until the next /refresh-bets."""
    cache_utils.set_mode("active")
    try:
        injury_client.get_offensive_injuries()
    finally:
        cache_utils.set_mode("passive")
    return redirect(url_for("dashboard", **request.args))


@app.route("/track-record")
def track_record():
    # Settling predictions is a write, not a fetch -- handled by /refresh-bets
    # and /refresh-settlements (both active mode) instead of on every plain
    # visit here, which used to re-run it as a no-op every time.
    cache_utils.set_mode("passive")
    return render_template("track_record.html", record=tracking.get_track_record())


@app.route("/refresh-settlements")
def refresh_settlements():
    """Checks pending predictions against real results -- free (ESPN via
    nflverse, no odds-quota involved), separate from /refresh-bets so
    checking for results never costs anything."""
    cache_utils.set_mode("active")
    try:
        tracking.settle_pending()
    finally:
        cache_utils.set_mode("passive")
    return redirect(url_for("track_record"))


def _leg_signature(d):
    """Identifies 'the same underlying bet' for exclude/dedup purposes: a
    player prop is keyed by PLAYER regardless of stat/line/side (a player's
    different stats are one correlated bet, and a parlay never holds two
    legs on the same player), a game-level bet by its market -- a matchup
    only has one live spread/total/moneyline market at a time, so that's
    specific enough."""
    player = d.get("player")
    if player:
        return ("prop", player)
    return ("game", d.get("market"))


def _exact_signature(d):
    """The specific bet being replaced -- player + stat, so a redo can still
    offer the same player's OTHER stat as the swap-in."""
    player = d.get("player")
    if player:
        return ("prop", player, d.get("stat_category"))
    return ("game", d.get("market"))


@app.route("/api/redo-leg", methods=["POST"])
def redo_leg():
    """Swaps one leg of an already-built parlay for a different, still-live
    candidate from the same game, chosen IN THE CONTEXT OF THE OTHER LEGS:
    every candidate is scored as part of the whole ticket (correlation-
    adjusted joint probability, payout near the target) rather than on its
    own. A same-game ticket is usually a correlated stack -- its hit chance
    comes largely from legs that tend to win together -- so a replacement
    picked on its standalone merits could be negatively correlated with
    the rest and collapse the ticket (seen in practice: one swap took a
    card from -10% EV to -46%). `exclude` carries the other legs with the
    fields the joint calculation needs; `stake`/`target` are the card's."""
    data = request.get_json(silent=True) or {}
    matchup = data.get("matchup")
    current = data.get("current") or {}
    exclude = data.get("exclude") or []
    show_matchup = bool(data.get("show_matchup"))
    try:
        stake = float(data.get("stake") or DEFAULT_STAKE)
        target = float(data.get("target") or DEFAULT_TARGET_PAYOUT)
    except (TypeError, ValueError):
        stake, target = DEFAULT_STAKE, DEFAULT_TARGET_PAYOUT

    if not matchup:
        return jsonify({"error": "Missing matchup."}), 400

    # Passive: pick from whatever's already cached rather than spending a
    # fresh odds-API call just to redo one leg -- click "Refresh Bets" first
    # if you want this to consider genuinely new lines.
    cache_utils.set_mode("passive")
    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool()
    except Exception as e:
        return jsonify({"error": f"Couldn't refresh live data: {e}"}), 500

    if meta.get("error"):
        return jsonify({"error": meta["error"]}), 400

    # Same research + coherence rule as the main pipeline (cached insight
    # only -- passive mode never spends a live call), so a swapped-in leg
    # can't contradict the note it displays either.
    game_cards.apply_expert_insight(pool, include_insight=True)
    game_legs = [l for l in _coherent(pool) if l["matchup"] == matchup]
    excluded = {_leg_signature(e) for e in exclude}
    excluded_exact = {_exact_signature(current)}

    player = current.get("player")
    if player:
        search_order = [
            [l for l in game_legs if l.get("player") == player],
            [l for l in game_legs if l.get("player") and l.get("player") != player],
            [l for l in game_legs if not l.get("player")],
        ]
    else:
        search_order = [
            [l for l in game_legs if not l.get("player")],
            [l for l in game_legs if l.get("player")],
        ]

    others = []
    for e in exclude:
        try:
            others.append({**e, "decimal_odds": float(e["decimal_odds"]), "model_prob": float(e["model_prob"])})
        except (KeyError, TypeError, ValueError):
            pass  # an old client without the stats fields -> fall back to standalone scoring below

    def in_context(leg):
        """(keeps payout near target, joint hit probability) -- the same
        objective the original search used, applied to this one swap."""
        combo = others + [leg]
        if not parlay_builder._combo_allowed(combo):
            return (-1, -1.0)
        dec_odds, payout, joint = parlay_builder.combo_stats(combo, stake)
        near_target = 1 if abs(payout - target) <= 0.5 * target else 0
        return (near_target, joint)

    score = in_context if others else parlay_builder.rank_score

    candidate = None
    for group in search_order:
        options = [
            l for l in group
            if _leg_signature(l) not in excluded and _exact_signature(l) not in excluded_exact
        ]
        if others:
            options = [l for l in options if in_context(l)[0] >= 0]
        if options:
            candidate = max(options, key=score)
            break

    if candidate is None:
        return jsonify({"error": "No alternative leg available for this game right now."}), 404

    template = app.jinja_env.get_template("_leg.html")
    html = template.module.render_leg(candidate, show_matchup=show_matchup)
    return jsonify({"leg": candidate, "html": str(html)})


@app.route("/api/parlay-stats", methods=["POST"])
def parlay_stats():
    """Recomputes a card's payout / combined probability / EV after the
    client swaps a leg. Done server-side because the combined probability
    is correlation-adjusted (app/correlations.py) -- it is not the naive
    product of the legs' probabilities, so the browser can't just multiply.
    The client sends each leg's identifying fields straight from the leg's
    data-* attributes."""
    data = request.get_json(silent=True) or {}
    legs = data.get("legs") or []
    try:
        stake = float(data.get("stake") or DEFAULT_STAKE)
        for leg in legs:
            leg["decimal_odds"] = float(leg["decimal_odds"])
            leg["model_prob"] = float(leg["model_prob"])
    except (TypeError, ValueError, KeyError):
        return jsonify({"error": "Bad leg data."}), 400
    if len(legs) < 1:
        return jsonify({"error": "No legs."}), 400
    result = parlay_builder.build_result(legs, stake)
    return jsonify({
        "payout": result["payout"],
        "american_equivalent": result["american_equivalent"],
        "combined_prob": result["combined_prob"],
        "breakeven_prob": result["breakeven_prob"],
        "ev_pct": result["ev_pct"],
        "has_same_game_legs": result["has_same_game_legs"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5057)
