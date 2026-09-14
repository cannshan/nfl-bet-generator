from flask import Flask, render_template, request, jsonify
from app.config import DEFAULT_STAKE, DEFAULT_TARGET_PAYOUT
from app import value_finder, odds_client, game_cards, parlay_builder, expert_insight, injury_client, tracking

app = Flask(__name__)

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


def _record_suggestions(cards, best_odds_parlays, season, week):
    """Logs every leg actually shown to the user -- the real suggestions,
    not the full candidate pool -- so the track record reflects what this
    app told you before any outcome was known. Best-effort: a logging
    failure should never break the page."""
    try:
        for card in cards:
            tracking.log_suggestions(card["legs"], season, week, section="same_game_parlay")
        for parlay in best_odds_parlays:
            tracking.log_suggestions(parlay["legs"], season, week, section="best_odds_parlay")
    except Exception:
        pass


@app.route("/")
def dashboard():
    stake = request.args.get("stake", type=float) or DEFAULT_STAKE
    target = request.args.get("target", type=float) or DEFAULT_TARGET_PAYOUT
    include_props = request.args.get("props", "1") != "0"
    include_insight = request.args.get("insight", "1") != "0"

    error = None
    pool, meta, cards, best_odds_parlays = [], {}, [], []

    try:
        tracking.settle_pending()
    except Exception:
        pass

    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool(include_props=include_props)
        if meta.get("error"):
            error = meta["error"]
        elif pool:
            cards = game_cards.build_game_cards(pool, stake, target, include_insight=include_insight)
            best_odds_parlays = parlay_builder.find_best_odds_parlays(pool, stake, target)
            _record_suggestions(cards, best_odds_parlays, meta.get("season"), meta.get("week"))
    except odds_client.OddsApiError as e:
        error = str(e)
    except Exception as e:  # surface any other failure plainly rather than a blank page
        error = f"Something went wrong pulling live data: {e}"

    return render_template(
        "index.html",
        error=error,
        game_cards=cards,
        best_odds_parlays=best_odds_parlays,
        meta=meta,
        stake=stake,
        target=target,
        include_props=include_props,
        include_insight=include_insight,
        insight_configured=expert_insight.is_configured(),
        offensive_injuries=_offensive_injuries(),
    )


@app.route("/track-record")
def track_record():
    try:
        tracking.settle_pending()
    except Exception:
        pass
    return render_template("track_record.html", record=tracking.get_track_record())


@app.route("/api/refresh")
def api_refresh():
    stake = request.args.get("stake", type=float) or DEFAULT_STAKE
    target = request.args.get("target", type=float) or DEFAULT_TARGET_PAYOUT
    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool()
        cards = game_cards.build_game_cards(pool, stake, target) if pool else []
        best_odds_parlays = parlay_builder.find_best_odds_parlays(pool, stake, target) if pool else []
        return jsonify({"game_cards": cards, "best_odds_parlays": best_odds_parlays, "meta": meta})
    except odds_client.OddsApiError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5057)
