"""Leah's Hairlays: player props on guys with notable hair.

The hair list is hand-curated (hairstyles change -- edit HAIR_PLAYERS freely).
"Plays on a regular basis" is enforced by the data rather than the list: a
player only shows up if the books are posting a prop line on him in this
week's cached pool, and books don't hang yardage/reception lines on bench
guys -- and he must have played at least MIN_GAMES_THIS_SEASON games. So a
benched or injured player on the list just drops out.

Each player's leg is the best-priced one the regular pipeline already
found for him (same market-anchored probabilities as the main Bets tab),
so these are real, honestly-priced bets -- the hair is only how the
players were picked.
"""
import datetime as dt
import re
from app.config import DEFAULT_TARGET_PAYOUT
from app import value_finder, game_cards, parlay_builder, roster_client, cache_utils, nflverse_client, formatting, espn_client, odds_client, tracking

# name -> (hair emoji, the look, craziness 1-10). Picked and scored by eye
# from each player's current ESPN headshot (Oct 2026) -- offensive skill
# players only, since those are the only ones the books post props on.
# Craziness decides the page's Top 3. Haircuts change; edit freely.
HAIR_PLAYERS = {
    # The flowing manes
    "George Kittle": ("🦁", "Long flowing ginger mane + full beard", 9),
    "Trevor Lawrence": ("💁", "Shampoo-commercial hair down to the shoulders", 8),
    "T.J. Hockenson": ("🏄", "Long surfer hair + backwards cap", 8),
    "Greg Dulcich": ("🥸", "Long hair + a full 1970s mustache", 9),
    "Trey McBride": ("👱", "Bleached blonde surfer mop", 6),
    # Volume
    "Troy Franklin": ("☁️", "Enormous afro", 10),
    "Dontayvion Wicks": ("☁️", "Big round afro", 8),
    "Zay Flowers": ("🌸", "Big curly fro", 7),
    "Matthew Golden": ("🌪️", "Twisty afro going every direction", 8),
    "Jordan Addison": ("⚡", "Twists sticking straight up", 7),
    "Dawson Knox": ("🐑", "Curly mop + beard", 6),
    "AJ Barner": ("🐑", "Curly mop", 5),
    # Long locs
    "CeeDee Lamb": ("🦁", "Long locs", 5),
    "George Pickens": ("🦁", "Long dreads, maximum swagger", 6),
    "Aaron Jones": ("🦁", "Long dreads past the shoulders", 6),
    "Ashton Jeanty": ("🦁", "Long locs", 5),
    "Marvin Harrison Jr.": ("🦁", "Long locs", 5),
    "Najee Harris": ("🦁", "Long locs", 5),
    "Jakobi Meyers": ("🦁", "Long locs", 5),
    "DeMario Douglas": ("🦁", "Long locs", 5),
    "Rashid Shaheed": ("🧢", "Long dreads + beanie", 6),
    "Rhamondre Stevenson": ("🦁", "Long locs", 5),
    "Malik Nabers": ("🦁", "Locs", 4),
    "Luther Burden III": ("🦁", "Locs", 4),
    "Omarion Hampton": ("🦁", "Locs", 4),
    "Kaleb Johnson": ("🦁", "Long locs", 5),
    "Jameson Williams": ("🦁", "Dreads", 5),
    "James Cook": ("🦁", "Locs", 4),
    "Javonte Williams": ("🦁", "Dreads", 5),
    "Jonathan Taylor": ("🦁", "Braids", 4),
    "Tyjae Spears": ("🦁", "Long dreads", 5),
    "Tre Tucker": ("🦁", "Locs", 4),
    "Pat Bryant": ("🦁", "Locs", 4),
    "Tai Felton": ("🦁", "Dreads", 5),
    "Wan'Dale Robinson": ("🦁", "Locs", 4),
    "Davante Adams": ("🦁", "Locs", 4),
    "Cam Ward": ("🦁", "Dreads", 5),
    "Kyler Murray": ("🦁", "Locs", 4),
    "Bhayshul Tuten": ("🦁", "Dreads", 5),
    "Isaiah Likely": ("☁️", "Curly fro", 6),
    "Oronde Gadsden II": ("🌀", "Curly top", 5),
    "Ty Johnson": ("🌀", "Curly mop + headband", 6),
    "Xavier Hutchinson": ("🌀", "Curly mop", 5),
    # Headwear department
    "Keon Coleman": ("🟡", "Bright yellow durag in his official headshot", 9),
    "Michael Penix Jr.": ("🏴‍☠️", "Durag in his official headshot", 6),
    # Honorable mention
    "Samaje Perine": ("🧔", "Zero hair on top, all of it moved to the beard", 7),
}

TOP_HAIR_COUNT = 3

# "Plays on a regular basis": a prop line this week, a game this season,
# and at least this many games across this season and last.
MIN_GAMES_THIS_SEASON = 3

# The hair parlay is one game's $5 -> ~$1000 ticket: every hair player in
# that game (up to this many) is on it, filled out with the game's best
# other legs by the same rules as the Bets tab.
MAX_HAIR_ANCHORS = 4


def _norm(name):
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", (name or "").lower())
    return re.sub(r"[^a-z]", "", name)


_HAIR_BY_NORM = {_norm(n): (n, emoji, look, crazy) for n, (emoji, look, crazy) in HAIR_PLAYERS.items()}


# Leah's picks lean positive too: a player's Over is taken over his Under
# unless the Under is this much likelier to hit.
OVER_PREFERENCE = 0.05


def _pick_score(leg):
    return leg["model_prob"] + (OVER_PREFERENCE if leg.get("side") == "Over" else 0.0)


def _headshots_by_norm():
    """ESPN headshots keyed like _HAIR_BY_NORM. The roster pass is free
    (ESPN, no odds quota), so unlike odds it may run on a plain page view --
    once a week at most, then it's a cache read."""
    shots = roster_client.get_headshots()
    if not shots:
        cache_utils.set_mode("active")
        try:
            shots = roster_client.get_headshots()
        finally:
            cache_utils.set_mode("passive")
    return {_norm(name): url for name, url in (shots or {}).items()}


def _games_played_this_season(season):
    """{normalized name: games with a stat row this season} from the same
    (cached) nflverse player index the prop model uses."""
    if not season:
        return {}
    try:
        index = nflverse_client.build_player_index(season)
    except Exception:
        return {}
    # (this season, this season + last season): early in a season a starter
    # can have only a game or two, so last season's games count toward
    # "plays regularly" as long as he has played this season too.
    return {
        _norm(name): (sum(1 for _r, w in rows if w >= 1.0), len(rows))
        for name, rows in index.items()
    }


def _iso_z(ts):
    """ESPN writes kickoffs as 2026-10-04T20:25Z; the rest of the app uses
    the odds feed's 2026-10-04T20:25:00Z."""
    if ts and len(ts) == 17 and ts.endswith("Z"):
        return ts[:-1] + ":00Z"
    return ts


def _games_this_week():
    """{team full name: {"matchup", "kickoff_iso", "status"}} for every game
    of the current NFL week (Tuesday-Monday, tracking.season_week_for_kickoff)
    -- finished, in progress and upcoming alike. ESPN's (cached) scoreboard
    lists the whole week; the odds feed's cached events fill in if that
    cache is from another week. Cache reads only."""
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    this_week = tracking.season_week_for_kickoff(now_iso)
    games = {}

    def _add(home, away, kickoff_iso, finished=False):
        if not home or not away or tracking.season_week_for_kickoff(kickoff_iso) != this_week:
            return
        if not formatting.has_kicked_off(kickoff_iso):
            status = "Upcoming"
        else:
            # The cached scoreboard can lag; any game 4h+ past kickoff is over.
            long_ago = formatting.has_kicked_off(
                kickoff_iso, now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4))
            status = "Final" if finished or long_ago else "Kicked off"
        info = {"matchup": f"{away} @ {home}", "kickoff_iso": kickoff_iso, "status": status}
        for team in (home, away):
            if team not in games or (finished and games[team]["status"] != "Final"):
                games[team] = info

    try:
        for event in espn_client.get_current_scoreboard().get("events", []):
            comp = event["competitions"][0]
            sides = {c.get("homeAway"): c["team"]["displayName"] for c in comp["competitors"]}
            finished = event.get("status", {}).get("type", {}).get("state") == "post"
            _add(sides.get("home"), sides.get("away"), _iso_z(event.get("date")), finished)
    except Exception:
        pass
    try:
        for event in odds_client.get_events():
            _add(event.get("home_team"), event.get("away_team"), event.get("commence_time"))
    except Exception:
        pass
    return games


def _top_styles_of_the_week(games_played, picks):
    """The TOP_HAIR_COUNT craziest hairstyles among regulars whose team
    plays this week -- including games already played, so the showcase
    covers the whole week rather than only what's still bettable. A
    player's pregame bet rides along when one is still available."""
    week_games = _games_this_week()
    if not week_games:
        return []
    try:
        rosters = roster_client.get_current_rosters()
    except Exception:
        rosters = {}
    rosters_by_norm = {_norm(name): team for name, team in rosters.items()}
    picks_by_norm = {_norm(p["name"]): p for p in picks}

    candidates = []
    for norm, (name, emoji, look, crazy) in _HAIR_BY_NORM.items():
        this_season, total = games_played.get(norm, (0, 0))
        team = rosters_by_norm.get(norm)
        if not team or team not in week_games or this_season < 1 or total < MIN_GAMES_THIS_SEASON:
            continue
        game = week_games[team]
        pick = picks_by_norm.get(norm)
        candidates.append({
            "name": name, "emoji": emoji, "look": look, "craziness": crazy,
            "matchup": game["matchup"], "status": game["status"],
            "kickoff_et": formatting.format_kickoff_et(game["kickoff_iso"]),
            "leg": pick["leg"] if pick else None,
        })
    # Craziest first; ties go to someone with a bet still available.
    candidates.sort(key=lambda c: (-c["craziness"], c["leg"] is None, c["name"]))
    top = candidates[:TOP_HAIR_COUNT]
    headshots = _headshots_by_norm() if top else {}
    for c in top:
        c["photo"] = headshots.get(_norm(c["name"]))
    return top


def _hair_parlay_for_game(pool, matchup, anchors, stake, target):
    """Best ~target ticket from ONE game that includes every anchor (hair
    player) leg. Uses the main search's price-floor tiers -- break-even legs
    first, relaxing only when the game can't reach the target otherwise
    (flagged relaxed_edge) -- and the highest fair-shot payout as a last
    resort, so a game with a hair player always gets a ticket."""
    anchor_players = {leg["player"] for leg in anchors}
    others = []
    for min_edge in parlay_builder.EDGE_FALLBACK_TIERS:
        others = parlay_builder.dedupe_best_per_bet([
            l for l in pool
            if l["matchup"] == matchup and l.get("player") not in anchor_players
            and parlay_builder.parlay_eligible(l, min_edge)
        ])
        others = sorted(parlay_builder._drop_redundant_favorite_bets(others),
                        key=parlay_builder.rank_score, reverse=True)[:parlay_builder.CROSS_GAME_CANDIDATE_POOL_SIZE]
        # Reaching the target is the point of this ticket: a price tier only
        # counts if it lands within 30% of it; the last (any-leg) tier may
        # widen like the main search does.
        tolerances = (0.15, 0.3) if min_edge is not None else (0.15, 0.3, 0.5, 0.75, 0.95)
        matches = parlay_builder.search_near_target(others, stake, target, tolerances=tolerances, required=anchors)
        if matches:
            result = parlay_builder.build_result(matches[0], stake)
            result["relaxed_edge"] = min_edge != parlay_builder.MIN_EDGE_FOR_PARLAY_LEG
            return result
    result = parlay_builder.best_effort_combo(others, stake, required=anchors)
    if result:
        result["relaxed_edge"] = True
    return result


def get_leahs_bets(stake, focus_game="", target=DEFAULT_TARGET_PAYOUT):
    """Returns (picks, top_hair, parlay, hair_games, selected_game, error):
    every eligible (pregame) hair pick, likeliest first; the week's Top
    Styles (any game this week, played or not -- see
    _top_styles_of_the_week); the stake -> ~target hair parlay for the
    selected game; the games that have hair picks; and which one is shown. Reads only what's already cached --
    callers set cache_utils mode to passive, so this never spends quota."""
    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool()
    except Exception as e:
        return [], [], None, [], "", f"Couldn't read cached odds: {e}"
    if meta.get("error"):
        return [], [], None, [], "", meta["error"]
    game_cards.apply_expert_insight(pool, include_insight=True)
    games_played = _games_played_this_season(meta.get("season"))

    best = {}
    for leg in pool:
        if not leg.get("player") or leg.get("contradicts_expert"):
            continue
        if formatting.has_kicked_off(leg.get("commence_time")):
            continue  # in-play price -- pregame bets only
        hair = _HAIR_BY_NORM.get(_norm(leg["player"]))
        this_season, total = games_played.get(_norm(leg["player"]), (0, 0))
        if not hair or this_season < 1 or total < MIN_GAMES_THIS_SEASON:
            continue
        key = hair[0]
        if key not in best or _pick_score(leg) > _pick_score(best[key][0]):
            best[key] = (leg, hair)

    headshots = _headshots_by_norm() if best else {}
    picks = [
        {"leg": leg, "name": name, "emoji": emoji, "look": look, "craziness": crazy,
         "photo": headshots.get(_norm(leg["player"]))}
        for leg, (name, emoji, look, crazy) in best.values()
    ]
    picks.sort(key=lambda p: -p["leg"]["model_prob"])
    top_hair = _top_styles_of_the_week(games_played, picks)

    # Games that have a pregame hair pick, in kickoff order; the parlay is
    # for the chosen one (default: the soonest).
    games = {}
    for p in picks:
        games.setdefault(p["leg"]["matchup"], {"value": p["leg"]["matchup"], "kickoff_iso": p["leg"].get("commence_time") or "", "hair": []})["hair"].append(p["name"])
    hair_games = sorted(games.values(), key=lambda g: g["kickoff_iso"])
    for g in hair_games:
        g["kickoff_et"] = formatting.format_kickoff_et(g["kickoff_iso"])
    selected = focus_game if focus_game in games else (hair_games[0]["value"] if hair_games else "")

    parlay = None
    if selected:
        anchors = [p["leg"] for p in picks if p["leg"]["matchup"] == selected][:MAX_HAIR_ANCHORS]
        parlay = _hair_parlay_for_game(pool, selected, anchors, stake, target)
        if parlay:
            parlay["hair_players"] = {leg["player"] for leg in anchors}
    return picks, top_hair, parlay, hair_games, selected, None
