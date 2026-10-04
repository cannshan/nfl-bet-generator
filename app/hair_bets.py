"""Leah's Hairplays: player props on guys with notable hair.

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
import re
from app import value_finder, game_cards, parlay_builder, roster_client, cache_utils, nflverse_client, formatting

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
    # Headwear department
    "Keon Coleman": ("🟡", "Bright yellow durag in his official headshot", 9),
    "Michael Penix Jr.": ("🏴‍☠️", "Durag in his official headshot", 6),
    # Honorable mention
    "Samaje Perine": ("🧔", "Zero hair on top, all of it moved to the beard", 7),
}

TOP_HAIR_COUNT = 3

# "Plays on a regular basis": a prop line this week AND at least this many
# games played this season.
MIN_GAMES_THIS_SEASON = 3

PARLAY_LEGS = 4


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
    return {_norm(name): sum(1 for _r, w in rows if w >= 1.0) for name, rows in index.items()}


def get_leahs_bets(stake):
    """Returns (picks, top_hair, parlay, error): every eligible hair pick
    (likeliest first), the TOP_HAIR_COUNT craziest of them, and a parlay of
    the likeliest picks. Reads only what's already cached --
    callers set cache_utils mode to passive, so this never spends quota."""
    try:
        _value_bets, pool, meta = value_finder.get_value_bets_and_pool()
    except Exception as e:
        return [], [], None, f"Couldn't read cached odds: {e}"
    if meta.get("error"):
        return [], [], None, meta["error"]
    game_cards.apply_expert_insight(pool, include_insight=True)
    games_played = _games_played_this_season(meta.get("season"))

    best = {}
    for leg in pool:
        if not leg.get("player") or leg.get("contradicts_expert"):
            continue
        if formatting.has_kicked_off(leg.get("commence_time")):
            continue  # in-play price -- pregame bets only
        hair = _HAIR_BY_NORM.get(_norm(leg["player"]))
        if not hair or games_played.get(_norm(leg["player"]), 0) < MIN_GAMES_THIS_SEASON:
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
    # Craziest hair first; ties go to the likelier bet.
    top_hair = sorted(picks, key=lambda p: (-p["craziness"], -p["leg"]["model_prob"]))[:TOP_HAIR_COUNT]

    parlay = None
    parlay_legs = [p["leg"] for p in picks[:PARLAY_LEGS]]
    if len(parlay_legs) >= 2:
        parlay = parlay_builder.build_result(parlay_legs, stake)
    return picks, top_hair, parlay, None
