"""Reconcile team-name spelling differences between ESPN and The Odds API.
Both generally use full display names (e.g. "Kansas City Chiefs"), but we
normalize defensively in case of mismatches (e.g. "Washington Commanders" vs
"Washington Football Team" in stale data, or minor punctuation differences).
"""

_ALIASES = {
    "washington football team": "washington commanders",
    "oakland raiders": "las vegas raiders",
    "san diego chargers": "los angeles chargers",
    "st. louis rams": "los angeles rams",
}


def normalize(name):
    if not name:
        return ""
    n = name.lower().strip()
    n = n.replace(".", "")
    return _ALIASES.get(n, n)


def build_lookup(rating_teams):
    """Map normalized name -> original key, so odds-api names can find ratings entries."""
    return {normalize(t): t for t in rating_teams}


def match(name, lookup):
    key = normalize(name)
    if key in lookup:
        return lookup[key]
    # fallback: match by last word (nickname), e.g. "Chiefs"
    nickname = key.split()[-1] if key.split() else key
    for norm_name, original in lookup.items():
        if norm_name.split()[-1] == nickname:
            return original
    return None
