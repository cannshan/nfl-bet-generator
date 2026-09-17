"""Live expert-opinion research via the Claude API's web search tool.

This is the one piece of this app that costs real money per use (Anthropic
API tokens) and requires its own API key -- everything else in this app is
free. It's called once per same-game card (not once per player) to keep
usage bounded, and cached for a few hours since expert takes don't change
minute to minute.

We ask for structured JSON back, but the model is still free-text underneath
(it sometimes wraps the JSON in markdown fences, adds a sentence before it,
or includes <cite> tags in the prose) so parsing here is deliberately
defensive -- any failure just means no insight is shown, never a crashed page.
"""
import json
import re
import requests
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_API_BASE, ANTHROPIC_MODEL
from app.cache_utils import cache_get, cache_set, live_fetch_allowed

TIMEOUT = 90
CACHE_TTL_SECONDS = 4 * 60 * 60
MAX_SEARCHES = 4

_CITE_TAG_RE = re.compile(r"</?cite[^>]*>")

# How much a read is worth, by what it rests on. Injury/role news from a
# team source or beat reporter is the one kind of public information the
# market can lag on for a while; a published projection is a real second
# opinion; matchup analysis is soft; "he had a big game last week" is
# already in the line and worth almost nothing as a reason to bet.
BASIS_WEIGHT = {"injury_or_role": 1.0, "projection": 0.7, "matchup": 0.4, "narrative": 0.15, "none": 0.0}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


def credibility(info):
    """0..1 score for one player's read: basis x confidence, with a small
    boost for multiple named sources. Reads cached from before these
    fields existed have no basis/confidence and score as a single medium
    source with no basis stated (0.3)."""
    if not info or info.get("sentiment") in (None, "neutral", "no_data"):
        return 0.0
    if "basis" not in info or "confidence" not in info:
        return 0.3
    score = BASIS_WEIGHT.get(info.get("basis"), 0.0) * CONFIDENCE_WEIGHT.get(info.get("confidence"), 0.3)
    if len(info.get("sources") or []) >= 2:
        score = min(1.0, score * 1.25)
    return round(score, 3)


def is_configured():
    return bool(ANTHROPIC_API_KEY)


def _extract_text(content_blocks):
    return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")


def _parse_json_array(text):
    text = _CITE_TAG_RE.sub("", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned = []
    for item in data:
        if not (isinstance(item, dict) and "player" in item and "note" in item):
            continue
        sources = item.get("sources")
        item["sources"] = [str(s) for s in sources if s] if isinstance(sources, list) else []
        item["basis"] = item.get("basis") if item.get("basis") in BASIS_WEIGHT else "none"
        item["confidence"] = item.get("confidence") if item.get("confidence") in CONFIDENCE_WEIGHT else "low"
        cleaned.append(item)
    return cleaned


def get_expert_insight(matchup, players, commence_time=None):
    """Returns [{"player", "sentiment", "note"}, ...] for the given players
    ahead of `matchup`, or [] if unconfigured/unavailable/unparseable."""
    if not is_configured() or not players:
        return []

    cache_key = "expert_insight_" + re.sub(r"[^a-z0-9]+", "_", (matchup + "_".join(sorted(players))).lower())
    cached = cache_get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    if not live_fetch_allowed():
        return []

    player_list = ", ".join(players)
    when = f" on {commence_time}" if commence_time else ""
    prompt = (
        f"Search the web for the latest NFL betting-relevant news, expert commentary, or projections "
        f"on these players ahead of {matchup}{when}: {player_list}.\n\n"
        "For each player, note in one short factual sentence whether current expert/analyst sentiment "
        "leans bullish, bearish, or neutral on their prop bets this game, and why (recent role, "
        "injury/practice status, matchup notes, etc.). If you find nothing specific for a player, say so "
        "rather than guessing.\n\n"
        "Also grade what you found, honestly:\n"
        '- "basis": what the read rests on -- "injury_or_role" (an injury, practice/snap report, depth-chart '
        'or role change from a beat reporter or team source), "projection" (a published numeric projection '
        'or model output for this game), "matchup" (analyst matchup/coverage analysis), or "narrative" '
        '(a recent big/bad game, a streak, general hype -- things the betting market has already priced).\n'
        '- "confidence": "high" only when multiple independent, specific sources agree; "medium" when one '
        'specific source; "low" when vague, dated, or conflicting.\n'
        '- "sources": the outlets/people the read came from, e.g. ["ESPN", "SportsGrid", "Ian Rapoport"].\n\n'
        "Respond with ONLY a JSON array, no other text before or after, no markdown code fences, in this "
        'exact format: [{"player": "Full Name", "sentiment": "bullish|bearish|neutral|no_data", '
        '"basis": "injury_or_role|projection|matchup|narrative|none", "confidence": "high|medium|low", '
        '"sources": ["..."], "note": "one short sentence"}]'
    )

    body = {
        "model": ANTHROPIC_MODEL,
        # ~20 players x ~80 tokens each with the grading fields; the old
        # 1200 cap truncated the JSON mid-array and the whole read was lost.
        "max_tokens": 4000,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCHES}],
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(
            ANTHROPIC_API_BASE,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    usage = data.get("usage", {})
    print(
        f"[expert_insight] {matchup}: {usage.get('input_tokens', '?')} in / "
        f"{usage.get('output_tokens', '?')} out tokens, "
        f"{usage.get('server_tool_use', {}).get('web_search_requests', '?')} searches"
    )

    text = _extract_text(data.get("content", []))
    results = _parse_json_array(text)
    if not results:
        # A failed/truncated parse is not "no news" -- don't cache it, or
        # the next 4 hours of refreshes would silently run without research.
        print(f"[expert_insight] {matchup}: could not parse a JSON array from the response "
              f"(stop_reason={data.get('stop_reason')}, {len(text)} chars)")
        return []
    cache_set(cache_key, results)
    return results
