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
    return [
        item for item in data
        if isinstance(item, dict) and "player" in item and "note" in item
    ]


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
        "Respond with ONLY a JSON array, no other text before or after, no markdown code fences, in this "
        'exact format: [{"player": "Full Name", "sentiment": "bullish|bearish|neutral|no_data", '
        '"note": "one short sentence"}]'
    )

    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1200,
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

    results = _parse_json_array(_extract_text(data.get("content", [])))
    cache_set(cache_key, results)
    return results
