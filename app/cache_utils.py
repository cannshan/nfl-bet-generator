"""Small key/value cache with a TTL, backed by Supabase (table `app_cache`).

Used to be a local JSON-file cache, which worked fine for self-hosting but
would break outright on Vercel: everything except /tmp is a read-only
filesystem there, so a plain file write would throw on nearly every request
(this module is called after almost every external API fetch: odds, ESPN,
nflverse, injuries, rosters, weather, expert insight). Moving it to the same
Supabase project already used for prediction tracking fixes that and also
means the cache actually persists BETWEEN invocations, which a serverless
function's own local disk never would anyway.
"""
import time
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

TABLE = "app_cache"
_client = None


def _sb():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def cache_get(key, ttl_seconds):
    try:
        rows = _sb().table(TABLE).select("data,cached_at").eq("key", key).limit(1).execute().data
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    if time.time() - row["cached_at"] > ttl_seconds:
        return None
    return row["data"]


def cache_set(key, data):
    try:
        _sb().table(TABLE).upsert(
            {"key": key, "data": data, "cached_at": time.time()},
            on_conflict="key",
        ).execute()
    except Exception:
        pass
