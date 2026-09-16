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
import json
import time
from supabase import create_client, ClientOptions
from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

TABLE = "app_cache"
# A ~4MB JSONB upsert (nflverse's player-week stats CSV, 1118 rows x 150
# columns) was observed to take well over 2 minutes -- Postgres/PostgREST
# apparently processes a large JSONB blob far slower than the raw byte count
# would suggest, and this app's underlying sources (GitHub release CSVs,
# ESPN) are already fast to fetch fresh (under ~1s even for a 7MB file), so
# there's no good reason to risk stalling a whole page load caching
# something this size. Skip caching anything over this; the caller just gets
# a cache miss and re-fetches from source every time, which is fine here.
MAX_CACHEABLE_BYTES = 1_000_000
_client = None


def _sb():
    global _client
    if _client is None:
        _client = create_client(
            SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
            options=ClientOptions(postgrest_client_timeout=20),
        )
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
        if len(json.dumps(data)) > MAX_CACHEABLE_BYTES:
            return
        _sb().table(TABLE).upsert(
            {"key": key, "data": data, "cached_at": time.time()},
            on_conflict="key",
        ).execute()
    except Exception:
        pass
