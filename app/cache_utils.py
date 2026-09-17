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
import threading
import time
from supabase import create_client, ClientOptions
from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

TABLE = "app_cache"
# A ~4MB JSONB upsert of nflverse's player-week stats CSV (1118 rows x its
# full 150 columns) was observed to hang well past a minute. The real driver
# turned out to be COLUMN WIDTH, not raw byte size: nflverse_client now
# trims every wide release to the ~10-20 columns this app actually reads
# before ever caching it (see PLAYER_WEEK_FIELDS etc.) -- a since-measured
# trimmed payload nearly double the original's byte size (7MB, 18540 narrow
# rows for a full season) cached in ~19s, not a hang. This cap is a second
# line of defense against some future payload that's wide OR just enormous;
# raised from the original 1MB (which was blocking legitimate, safe-to-cache
# payloads like the trimmed player/team/games releases) now that the actual
# cause is understood and addressed at the source. Passive-mode page views
# never hit this path at all (they only read already-cached data, which is
# fast regardless of size -- ~1s even for the 7MB example above); this only
# matters for the write that happens during an explicit refresh.
MAX_CACHEABLE_BYTES = 10_000_000
# One Supabase client PER THREAD, not one shared by all of them. This
# module is called from several thread pools at once (event props, weather,
# expert insight, model + odds), and a single shared client -- an httpx
# session underneath -- produced "[WinError 10035] A non-blocking socket
# operation could not be completed immediately" read errors under that
# load on Windows. cache_get swallowed those as "not cached", so the page
# would silently build from a PARTIAL pool (measured: the same passive
# load returned 360, 206, 598 and 0 legs on four consecutive runs) or
# show nothing at all. Thread-local clients remove the sharing entirely.
_local = threading.local()
# A read that still fails gets one immediate retry before being treated as
# a miss -- a transient socket error must never look like "no data".
CACHE_READ_ATTEMPTS = 2

# Two modes, controlling whether an external API call is allowed at all:
#
# "passive" (the default, used for a plain page view): cache_get ignores
# ttl_seconds entirely and returns cached data at ANY age, and
# live_fetch_allowed() is False -- every fetch function's cache-miss branch
# must skip the live call and return an empty/None default instead. A plain
# page load must never make an outbound API call, no matter how stale the
# cache is.
#
# "active" (used only inside an explicit refresh route, e.g. "Refresh Bets"
# or "Refresh Injury Report"): cache_get behaves normally (respects
# ttl_seconds, so still-fresh data isn't needlessly re-fetched even during a
# refresh), and live_fetch_allowed() is True, so a genuinely stale/missing
# cache entry triggers a real fetch.
_MODE = "passive"

# In-process memoization for the lifetime of ONE request (cleared by
# app.py's before_request hook) -- purely to avoid asking Supabase for the
# SAME key twice within a single page load (e.g. team-week stats are read
# by both compute_allowed_yardage and compute_net_epa_ratings). Not a
# substitute for the real cache: this dict starts empty on every request
# (and on every cold serverless instance), it just stops one request from
# repeating a lookup it already made a moment ago. Guarded by a lock since
# player_props.py now fetches multiple events concurrently.
_request_cache = {}
_request_cache_lock = threading.Lock()


def reset_request_cache():
    global _request_cache
    with _request_cache_lock:
        _request_cache = {}


def set_mode(mode):
    assert mode in ("passive", "active")
    global _MODE
    _MODE = mode


def live_fetch_allowed():
    return _MODE == "active"


def _sb():
    client = getattr(_local, "client", None)
    if client is None:
        client = create_client(
            SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
            options=ClientOptions(postgrest_client_timeout=45),
        )
        _local.client = client
    return client


def _select_row(key):
    """The cache row for `key`, retried once on a transient failure. Raises
    only if every attempt fails."""
    last_error = None
    for attempt in range(CACHE_READ_ATTEMPTS):
        try:
            rows = _sb().table(TABLE).select("data,cached_at").eq("key", key).limit(1).execute().data
            return rows[0] if rows else None
        except Exception as e:  # noqa: BLE001 - any transport/API failure
            last_error = e
            _local.client = None  # drop the (possibly wedged) connection before retrying
    raise last_error


def cache_get(key, ttl_seconds):
    with _request_cache_lock:
        if key in _request_cache:
            return _request_cache[key][0]
    try:
        row = _select_row(key)
    except Exception as e:  # noqa: BLE001
        print(f"[cache_utils] read failed for {key[:60]}: {type(e).__name__}: {e}")
        return None
    if row is None:
        return None
    if _MODE == "active" and time.time() - row["cached_at"] > ttl_seconds:
        return None
    with _request_cache_lock:
        _request_cache[key] = (row["data"], row["cached_at"])
    return row["data"]


def get_cache_age(key):
    """Seconds since `key` was cached, from whatever this SAME request already
    looked up via cache_get -- not an extra Supabase round trip. Returns None
    if cache_get hasn't been called for this key yet this request (e.g. a
    cache miss, or simply not looked up)."""
    with _request_cache_lock:
        entry = _request_cache.get(key)
    if not entry or entry[1] is None:
        return None
    return time.time() - entry[1]


def cache_set(key, data):
    now = time.time()
    with _request_cache_lock:
        _request_cache[key] = (data, now)
    try:
        if len(json.dumps(data)) > MAX_CACHEABLE_BYTES:
            return
        _sb().table(TABLE).upsert(
            {"key": key, "data": data, "cached_at": now},
            on_conflict="key",
        ).execute()
    except Exception:
        pass


def get_raw(key):
    """Reads a key with no TTL/mode restriction at all -- for small bits of
    state (e.g. odds-API quota usage) that should always show whatever was
    last recorded, even on a plain passive page view, since reading it isn't
    a live fetch itself."""
    try:
        row = _select_row(key)
    except Exception:  # noqa: BLE001
        return None
    return row["data"] if row else None


def set_raw(key, data):
    """Writes a key with no size/mode gating -- for small pieces of state
    that should always be recorded when available, notably as a side effect
    of a call that already happened (not a new live fetch of its own)."""
    with _request_cache_lock:
        _request_cache[key] = data
    try:
        _sb().table(TABLE).upsert(
            {"key": key, "data": data, "cached_at": time.time()},
            on_conflict="key",
        ).execute()
    except Exception:
        pass
