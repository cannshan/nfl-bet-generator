import json
import re
import time
from app.config import CACHE_DIR


def _safe_path(key):
    """Sanitize cache keys for use as filenames -- callers build keys from
    things like ISO timestamps (colons) and team names (spaces), which are
    invalid or awkward in Windows filenames."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", key)
    return CACHE_DIR / f"{safe}.json"


def cache_get(key, ttl_seconds):
    path = _safe_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - payload.get("_cached_at", 0) > ttl_seconds:
        return None
    return payload.get("data")


def cache_set(key, data):
    path = _safe_path(key)
    payload = {"_cached_at": time.time(), "data": data}
    path.write_text(json.dumps(payload))
