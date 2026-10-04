from datetime import datetime
import pytz

EASTERN = pytz.timezone("America/New_York")


def format_kickoff_et(commence_time_iso):
    """The Odds API gives commence_time as UTC ISO 8601 (e.g. '2026-09-15T00:15:00Z').
    Returns something like 'Mon 9/14 8:15 PM ET', or None if not parseable."""
    if not commence_time_iso:
        return None
    try:
        dt_utc = datetime.strptime(commence_time_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
    except ValueError:
        return None
    dt_et = dt_utc.astimezone(EASTERN)
    hour12 = dt_et.hour % 12 or 12
    ampm = "AM" if dt_et.hour < 12 else "PM"
    return f"{dt_et.strftime('%a')} {dt_et.month}/{dt_et.day} {hour12}:{dt_et.strftime('%M')} {ampm} ET"


def has_kicked_off(commence_time_iso, now=None):
    """True once a game's kickoff (UTC ISO from the odds feed) has passed.
    The odds feed keeps a game listed while it's being played, with in-play
    prices -- this app is pregame only, so a started game's prices must
    never become legs. Unparseable -> treated as upcoming (never hide a
    game by accident). The feed sometimes rewrites commence_time to the
    actual kickoff with seconds (e.g. 13:32:19Z), which parses the same."""
    if not commence_time_iso:
        return False
    try:
        kickoff = datetime.strptime(commence_time_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
    except ValueError:
        return False
    return kickoff <= (now or datetime.now(pytz.utc))


def format_odds_age(age_seconds):
    """How long ago this leg's odds were actually fetched, e.g. 'just now',
    '12m ago', '3h ago'. A plain page view can serve odds of very different
    ages for different games (each game's props refresh independently), so
    this is shown per-card/per-leg rather than assumed to always be fresh.
    Returns None if the age isn't known (e.g. never fetched yet)."""
    if age_seconds is None:
        return None
    age_seconds = max(0, age_seconds)
    if age_seconds < 60:
        return "just now"
    minutes = round(age_seconds / 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = round(age_seconds / 3600)
    if hours < 24:
        return f"{hours}h ago"
    days = round(age_seconds / 86400)
    return f"{days}d ago"
