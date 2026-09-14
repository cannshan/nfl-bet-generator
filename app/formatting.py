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
