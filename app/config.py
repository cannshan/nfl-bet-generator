import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Persistent (never cache-cleared) storage for the prediction track record --
# unlike cache/, this is meant to accumulate indefinitely, not expire.
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PREDICTIONS_DB_PATH = DATA_DIR / "predictions.db"

ODDS_CACHE_TTL_SECONDS = 10 * 60
RATINGS_CACHE_TTL_SECONDS = 6 * 60 * 60

# Default stake/target for parlay & card payout illustrations
DEFAULT_STAKE = 5.0
DEFAULT_TARGET_PAYOUT = 1000.0
