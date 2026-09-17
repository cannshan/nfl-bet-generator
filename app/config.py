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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Free fallback odds source, used only when The Odds API's key is rejected or
# its quota is exhausted -- see sportsgameodds_client.py.
SPORTSGAMEODDS_API_KEY = os.environ.get("SPORTSGAMEODDS_API_KEY", "")

# The ONE sportsbook every leg is offered at. The fair-value reference for
# a bet still comes from the consensus across every book in the feed (that
# is what tells you when this book's price is off), but the line, price and
# payout shown are only ever this book's -- so the ticket is one you can
# actually place, not a best-price patchwork across a dozen apps. The Odds
# API bookmaker key ("draftkings", "fanduel", "betmgm", ...); leave empty
# to shop every book instead.
BOOKMAKER_KEY = os.environ.get("BOOKMAKER_KEY", "draftkings").strip().lower()

ODDS_CACHE_TTL_SECONDS = 10 * 60
RATINGS_CACHE_TTL_SECONDS = 6 * 60 * 60

# Default stake/target for parlay & card payout illustrations
DEFAULT_STAKE = 5.0
DEFAULT_TARGET_PAYOUT = 1000.0
