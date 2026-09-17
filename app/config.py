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

# Regions of The Odds API to pull GAME odds from. "us" is the offer book's
# home; "eu" adds Pinnacle (see SHARP_BOOK_KEY). Cost is markets x regions
# credits per refresh (3 markets x 2 regions = 6), trivial on a 20k plan.
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "us,eu")
# The sharp reference book. Pinnacle takes the biggest bets and the
# sharpest money and moves its lines to match, so its devigged price is
# the closest thing to the market's "true" probability available. When it
# has a price for a game-level market, the fair-value reference is
# SHARP_BOOK_WEIGHT on Pinnacle and the rest on the consensus of every
# other book. (Player props stay consensus-only: Pinnacle posts few.)
SHARP_BOOK_KEY = os.environ.get("SHARP_BOOK_KEY", "pinnacle").strip().lower()
SHARP_BOOK_WEIGHT = 0.7

ODDS_CACHE_TTL_SECONDS = 10 * 60
RATINGS_CACHE_TTL_SECONDS = 6 * 60 * 60

# Default stake/target for parlay & card payout illustrations
DEFAULT_STAKE = 5.0
DEFAULT_TARGET_PAYOUT = 1000.0
