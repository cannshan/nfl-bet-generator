-- Run this once in your new Supabase project's SQL Editor
-- (Dashboard -> SQL Editor -> New query -> paste -> Run)

CREATE TABLE IF NOT EXISTS nfl_predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    home_team TEXT,
    away_team TEXT,
    commence_time TEXT,
    market TEXT,
    selection TEXT,
    player TEXT,
    stat_category TEXT,
    side TEXT,
    line DOUBLE PRECISION,
    american_odds INTEGER,
    decimal_odds DOUBLE PRECISION,
    bookmaker TEXT,
    model_prob DOUBLE PRECISION,
    book_fair_prob DOUBLE PRECISION,
    edge DOUBLE PRECISION,
    section TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    actual_value DOUBLE PRECISION,
    settled_at TEXT,
    closing_decimal_odds DOUBLE PRECISION,
    closing_american_odds INTEGER,
    closing_selection TEXT,
    closing_captured_at TEXT,
    UNIQUE (home_team, away_team, commence_time, market, selection)
);

-- Generic key/value cache (replaces the old local-file cache, which won't
-- work on a read-only serverless filesystem like Vercel's).
CREATE TABLE IF NOT EXISTS app_cache (
    key TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    cached_at DOUBLE PRECISION NOT NULL
);

-- ===========================================================================
-- Migration 2026-10-04: tracking accuracy (app/tracking.py)
--
-- Run once in the SQL Editor, same as above. Safe to run more than once
-- (every statement is IF NOT EXISTS / idempotent). Until it has been run the
-- app keeps working on the old schema: writes that name the new columns are
-- retried without them and the ticket log is skipped -- nothing breaks, the
-- new fields just stay empty.
-- ===========================================================================

-- Per-leg fields for later model fitting:
--   history_prob      -- the player's game-log-only probability for the bet
--                        (before the small capped tilt into model_prob);
--                        NULL for game-level bets.
--   closing_fair_prob -- the market's consensus fair probability for OUR
--                        exact bet (our line, our side) at the last capture
--                        before kickoff; NULL when the line moved on a
--                        spread/total (no honest conversion across key
--                        numbers) or no close was captured.
ALTER TABLE nfl_predictions ADD COLUMN IF NOT EXISTS history_prob DOUBLE PRECISION;
ALTER TABLE nfl_predictions ADD COLUMN IF NOT EXISTS closing_fair_prob DOUBLE PRECISION;

-- settle_pending() filters on status, capture_closing_lines() on kickoff.
CREATE INDEX IF NOT EXISTS nfl_predictions_status_idx ON nfl_predictions (status);
CREATE INDEX IF NOT EXISTS nfl_predictions_commence_time_idx ON nfl_predictions (commence_time);

-- Whole parlay tickets as shown (one row per distinct combination of legs;
-- the same ticket recomputed on a later refresh is ignored via ticket_key).
-- legs holds each leg's identity (the same five fields as nfl_predictions'
-- unique key) plus its price/probability on this ticket.
CREATE TABLE IF NOT EXISTS nfl_tickets (
    id BIGSERIAL PRIMARY KEY,
    ticket_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    section TEXT,
    num_legs INTEGER,
    legs JSONB NOT NULL,
    stake DOUBLE PRECISION,
    decimal_odds DOUBLE PRECISION,
    payout DOUBLE PRECISION,
    combined_prob DOUBLE PRECISION,
    breakeven_prob DOUBLE PRECISION,
    ev_pct DOUBLE PRECISION,
    has_same_game_legs BOOLEAN,
    first_commence_time TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    settled_decimal_odds DOUBLE PRECISION,  -- odds actually paid on a hit (void legs removed)
    settled_at TEXT
);
-- A unique INDEX (not an inline constraint) so this also applies to a table
-- created by an earlier draft of this migration; upsert(on_conflict=
-- "ticket_key") needs it.
CREATE UNIQUE INDEX IF NOT EXISTS nfl_tickets_ticket_key_key ON nfl_tickets (ticket_key);
CREATE INDEX IF NOT EXISTS nfl_tickets_status_idx ON nfl_tickets (status);

-- Server-side only (service_role key, which bypasses RLS): enabling RLS with
-- no policies keeps the public anon key from reading or writing it.
ALTER TABLE nfl_tickets ENABLE ROW LEVEL SECURITY;

-- Make PostgREST (the API layer) see the new column/table immediately.
NOTIFY pgrst, 'reload schema';
