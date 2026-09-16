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
