# NFL Bet Generator

A local web dashboard that pulls **live NFL odds** and **real season stats**,
runs a transparent statistical model to estimate "fair" win/cover/total
probabilities, and builds two kinds of parlay aimed at a target payout
(default: $5 → $1000):

- **Same Game Parlays** — one per matchup that has player props live, mostly
  built from that game's props. Legs here are correlated (a team winning big
  moves its players' stats with it), so the payout/hit-chance shown is
  illustrative, not a real sportsbook quote — build the actual combo in your
  sportsbook's own SGP tool for the real price.
- **Statistically Best Bets** — no restriction on which games a leg comes
  from; it just chases the highest real hit chance at the target payout using
  every signal the model has. When that happens to include more than one leg
  from the same game, the combo is flagged (`has_same_game_legs`) and its
  probability carries the same illustrative caveat as a Same Game Parlay —
  never silently presented as clean independent math when it isn't.

A **Track Record** tab tracks every suggestion automatically and checks it
against the real result once the game is played — see below.

## How it works

- **Odds data**: [The Odds API](https://the-odds-api.com) — live NFL lines
  (moneyline, spread, totals, and player props) from multiple sportsbooks.
  Free tier is 500 requests/month; if that key ever gets rejected or the
  quota runs out, the app automatically falls back to a second free
  provider, [sportsgameodds.com](https://sportsgameodds.com) (optional —
  set `SPORTSGAMEODDS_API_KEY` to enable it), for moneyline/spread/total
  odds only. The Odds API is always tried first; the fallback only kicks in
  on an actual failure.
- **Team scores**: [nflverse](https://github.com/nflverse/nflverse-data)'s
  free `games.csv` release — every NFL game's final score since 1999.
  (Originally used ESPN's scoreboard API for this, but that endpoint turned
  out to have a much stricter rate limit than ESPN's other endpoints and
  got this app's own IP blocked during testing — nflverse's GitHub-hosted
  CSV doesn't have that problem.)
- **Efficiency & player stats**: nflverse's free, no-key CSV data releases —
  team-level EPA (Expected Points Added) per game, and per-player weekly
  stat lines (passing/rushing/receiving) used to project player props.
- **Team model**: an opponent-adjusted power rating (similar in spirit to a
  Massey rating) computed from real scoring margins, blended with nflverse's
  EPA-based team efficiency ratings, converted to win/cover/total
  probabilities via a normal-distribution margin model.
- **Player prop model**: each player's own weighted recent-game average and
  standard deviation for the relevant stat (now including pass attempts,
  completions, and pass TDs, not just yardage — see "which prop type is more
  predictable" below), adjusted for several things at once:
  - **opponent matchup** — how generous/stingy the opponent's run or pass
    defense has been vs. league average
  - **recency** — within the current season, each week further back counts
    ~10% less, so a real recent trend outweighs an early-season game
  - **role trend** — a receiver's target share over their last 3 games vs.
    their own season baseline, a direct, data-driven "bigger/smaller role"
    signal independent of what any injury report says
  - **weather** — see below
  - **the opponent defense's own fresh injuries** — a real (not long-standing)
    Out/Doubtful starter-level defender missing this week gets a small boost
    to the offense facing them (capped, and only counts players with genuine
    tackle/sack/INT involvement, so a healthy scratch doesn't trigger it)

  Players with too little game history are skipped rather than guessed at.
- **Weather** ([Open-Meteo](https://open-meteo.com), free, no key): for
  outdoor stadiums, wind and precipitation at kickoff time adjust passing-game
  projections down (high wind hurts accuracy/distance, rain/snow hurts grip
  and footing) and nudge rushing-game projections up slightly (teams lean
  run more in bad weather). Domes and retractable roofs are treated as
  climate-controlled and skipped — a static 32-stadium lookup table decides
  which is which. The note shown states the actual effect on that leg's
  category directly (e.g. "Windy (15 mph wind): -7% to this projection"),
  not just the raw weather numbers.
- **Which prop type is actually more predictable**: computed fresh each
  refresh from real data — the average player's own week-to-week coefficient
  of variation (std/mean) per stat category. In practice, pass attempts and
  completions are consistently the most predictable (CV ~0.4), followed by
  pass TDs and receptions (~0.5), then passing yards (~0.56), with rushing
  yards and receiving yards the least predictable (~1.0 — their week-to-week
  swing is roughly as large as the average itself). This number quietly
  discounts a prop's edge when ranking which legs make a card, so two props
  with similar edge but different volatility don't get treated as equally
  good bets.
- **Edge**: for each bet, we de-vig the sportsbook's two-sided price to get
  its true implied probability, then compare to the model's probability. A
  positive edge means the model thinks the bet is priced better than the
  market says.
- **Live injury reports**: ESPN's public injuries endpoint (one call for the
  whole league, including beat-reporter comments on practice participation —
  the "sore at practice but expected to play" kind of nuance) is checked on
  every refresh. A player listed Out/Doubtful/Injured Reserve is dropped from
  the prop pool entirely; a Questionable player stays in but with a
  conservative haircut on their projected stat line, and the note is shown
  under their prop. Separately, each team's presumptive starting QB (most
  recent pass attempts at the position) is checked the same way — if they're
  out, the team's rating takes a real point penalty before any probabilities
  are computed, not just a cosmetic flag. This is the single biggest lever
  available for catching an injury that would otherwise blindside a pick.
- **Injury history context, without fabricating a discount for it**: even a
  player with no current game-status designation ("Active") can have an
  injury-history comment attached (e.g. "does not have an injury designation"
  after coming back from a significant injury) — that gets surfaced as an
  informational note (ℹ️) rather than silently discarded, since the model has
  no statistical basis to discount an officially-cleared player but the user
  might still want the context. Separately, any player with zero games played
  so far this season gets an explicit "based entirely on last season's stats"
  flag, since that's exactly the situation where an offseason injury, role
  change, or team change wouldn't show up in the numbers yet — it's a real,
  temporary blind spot that resolves itself as soon as the player has current-
  season games to draw on.
- **A hurt/backup QB drags down his own receivers, not just his own stats**:
  when a team's real starting QB (per the real depth chart, see above) is
  out or doubtful, every pass-catcher on that team gets a modest discount on
  their receiving props — a backup center-fielding the passing game tends to
  produce worse targets and looks for everyone, not just worse numbers for
  the QB himself (who's already excluded from the pool if he's the one hurt).
- **Offensive injury report**: a collapsed-by-default table — no rows render
  at all until you click to expand — of QB/RB/WR/TE players who are actually
  Out, Doubtful, Questionable, or on Injured Reserve. "Active" players are
  excluded, since most of those entries are just routine post-game recap
  notes, not injury news. Ranked Out/Doubtful/Questionable first (this
  week's live decisions) then Injured Reserve (older, already-known
  absences). Expanding shows the full list in a scrollable, sortable
  (click any column header) box with a sticky header row, rather than either
  truncating it or pushing the rest of the page down. Replaces what used to
  be a fixed "starting QB injuries" banner at the top of the page.
- **Live expert insight (optional, costs real money)**: if `ANTHROPIC_API_KEY`
  is set, one Claude API call per prop-rich game (not per player) uses Claude's
  web search tool to find current betting-expert commentary, projections, and
  buzz on every candidate player in that game — before the card is finalized,
  not after. A bullish/bearish read nudges that leg's probability by a small,
  fixed amount (±3pp), which can genuinely change which legs make the card,
  not just decorate whatever the stats model already picked. This is the one
  part of the app with a real per-use cost (roughly a few cents per game per
  refresh, cached for 4 hours) — uncheck "Include expert insight" to skip it,
  or just don't set the API key. Token usage is printed to the terminal
  running `run.py` for visibility.
- **Track Record (automatic, no manual bet logging)**: every leg actually
  shown on the page — same-game and best-odds sections both — gets recorded
  in a Supabase (Postgres) table (`nfl_predictions`) the moment it's
  suggested — see `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` below and
  `supabase_setup.sql` for the one-time table setup. On every later page
  load, any pending prediction whose game started more than 4 hours ago gets
  checked against the real result: final score from ESPN for game-level
  bets, final box-score stat from nflverse for player props. Nothing to do
  on your end — it just accumulates. The **Track Record** tab shows total
  settled/pending, overall hit rate, straight-bet ROI, a calibration table
  (does a "70% pick" actually hit ~70% of the time, broken out by
  probability range), closing line value (CLV — did the suggested price
  beat the market's own closing number, a faster skill signal than
  win/loss), and the most recently settled bets. Explicitly does NOT feed
  results back into the model yet — with only a handful of settled bets at
  first, auto-recalibration would just be overfitting to noise. The report
  itself says so plainly (a visible "too small a sample" warning) until at
  least 20 bets have settled (200+ for CLV).
- **Current rosters, not just historical stats**: ESPN's team roster endpoints
  (32 teams, cached 12 hours) are the source of truth for "which team is this
  player on right now." This matters because nflverse's weekly-stats data only
  reflects a player's team from games actually played — someone traded or
  signed this offseason who hasn't played yet still shows their OLD team in
  that data. Caught in practice: a QB who changed teams was still being
  attributed to his old team for the starting-QB injury check, and two other
  players' real (correct) props were being wrongly excluded as "mislabeled"
  because the old verification only had stale data to check against.
- **Real depth charts for "who's the starting QB," not a historical-usage
  guess**: ESPN's per-team depth-chart endpoint gives the actual current QB1
  (and QB2) directly. This replaced an earlier "whoever has the most recent
  pass attempts" proxy, which broke whenever a starter changes for any reason
  other than a trade — caught in practice twice: a healthy 3rd-string
  emergency QB was flagged as "out" (he'd started more games historically,
  but wasn't the real QB1 anymore) while, separately, another team's actual
  injured starter went completely uncaught because his healthy backup had
  more career attempts. Same lesson as the roster fix: prefer the data source
  that reflects *now*, not the one that reflects *history*. If the backup
  (QB2) is ALSO out when the starter is out, an additional rating penalty
  stacks on top — losing your top two QBs is much worse than losing just one.
- **Conservative discount for players coming back from a significant injury**:
  even once a player is fully cleared ("Active"), ESPN's longer injury comment
  (not just the terse short one — seen in practice, the short comment for one
  player omitted the ACL tear that the long comment named explicitly) is
  scanned for real injury-history language (ACL, Achilles, hamstring, torn,
  surgery, etc.). When found, rushing-category props get a real discount
  (workload/mobility is the actual risk for a lower-body injury) while
  passing-category props get a much lighter one (arm health usually isn't
  affected the same way) — this is exactly the "don't trust his workload yet"
  instinct made concrete, applied to the number itself rather than left as
  a comment nobody has to act on.
- **Same Game Parlays**: for each matchup with live props, searches
  combinations of that game's own legs — player props dominate the candidate
  pool, with the moneyline added only when it's a strong pick on its own —
  for the payout landing closest to your target, ranked by the highest hit
  probability achievable at that payout level. A team's own spread is
  excluded when its moneyline is already in the combo, since covering a
  spread and winning outright are nearly the same bet. Games without live
  props yet fall back to a smaller spread/total combo, shown separately.
- **Statistically Best Bets**: the same target-payout search, unrestricted —
  run on the whole week's pool with no per-game limit, ranked by edge
  discounted for category reliability. Two dedupe passes keep this honest:
  `dedupe_best_per_bet` collapses every book's different line for the same
  player+stat down to the single best one (otherwise the search could stack
  "Over 39.5," "Over 44.5," and "Over 49.5" rushing yards for the same player
  as if they were three independent bets, which they obviously aren't), and
  `_drop_redundant_favorite_bets` keeps only the better of a team's own
  spread vs. their moneyline when both would otherwise appear together. When
  the result still includes more than one leg from the same game, it's
  flagged rather than presented as if the math were clean.

## Setup

1. Get a free API key at [the-odds-api.com](https://the-odds-api.com) (free
   tier: 500 requests/month — plenty for personal use with caching).
2. Create a free project at [supabase.com](https://supabase.com) (used for
   the persistent prediction-tracking table, not the live odds/stats). In
   its SQL Editor, run `supabase_setup.sql` once. Then, from that project's
   Settings -> API page, grab the Project URL and `service_role` key.
3. Copy `.env.example` to `.env` and fill in:
   ```
   ODDS_API_KEY=your_key_here
   SUPABASE_URL=your_project_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```
   Optionally add `ANTHROPIC_API_KEY=your_key_here` too, to enable live expert
   insight (see below) — the app works fine without it, just without that
   one section. Also optionally add `SPORTSGAMEODDS_API_KEY=your_key_here`
   (free at [sportsgameodds.com](https://sportsgameodds.com), no credit
   card) so the app can fall back automatically if The Odds API's key ever
   gets rejected or its quota runs out.
4. Install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Run it:
   ```bash
   python run.py
   ```
5. Open http://localhost:5057

Use the stake/target fields on the page to change the parlay target (defaults
to $5 stake, $1000 payout).

## Notes / limitations

- **Injury comments are simplified for readability**: ESPN's raw text reads
  like a news blurb, complete with reporter attribution ("...Chris Tomasson
  of The Denver Gazette reports."). That attribution is stripped and the
  result is length-capped so notes stay skimmable. For the "coming back from
  injury" note specifically, the relevant sentence is extracted from the full
  comment (not just the first ~140 characters), since the actual detail (e.g.
  "tore his ACL") is often buried well past where a flat truncation would cut.

- Free-tier odds requests are cached for 10 minutes to conserve your monthly
  quota. Player props only cost API credits for games sportsbooks have
  actually posted props for (usually within a day or two of kickoff), so a
  full-week refresh early in the week is cheap.
- Early in the season, both the team model and player projections blend in
  last season's data (at lower weight) since there isn't enough current-season
  data yet.
- Uncheck "Include player props" on the page to skip that call entirely
  (e.g. to conserve quota, or if you only care about game lines).
- Player props are matched against nflverse's stats using the player's exact
  display name; a handful of edge cases (name variants, mid-season trades,
  rookies with no game history yet) will be silently skipped rather than
  produce a bad projection.
- **The role-trend, weather, and defensive-injury adjustments are all rough,
  transparent heuristics with clamped magnitudes** (e.g. role trend is capped
  at ±30%, defensive-injury boost at +15% total), not fitted models — they're
  there to catch real, directionally-correct signal without pretending to
  precision the data doesn't support. Same goes for the ±3pp expert-sentiment
  nudge: a fixed amount, not a calibrated probability shift.
- **What injury data can't do**: player-vs-player coverage matchups (which
  specific cornerback shadows which receiver, and how that corner performs
  against that receiver's profile) aren't available from any free source —
  that's proprietary data (PFF and similar) this tool doesn't have and won't
  fake. The starting-QB-out penalty (6 points) and Questionable-player
  discount (15%) are also rough, widely-cited heuristics, not precisely
  fitted numbers — real injury impact varies a lot by backup quality and
  specific injury.
- **Expert insight is LLM-synthesized from live search results, not a
  structured stats feed like the rest of the app — treat it as a second
  opinion to skim, not gospel.** It's grounded in real search results (you can
  verify by checking the players/facts it mentions), but summarization can
  occasionally misattribute a stat or blend two players' context together.
  The rest of this app's numbers come from deterministic math over structured
  data; this one section is the exception, and the UI/README say so on
  purpose rather than presenting it with false uniformity.
- **Same-game legs are correlated, and the tool doesn't pretend otherwise.**
  A team winning big tends to move its players' stat lines with it, so
  multiplying independent probabilities together (what the payout/hit-chance
  shown does) overstates precision for a same-game combo. It's a reference
  number; the real, correlation-adjusted price only exists in your
  sportsbook's own Same Game Parlay builder once you enter the same legs
  there. A payout close to your target also inherently means a lower hit
  chance than a smaller payout would — the tool always shows that real
  number rather than a rosier one.

## Responsible gambling

Only bet what you can afford to lose. If gambling stops being fun, the
National Problem Gambling Helpline is 1-800-522-4700.
