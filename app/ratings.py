"""Simple opponent-adjusted power ratings (Massey-style) built from scoring margins,
plus normal-distribution win/cover/total probability conversions.

This is a transparent, lightweight statistical model -- not a guarantee of anything.
It gives a data-driven estimate of "fair" probability to compare against the
sportsbook's price, so we can flag where the book's line looks softest.
"""
import math
from collections import defaultdict

HOME_FIELD_ADV = 1.5   # points, modest modern-NFL home edge
MARGIN_SIGMA = 13.0    # stddev of NFL game margins, used for win-prob conversion
# Backtested against 544 real games across the 2024-2025 seasons (walk-forward,
# no lookahead): actual std dev of (final total - predicted total) came out to
# ~13.25, well above the 10.0 originally assumed here -- that made total-line
# probabilities meaningfully overconfident in both directions. Raised to match.
TOTAL_SIGMA = 13.0     # stddev of combined-score totals

# Win-probability recalibration, fit via isotonic regression (Pool Adjacent
# Violators) against 2,634 real games across the 2016-2025 seasons
# (walk-forward, no lookahead -- ratings recomputed from only prior games each
# week). The raw normal-CDF win_probability() is honest up to ~65-70%, but
# gets meaningfully overconfident above that: a raw 93.9% average prediction
# actually won only ~75.3% of the time. Isotonic regression was used
# specifically because it's monotonic by construction -- earlier attempts at
# a logistic (Platt-scaling) correction produced unstable/inverted curves on
# noisier subsets of this data, which isotonic regression can't do. Each pair
# is (raw favored-side probability, calibrated favored-side probability);
# WIN_PROB_CALIBRATION[-1] effectively caps real-world confidence at ~75%
# no matter how lopsided the raw rating gap looks. Applies only to
# win_probability() (Moneyline) -- cover_probability/total_probability
# weren't shown to have this specific issue and are untouched.
WIN_PROB_CALIBRATION = [
    (0.500, 0.500),
    (0.587, 0.569),
    (0.688, 0.626),
    (0.758, 0.697),
    (0.849, 0.706),
    (0.887, 0.722),
    (0.939, 0.753),
    (1.000, 0.753),
]
# The calibrated ceiling above -- exposed so callers comparing this
# probability against a REAL market price (i.e. computing "edge") know where
# our own validated confidence stops. See WIN_PROB_CEILING's docstring-style
# note in value_finder.py for why this matters: the market routinely prices
# real favorites above this ceiling, and backtesting against real historical
# moneylines (2016-2025) showed those favorites win at roughly their market
# rate -- the market's higher confidence there is usually justified, not
# overconfident. Treating "our capped probability is lower than the market's"
# as a real edge on the underdog was a purely mechanical artifact of the cap,
# not genuine insight, and back-tested as a LOSING signal (the more "edge"
# it claimed, the worse it did).
WIN_PROB_CEILING = WIN_PROB_CALIBRATION[-1][1]

# Player-prop recalibration, same isotonic-regression methodology as
# WIN_PROB_CALIBRATION above, fit against real historical player-game
# outcomes (2018-2025, walk-forward, no lookahead, matchup + role-trend
# adjustments applied) rather than assumed. Four tables, not two: a first
# pass pooling all usage levels together found RAW_NORMAL (passing/rushing
# yards, attempts, completions) mildly underconfident near the middle but
# overconfident in the tail, and LOG_SCALE (receiving yards/receptions)
# running the opposite way -- but a follow-up split by usage level (this is
# what a real leg shown in production surfaced: "Sione Vaki Under 7.5 Rush
# Yds" displayed 97.1% raw confidence) found LOW-usage players (thin,
# erratic recent-game samples -- backup RBs, depth targets) are
# meaningfully MORE overconfident in the tail than stable, every-down
# players once you separate them: at the same raw 97.7%, low-usage players'
# real historical accuracy was ~93.0% vs ~95.7% for high-usage ones. Pooling
# them under-corrected the exact case that prompted this fix. Each table
# lists (raw favored-side probability, calibrated favored-side probability);
# the usage split point is USAGE_SPLIT_MEAN in player_props.py.
PROP_CALIBRATION_RAW_LOW_USAGE = [
    (0.5000, 0.5000), (0.5793, 0.6222), (0.6554, 0.6743), (0.7257, 0.7291),
    (0.7881, 0.7720), (0.8413, 0.8131), (0.8849, 0.8468), (0.9192, 0.8769),
    (0.9452, 0.8983), (0.9641, 0.9165), (0.9772, 0.9304), (0.9861, 0.9426),
    (0.9918, 0.9518), (0.9953, 0.9596), (0.9974, 0.9664), (0.9987, 0.9702),
    (1.0000, 0.9702),
]
PROP_CALIBRATION_RAW_HIGH_USAGE = [
    (0.5000, 0.5000), (0.5793, 0.6317), (0.6554, 0.6954), (0.7257, 0.7539),
    (0.7881, 0.8030), (0.8413, 0.8443), (0.8849, 0.8777), (0.9192, 0.9046),
    (0.9452, 0.9290), (0.9641, 0.9438), (0.9772, 0.9568), (0.9861, 0.9681),
    (0.9918, 0.9763), (0.9953, 0.9806), (0.9974, 0.9853), (0.9987, 0.9886),
    (1.0000, 0.9886),
]
PROP_CALIBRATION_LOG_LOW_USAGE = [
    (0.5000, 0.5000), (0.5793, 0.5296), (0.6554, 0.6099), (0.7257, 0.6922),
    (0.7881, 0.7704), (0.8413, 0.8403), (0.8849, 0.8956), (0.9192, 0.9351),
    (0.9452, 0.9609), (0.9641, 0.9758), (0.9772, 0.9852), (0.9861, 0.9914),
    (0.9918, 0.9947), (0.9953, 0.9967), (0.9974, 0.9980), (0.9987, 0.9986),
    (1.0000, 0.9986),
]
PROP_CALIBRATION_LOG_HIGH_USAGE = [
    (0.5000, 0.5000), (0.5793, 0.5590), (0.6554, 0.6519), (0.7257, 0.7447),
    (0.7881, 0.8242), (0.8413, 0.8916), (0.8849, 0.9367), (0.9192, 0.9635),
    (0.9452, 0.9791), (0.9641, 0.9882), (0.9772, 0.9937), (0.9861, 0.9960),
    (0.9918, 0.9980), (0.9953, 0.9989), (0.9974, 0.9993), (0.9987, 0.9994),
    (1.0000, 0.9994),
]


def _interp(x, breakpoints):
    for (x0, y0), (x1, y1) in zip(breakpoints, breakpoints[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return breakpoints[-1][1]


def calibrate_prop_prob(raw_p, calibration_table):
    """Same idea as _calibrate_win_prob below, generalized to any
    (raw, calibrated) breakpoint table: recalibrates the FAVORED side (max
    of p, 1-p) and mirrors it back for the other side, since the
    calibration curve is fit on favored-side accuracy."""
    favored = max(raw_p, 1 - raw_p)
    calibrated_favored = _interp(favored, calibration_table)
    return calibrated_favored if raw_p >= 0.5 else 1 - calibrated_favored


def _calibrate_win_prob(raw_p):
    favored = max(raw_p, 1 - raw_p)
    calibrated_favored = _interp(favored, WIN_PROB_CALIBRATION)
    return calibrated_favored if raw_p >= 0.5 else 1 - calibrated_favored


def compute_power_ratings(weighted_games, iterations=25):
    """weighted_games: list of (home, away, home_score, away_score, weight).
    Returns {team: rating} where rating is point strength relative to league average.
    """
    teams = set()
    for home, away, *_ in weighted_games:
        teams.add(home)
        teams.add(away)
    ratings = {t: 0.0 for t in teams}

    for _ in range(iterations):
        sums = defaultdict(float)
        weights = defaultdict(float)
        for home, away, hs, as_, w in weighted_games:
            home_margin = (hs - as_) - HOME_FIELD_ADV
            away_margin = -home_margin
            sums[home] += w * (home_margin + ratings[away])
            weights[home] += w
            sums[away] += w * (away_margin + ratings[home])
            weights[away] += w
        new_ratings = {}
        for t in teams:
            new_ratings[t] = sums[t] / weights[t] if weights[t] > 0 else 0.0
        mean_r = sum(new_ratings.values()) / len(new_ratings) if new_ratings else 0.0
        ratings = {t: v - mean_r for t, v in new_ratings.items()}

    return ratings


def compute_scoring_averages(weighted_games):
    """Returns {team: {"pf": avg points for, "pa": avg points against}} using weighted average."""
    pf_sum = defaultdict(float)
    pa_sum = defaultdict(float)
    w_sum = defaultdict(float)
    for home, away, hs, as_, w in weighted_games:
        pf_sum[home] += hs * w
        pa_sum[home] += as_ * w
        w_sum[home] += w
        pf_sum[away] += as_ * w
        pa_sum[away] += hs * w
        w_sum[away] += w
    out = {}
    for t in w_sum:
        out[t] = {
            "pf": pf_sum[t] / w_sum[t] if w_sum[t] else 21.0,
            "pa": pa_sum[t] / w_sum[t] if w_sum[t] else 21.0,
        }
    return out


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def win_probability(home_rating, away_rating, home_field=HOME_FIELD_ADV, sigma=MARGIN_SIGMA):
    """P(home team wins outright), recalibrated against real historical
    outcomes -- see WIN_PROB_CALIBRATION above."""
    predicted_margin = (home_rating - away_rating) + home_field
    raw_p = normal_cdf(predicted_margin / sigma)
    return _calibrate_win_prob(raw_p)


def predicted_margin(home_rating, away_rating, home_field=HOME_FIELD_ADV):
    return (home_rating - away_rating) + home_field


def cover_probability(predicted_margin_value, spread_for_home, sigma=MARGIN_SIGMA):
    """P(home team covers a given spread). spread_for_home is the number as posted
    for the home side, e.g. -3.5 means home is favored by 3.5.
    Home covers if actual_margin > -spread_for_home.
    """
    threshold = -spread_for_home
    z = (predicted_margin_value - threshold) / sigma
    return normal_cdf(z)


def total_probability(predicted_total, line, sigma=TOTAL_SIGMA):
    """Returns (p_over, p_under) for a given total line.

    BUG FIX (found in a full model audit): this previously computed
    z = (predicted_total - line) / sigma and returned (1 - normal_cdf(z),
    normal_cdf(z)) as (p_over, p_under) -- backwards. A HIGHER predicted
    total relative to the line should mean a HIGHER P(Over), i.e.
    p_over = normal_cdf(z) directly; the old code assigned that value to
    p_under instead, so every Total leg's model probability was swapped
    with its complement (e.g. a real 68% Over was shown as 32% Over / 68%
    Under). This affected every Total recommendation and any edge
    calculated from it since this function was written."""
    z = (predicted_total - line) / sigma
    p_over = normal_cdf(z)
    return p_over, 1 - p_over
