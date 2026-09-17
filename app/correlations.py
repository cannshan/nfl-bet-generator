"""Turns per-leg probabilities into an honest JOINT probability for a parlay
whose legs share a game, using correlations measured from real outcomes.

Multiplying leg probabilities together assumes independence. Same-game
legs aren't independent: a QB's passing yards and his receivers' yards
move together, a team's rushing yards move with its margin of victory, a
QB's passing TDs move with the game total, and so on. The size of each of
those relationships was measured -- not guessed -- by
scripts/backtest_correlations.py over every regular-season player-game
2018-2025 against real closing lines; the results live in
correlation_table.json as Pearson correlations of standardized residuals.

Those feed a Gaussian copula: each leg is modeled as "a latent standard
normal falls below z(p)", latents correlated per the table, and the
probability all legs hit is the multivariate-normal orthant probability.
Two evaluations are provided:

  joint_probability      -- fast pairwise approximation (product of legs
                            times a lift per correlated pair), used to rank
                            thousands of candidate combos in the search;
  joint_probability_mc   -- seeded Monte Carlo over the full copula, used
                            for the final displayed number, where the
                            pairwise product would overcount overlapping
                            correlations in a large same-game combo.

Legs from different games are independent. Two legs on the SAME player are
never meant to reach here (parlay_builder keeps one leg per player) but
are treated as near-duplicates (rho 0.9) if they do, so they can't be
counted as two independent wins.
"""
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

_NORMAL = NormalDist()
_TABLE_PATH = Path(__file__).resolve().parent / "correlation_table.json"
try:
    _TABLE = {k: v["rho"] for k, v in json.loads(_TABLE_PATH.read_text()).items()}
except (OSError, ValueError):
    _TABLE = {}

SAME_PLAYER_RHO = 0.9
# Below this a correlation is treated as zero (measurement noise on a
# 20,000-pair estimate is ~0.01; nothing this small changes a parlay).
MIN_MEANINGFUL_RHO = 0.03
MC_SAMPLES = 4000
MC_SEED = 7


def _leg_identity(leg):
    """(stat, team, side, player) in the backtest's vocabulary. Game-level
    markets map to the two pseudo-stats the backtest measured: 'margin'
    (a team's scoring margin vs. the closing spread -- Spread and Moneyline
    both win when it's high, so both are side 'Over' for their team) and
    'total' (combined points vs. the closing total)."""
    market = leg.get("market", "")
    if leg.get("player"):
        return leg.get("stat_field"), leg.get("team"), leg.get("side"), leg["player"]
    if market == "Total":
        side = "Over" if leg.get("selection", "").startswith("Over") else "Under"
        return "total", None, side, None
    if market in ("Spread", "Moneyline"):
        return "margin", leg.get("team"), "Over", None
    return None, None, None, None


def pair_correlation(leg_a, leg_b):
    """Correlation between the two legs' latent outcomes (sign already
    flipped for opposite sides), or 0 when independent/unknown."""
    if leg_a.get("matchup") != leg_b.get("matchup"):
        return 0.0
    stat_a, team_a, side_a, player_a = _leg_identity(leg_a)
    stat_b, team_b, side_b, player_b = _leg_identity(leg_b)
    if not stat_a or not stat_b:
        return 0.0
    same_side = side_a == side_b

    if player_a and player_b and player_a == player_b:
        rho = SAME_PLAYER_RHO
    elif stat_a == "margin" and stat_b == "margin":
        # Same team: near-duplicate bet (parlay_builder drops these pairs).
        # Opposite teams: a hedge -- they can't both cover.
        rho = SAME_PLAYER_RHO if team_a == team_b else -0.99
    elif stat_a == "total" and stat_b == "total":
        rho = SAME_PLAYER_RHO
    elif "margin" in (stat_a, stat_b) and "total" in (stat_a, stat_b):
        rho = 0.0  # measured +0.02: nothing
    elif "total" in (stat_a, stat_b):
        stat = stat_b if stat_a == "total" else stat_a
        rho = _TABLE.get(f"{stat}|total|same_game", 0.0)
    elif "margin" in (stat_a, stat_b):
        stat, player_team = (stat_b, team_b) if stat_a == "margin" else (stat_a, team_a)
        margin_team = team_a if stat_a == "margin" else team_b
        rho = _TABLE.get(f"{stat}|margin|own_team", 0.0)
        if margin_team != player_team:
            rho = -rho  # the opponent's margin is the mirror image of the player's team's
    else:
        relation = "same_team" if team_a == team_b else "opponent"
        rho = _TABLE.get(f"{stat_a}|{stat_b}|{relation}", 0.0)

    if abs(rho) < MIN_MEANINGFUL_RHO:
        return 0.0
    return rho if same_side else -rho


def _bvn_lower(a, b, rho, steps=24):
    """P(X < a, Y < b) for standard bivariate normals with correlation rho:
    Phi(a)Phi(b) plus the integral of the bivariate density over r in
    [0, rho] (Simpson's rule -- plenty for |rho| < 0.95)."""
    base = _NORMAL.cdf(a) * _NORMAL.cdf(b)
    if rho == 0:
        return base
    h = rho / steps

    def density(r):
        denom = 1 - r * r
        return math.exp(-(a * a - 2 * r * a * b + b * b) / (2 * denom)) / (2 * math.pi * math.sqrt(denom))

    total = density(0) + density(rho)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * density(i * h)
    return base + total * h / 3


def pair_lift(p_a, p_b, rho):
    """P(both hit) / (P(a) * P(b)) under the copula -- 1.0 when independent."""
    if rho == 0 or p_a <= 0 or p_b <= 0 or p_a >= 1 or p_b >= 1:
        return 1.0
    joint = _bvn_lower(_NORMAL.inv_cdf(p_a), _NORMAL.inv_cdf(p_b), rho)
    return max(joint, 0.0) / (p_a * p_b)


def joint_probability(legs, probs=None, pair_lifts=None):
    """Fast pairwise approximation. `pair_lifts`, when given, is a cache of
    {(id(leg_a), id(leg_b)): lift} so a search over many combos of the same
    pool computes each pair once."""
    probs = probs or [leg["model_prob"] for leg in legs]
    joint = 1.0
    for p in probs:
        joint *= p
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            key = (id(legs[i]), id(legs[j]))
            lift = pair_lifts.get(key) if pair_lifts is not None else None
            if lift is None:
                lift = pair_lift(probs[i], probs[j], pair_correlation(legs[i], legs[j]))
                if pair_lifts is not None:
                    pair_lifts[key] = lift
            joint *= lift
    return min(joint, 1.0)


def _cholesky(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                diag = matrix[i][i] - s
                if diag <= 1e-9:
                    return None
                lower[i][j] = math.sqrt(diag)
            else:
                lower[i][j] = (matrix[i][j] - s) / lower[j][j]
    return lower


def joint_probability_mc(legs, probs=None, samples=MC_SAMPLES, seed=MC_SEED):
    """Seeded GHK simulation of the full Gaussian copula: P(every leg's
    correlated latent lands below its z(p)). GHK (Geweke-Hajivassiliou-
    Keane) rather than naive hit-counting because a $5 -> $1000 ticket
    sits around 0.5%: counting hits would need ~100k draws for a stable
    number, while GHK draws each latent from its truncated conditional
    and averages the product of conditional probabilities -- every draw
    contributes, so a few thousand give a relative error well under 1%.
    Deterministic for a given set of legs (fixed seed) so the number shown
    doesn't wobble between page loads. Falls back to the pairwise
    approximation if the assembled correlation matrix isn't positive
    definite even after shrinking it toward independence, or skips the
    simulation entirely when nothing in the combo is correlated (then the
    plain product is already exact)."""
    probs = probs or [leg["model_prob"] for leg in legs]
    n = len(legs)
    corr = [[1.0 if i == j else pair_correlation(legs[i], legs[j]) for j in range(n)] for i in range(n)]
    if all(corr[i][j] == 0 for i in range(n) for j in range(n) if i != j):
        return joint_probability(legs, probs)

    lower = None
    for _ in range(12):
        lower = _cholesky(corr)
        if lower is not None:
            break
        corr = [[1.0 if i == j else corr[i][j] * 0.9 for j in range(n)] for i in range(n)]
    if lower is None:
        return joint_probability(legs, probs)

    thresholds = [_NORMAL.inv_cdf(min(max(p, 1e-6), 1 - 1e-6)) for p in probs]
    rng = random.Random(seed)
    total = 0.0
    for _ in range(samples):
        weight = 1.0
        eta = [0.0] * n  # the truncated standard-normal draws so far
        for i in range(n):
            cond_mean = sum(lower[i][k] * eta[k] for k in range(i))
            bound = (thresholds[i] - cond_mean) / lower[i][i]
            p_i = _NORMAL.cdf(bound)
            weight *= p_i
            if p_i <= 1e-12:
                break
            # draw eta_i from N(0,1) truncated to (-inf, bound)
            u = rng.random() * p_i
            eta[i] = _NORMAL.inv_cdf(min(max(u, 1e-12), 1 - 1e-12))
        total += weight
    return min(total / samples, 1.0)
