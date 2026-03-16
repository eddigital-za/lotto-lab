# lotto_lab.py
# Outrageous Lottery Prediction Lab (for fun, not profit)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import math
import random
import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np

from fetch_sa_lotto import fetch_sa_lotto_history, fetch_current_lotto_jackpot

# Optional: comment these if you don't have scipy/sklearn installed
try:
    from scipy.stats import chisquare
except ImportError:
    chisquare = None

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
except ImportError:
    KMeans = None
    LogisticRegression = None


# ===== Strategy profiles (Module E+) =====

BASE_WEIGHTS = {
    "freq": 1.0,
    "recency": 0.7,
    "pattern": 0.6,
    "crowd": 0.9,
    "spacing": 0.5,
}

JACKPOT_WEIGHTS = {
    "freq": 1.3,
    "recency": 0.4,
    "pattern": 0.9,
    "crowd": 0.4,
    "spacing": 0.9,
}

EV_INTENSITY_LOW = 0.20
EV_INTENSITY_HIGH = 0.60


def choose_strategy_weights(ev_intensity: float):
    """
    Blend between BASE_WEIGHTS and JACKPOT_WEIGHTS based on EV intensity.
    """
    ev_intensity = max(0.0, min(1.0, ev_intensity))

    if ev_intensity <= EV_INTENSITY_LOW:
        alpha = 0.0
        mode = "HYBRID"
    elif ev_intensity >= EV_INTENSITY_HIGH:
        alpha = 1.0
        mode = "JACKPOT_HUNTER"
    else:
        alpha = ((ev_intensity - EV_INTENSITY_LOW) /
                 (EV_INTENSITY_HIGH - EV_INTENSITY_LOW))
        mode = "HYBRID -> JACKPOT_HUNTER (mixed)"

    blended = {}
    for key in BASE_WEIGHTS.keys():
        b = BASE_WEIGHTS[key]
        j = JACKPOT_WEIGHTS[key]
        blended[key] = (1 - alpha) * b + alpha * j

    return mode, alpha, blended


# =========================
# 1. CORE DATA STRUCTURES
# =========================

@dataclass
class Draw:
    draw_id: int
    date: str          # YYYY-MM-DD or any string
    numbers: List[int]
    machine_id: Optional[str] = None
    ball_set_id: Optional[str] = None
    jackpot: Optional[float] = None
    rollover_count: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


def load_draws_from_csv(path: str, n_per_draw: int) -> List[Draw]:
    """
    TODO: Implement CSV loading for your actual data.
    For now this is just a placeholder.
    """
    import csv
    draws = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums = []
            for i in range(1, n_per_draw + 1):
                key = f"n{i}"
                nums.append(int(row[key]))
            d = Draw(
                draw_id=int(row.get("draw_id", len(draws))),
                date=row.get("date", ""),
                numbers=nums,
                machine_id=row.get("machine_id"),
                ball_set_id=row.get("ball_set_id"),
                jackpot=float(row["jackpot"]) if row.get("jackpot") else None,
                rollover_count=int(row["rollover_count"]) if row.get("rollover_count") else None,
                meta={}
            )
            draws.append(d)
    return draws


# =========================
# 2. FREQUENCY & RECENCY
# =========================

class FrequencyRecencyModule:
    def __init__(self, n_max: int):
        self.n_max = n_max

    def fit(self, draws: List[Draw]):
        counts = np.zeros(self.n_max + 1, dtype=float)
        last_seen = np.full(self.n_max + 1, fill_value=np.inf, dtype=float)

        for t, d in enumerate(draws):
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    counts[n] += 1
                    last_seen[n] = t

        total = counts.sum()
        alpha = 1.0
        freq = (counts + alpha) / (total + alpha * self.n_max)

        max_t = len(draws)
        age = np.where(np.isfinite(last_seen), max_t - last_seen, max_t)
        age_norm = age / (age.max() if age.max() > 0 else 1.0)

        # hot: recent = high score, cold: old = high score
        hot_score = 1.0 + 0.8 * (1 - age_norm)
        cold_score = 1.0 + 0.8 * age_norm

        self.counts = counts
        self.freq = freq
        self.hot = hot_score
        self.cold = cold_score
        self.age = age
        self.age_norm = age_norm

    def get_scores(self) -> Dict[str, np.ndarray]:
        return {
            "freq": self.freq,
            "hot": self.hot,
            "cold": self.cold,
            "age": self.age,
            "age_norm": self.age_norm,
        }


# =========================
# 3. PAIR & TRIPLET SYNERGY (UPGRADED)
# =========================

class PairTripletModule:
    def __init__(self, n_max: int, min_triplet_count: int = 3, min_triplet_synergy: float = 1.2):
        """
        min_triplet_count: only keep triplets that appeared at least this many times
        min_triplet_synergy: only keep triplets whose observed/expected >= this
        """
        self.n_max = n_max
        self.min_triplet_count = min_triplet_count
        self.min_triplet_synergy = min_triplet_synergy

    def fit(self, draws: List[Draw]):
        n = self.n_max

        # ---------- PAIRS ----------
        pair_counts = np.zeros((n + 1, n + 1), dtype=float)

        # ---------- TRIPLETS (sparse dict) ----------
        triplet_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        for d in draws:
            nums = sorted(set(d.numbers))
            # pairs
            for a, b in combinations(nums, 2):
                pair_counts[a, b] += 1
                pair_counts[b, a] += 1
            # triplets
            for a, b, c in combinations(nums, 3):
                key = (a, b, c)
                triplet_counts[key] += 1

        total_draws = len(draws)
        if total_draws == 0:
            self.synergy_log = np.zeros_like(pair_counts)
            self.triplet_bonus_dict = {}
            return

        # ----- pair synergy: observed vs expected under independence -----
        single_counts = pair_counts.sum(axis=1)
        expected_pair = np.outer(single_counts, single_counts) / float(total_draws)

        with np.errstate(divide="ignore", invalid="ignore"):
            synergy_pair = np.where(expected_pair > 0, pair_counts / expected_pair, 1.0)

        # compress a bit: log1p keeps things tame, 0 when synergy==1
        self.synergy_log = np.log1p(0.2 * (synergy_pair - 1.0))

        # ----- triplet synergy -----
        # In a 6-from-N game, probability that a specific triplet appears in a draw:
        # P(triplet) = C(N-3, K-3) / C(N, K)
        # Expected count = P(triplet) * total_draws
        # We'll assume K is constant and infer it from any draw.
        K = len(draws[0].numbers)
        total_combos = math.comb(n, K)
        triplet_combos = math.comb(n - 3, K - 3)
        p_triplet = triplet_combos / float(total_combos)
        expected_triplet_count = p_triplet * total_draws

        triplet_bonus_dict: Dict[Tuple[int, int, int], float] = {}

        for key, obs in triplet_counts.items():
            if obs < self.min_triplet_count:
                continue
            synergy = obs / expected_triplet_count if expected_triplet_count > 0 else 1.0
            if synergy >= self.min_triplet_synergy:
                # convert to a small additive log-bonus similar to pairs
                bonus = math.log1p(0.2 * (synergy - 1.0))
                triplet_bonus_dict[key] = bonus

        self.triplet_bonus_dict = triplet_bonus_dict

    def pair_bonus(self, a: int, b: int) -> float:
        return float(self.synergy_log[a, b])

    def triplet_bonus(self, a: int, b: int, c: int) -> float:
        s = sorted((a, b, c))
        key: Tuple[int, int, int] = (s[0], s[1], s[2])
        return self.triplet_bonus_dict.get(key, 0.0)


# =========================
# 4. REGIME / CLUSTERING
# =========================

class RegimeModule:
    def __init__(self, n_max: int, n_clusters: int = 3):
        self.n_max = n_max
        self.n_clusters = n_clusters
        self.enabled = KMeans is not None

    def _draw_features(self, d: Draw) -> List[float]:
        nums = np.array(d.numbers, dtype=float)
        feat = []
        feat.append(nums.mean())
        feat.append(nums.std())
        feat.append(((nums % 2) == 0).sum())
        bands = self.n_max // 10 + 1
        band_counts = np.zeros(bands, dtype=float)
        for n in nums:
            band_counts[(int(n) - 1) // 10] += 1
        feat.extend(band_counts.tolist())
        return feat

    def fit(self, draws: List[Draw]):
        if not self.enabled or len(draws) < self.n_clusters:
            self.labels_ = np.zeros(len(draws), dtype=int)
            self.current_regime = 0
            self.km = None
            return

        X = np.array([self._draw_features(d) for d in draws])
        self.km = KMeans(n_clusters=self.n_clusters, n_init=10)
        self.labels_ = self.km.fit_predict(X)
        self.current_regime = int(self.labels_[-1])

    def regime_for_draw(self, d: Draw) -> int:
        if self.km is None:
            return 0
        x = np.array(self._draw_features(d)).reshape(1, -1)
        return int(self.km.predict(x)[0])


# =========================
# 5. BIAS & ANOMALY DETECTION
# =========================

class BiasModule:
    def __init__(self, n_max: int):
        self.n_max = n_max

    def fit(self, draws: List[Draw]):
        counts = np.zeros(self.n_max + 1, dtype=float)
        for d in draws:
            for n in d.numbers:
                counts[n] += 1

        observed = counts[1:]
        if observed.sum() == 0:
            self.weights = np.ones_like(counts)
            return

        expected = np.full_like(observed, observed.mean())
        if chisquare is not None:
            chi, p = chisquare(observed, expected)
        else:
            # If scipy not available, fake a "no anomaly" p-value
            chi, p = 0.0, 1.0

        if p < 0.01:
            deviation = observed / expected
            bias_weight = deviation / deviation.mean()
        else:
            bias_weight = np.ones_like(observed, dtype=float)

        self.weights = np.concatenate([[1.0], bias_weight])

    def get_weights(self) -> np.ndarray:
        return self.weights


# =========================
# 6. GAME-STRUCTURE / EV (UPGRADED USAGE)
# =========================

class EVModule:
    """
    Generic K-of-N lotto EV calculator.
    prize_table: {matches: prize_amount} for NON-jackpot tiers.
    Jackpot passed in per run.
    """
    def __init__(self, n_max: int, k: int, ticket_price: float, prize_table: Dict[int, float]):
        self.n_max = n_max
        self.k = k
        self.ticket_price = ticket_price
        self.prize_table = prize_table

    def _comb(self, n: int, r: int) -> int:
        return math.comb(n, r)

    def baseline_ev_ratio(self, jackpot: float) -> float:
        """
        Returns EV / ticket_price (e.g. 0.6 means 60c back on R1 ticket).
        Very general: you plug in current jackpot & prize_table.
        """
        N = self.n_max
        K = self.k
        total_combos = self._comb(N, K)

        ev = 0.0

        # non-jackpot tiers
        for matches, prize in self.prize_table.items():
            if matches > K:
                continue
            ways = self._comb(K, matches) * self._comb(N - K, K - matches)
            prob = ways / float(total_combos)
            ev += prob * prize

        # jackpot tier (full K matches)
        ways_full = 1
        prob_full = ways_full / float(total_combos)
        ev += prob_full * jackpot

        return ev / self.ticket_price

    def play_intensity(self, jackpot: float) -> float:
        """
        Map EV ratio → [0.2, 1.5] intensity multiplier.
        <1 means negative EV but we still allow "for fun" play.
        >1 only if EV is extremely good (rare).
        """
        r = self.baseline_ev_ratio(jackpot)
        # squish into a sane range
        if r <= 0:
            return 0.2
        # 0.5 → ~0.4, 1.0 → ~1.0, 2.0 → ~1.5
        intensity = max(0.2, min(1.5, r ** 0.7))
        return intensity


# =========================
# 7. BAYESIAN PRIOR MODULE (replaces VoodooModule)
# =========================

class BayesianPriorModule:
    """
    Models each number's appearance as Bernoulli(theta_i) with a symmetric
    Beta(alpha, alpha) prior (Dirichlet over the full number set).

    Posterior mean and uncertainty are combined into a per-number weight:
      - Numbers appearing LESS than their prior predicts get a regression-to-mean bonus
      - Numbers with high posterior uncertainty (sparse data) get a small exploration bonus

    This replaces VoodooModule's meaningless random noise with a principled signal.
    """
    def __init__(self, n_max: int, prior_alpha: float = 2.0, uncertainty_bonus: float = 0.12):
        self.n_max = n_max
        self.prior_alpha = prior_alpha
        self.uncertainty_bonus = uncertainty_bonus

    def fit(self, draws: List[Draw]):
        n = self.n_max
        counts = np.zeros(n + 1, dtype=float)
        total_draws = len(draws)

        for d in draws:
            for num in d.numbers:
                if 1 <= num <= n:
                    counts[num] += 1

        alpha = self.prior_alpha
        weights = np.ones(n + 1, dtype=float)
        expected_p = 6.0 / n  # theoretical probability for 6-from-N

        for i in range(1, n + 1):
            a = counts[i] + alpha
            b = (total_draws - counts[i]) + alpha
            ab = a + b
            posterior_mean = a / ab
            posterior_var = (a * b) / (ab * ab * (ab + 1.0))
            posterior_std = posterior_var ** 0.5

            # Bonus when number appears less than expected (regression to mean)
            underperformance = max(0.0, expected_p - posterior_mean) / max(expected_p, 1e-8)
            # Exploration bonus proportional to uncertainty
            uncertainty = self.uncertainty_bonus * posterior_std / (posterior_std + 0.01)

            weights[i] = 1.0 + 0.4 * underperformance + uncertainty

        weights[1:] = np.clip(weights[1:], 0.6, 1.8)
        self.weights = weights

    def get_weights(self) -> np.ndarray:
        return self.weights


# Keep VoodooModule as alias for backward compatibility
VoodooModule = BayesianPriorModule


# =========================
# 7b. MARKOV CHAIN MODULE
# =========================

class MarkovModule:
    """
    First-order Markov chain: models P(j appears in draw t+1 | i appeared in draw t).

    Uses recency-weighted transition counts so recent draw patterns carry more
    weight than transitions from years ago. The result is a per-number bonus/penalty
    based on how strongly the current (last) draw's numbers predict each candidate.
    """
    def __init__(self, n_max: int, decay: float = 0.93, strength: float = 0.22):
        self.n_max = n_max
        self.decay = decay      # per-draw exponential decay for older transitions
        self.strength = strength
        self.transition_prob = np.zeros((n_max + 1, n_max + 1), dtype=float)
        self.last_draw_numbers: List[int] = []

    def fit(self, draws: List[Draw]):
        n = self.n_max
        transition_counts = np.zeros((n + 1, n + 1), dtype=float)
        from_counts = np.zeros(n + 1, dtype=float)

        total = len(draws)
        for t in range(total - 1):
            curr = draws[t].numbers
            nxt = draws[t + 1].numbers
            age = total - t - 1
            w = self.decay ** age
            for i in curr:
                if 1 <= i <= n:
                    from_counts[i] += w
                    for j in nxt:
                        if 1 <= j <= n:
                            transition_counts[i, j] += w

        self.transition_prob = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(1, n + 1):
            if from_counts[i] > 0:
                self.transition_prob[i, :] = transition_counts[i, :] / from_counts[i]

        self.last_draw_numbers = list(draws[-1].numbers) if draws else []

    def number_bonus(self, num: int) -> float:
        """Bonus for num based on Markov carry-over from the most recent draw."""
        if not self.last_draw_numbers or num < 1 or num > self.n_max:
            return 0.0
        total_p = 0.0
        count = 0
        for prev in self.last_draw_numbers:
            if 1 <= prev <= self.n_max:
                total_p += self.transition_prob[prev, num]
                count += 1
        if count == 0:
            return 0.0
        avg_p = total_p / count
        expected_p = 6.0 / self.n_max
        deviation = (avg_p - expected_p) / max(expected_p, 1e-8)
        return float(self.strength * float(np.tanh(deviation)))


# =========================
# A. ODD/EVEN & HIGH/LOW BALANCER
# =========================

class PatternBalanceModule:
    """
    Learns typical odd/even and high/low counts per draw, and
    applies a small penalty to tickets that deviate too far.
    """
    def __init__(
        self,
        n_max: int,
        high_split: int | None = None,
        lambda_odd: float = 0.15,
        lambda_high: float = 0.15,
    ):
        self.n_max = n_max
        self.high_split = high_split if high_split is not None else n_max // 2
        self.lambda_odd = lambda_odd
        self.lambda_high = lambda_high
        self.target_odd = None
        self.target_high = None

    def fit(self, draws: List[Draw]):
        if not draws:
            k = 6
            self.target_odd = k / 2.0
            self.target_high = k / 2.0
            return

        k = len(draws[0].numbers)
        odd_counts = []
        high_counts = []

        for d in draws:
            nums = d.numbers
            odd_c = sum(1 for n in nums if n % 2 == 1)
            high_c = sum(1 for n in nums if n > self.high_split)
            odd_counts.append(odd_c)
            high_counts.append(high_c)

        self.target_odd = float(np.mean(odd_counts))
        self.target_high = float(np.mean(high_counts))

    def ticket_penalty(self, ticket: List[int]) -> float:
        """
        Returns a non-positive penalty (<= 0) that is added to the ticket score.
        Larger deviations from the historical pattern => more negative.
        """
        if self.target_odd is None or self.target_high is None:
            return 0.0

        odd_c = sum(1 for n in ticket if n % 2 == 1)
        high_c = sum(1 for n in ticket if n > self.high_split)

        dev_odd = abs(odd_c - self.target_odd)
        dev_high = abs(high_c - self.target_high)

        penalty = - (self.lambda_odd * dev_odd + self.lambda_high * dev_high)
        return float(penalty)


# =========================
# B. REPEAT / SKIP CYCLE DETECTOR
# =========================

class RepeatSkipModule:
    """
    Learns typical appearance gaps for each number over a recent window,
    then scores numbers by how close their current 'age' is to that gap.

    Idea:
      - For each number, compute average gap between hits in last W draws.
      - Compute 'age' = draws since last seen.
      - If age ~ avg_gap => slight positive bonus (on-cycle).
      - If much smaller or much larger => slight penalty (off-cycle).
    """

    def __init__(
        self,
        n_max: int,
        window: int = 120,
        spread: float = 0.75,
        lam: float = 0.25,
    ):
        self.n_max = n_max
        self.window = window
        self.spread = spread
        self.lam = lam

        self.avg_gap = np.zeros(n_max + 1, dtype=float)
        self.age = np.zeros(n_max + 1, dtype=float)
        self.has_stats = np.zeros(n_max + 1, dtype=bool)

    def fit(self, draws: List[Draw]):
        if not draws:
            return

        relevant = draws[-self.window :] if len(draws) > self.window else draws
        last_seen = [-1] * (self.n_max + 1)
        total_gap = np.zeros(self.n_max + 1, dtype=float)
        count_gap = np.zeros(self.n_max + 1, dtype=float)

        for idx, d in enumerate(relevant):
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    if last_seen[n] != -1:
                        gap = idx - last_seen[n]
                        if gap > 0:
                            total_gap[n] += gap
                            count_gap[n] += 1.0
                    last_seen[n] = idx

        last_index = len(relevant) - 1
        for n in range(1, self.n_max + 1):
            if count_gap[n] > 0 and last_seen[n] != -1:
                self.avg_gap[n] = total_gap[n] / count_gap[n]
                self.age[n] = last_index - last_seen[n]
                self.has_stats[n] = True
            else:
                self.has_stats[n] = False

    def number_bonus(self, n: int) -> float:
        """
        Returns a small bonus/penalty for number n based on its cycle phase.
        Range roughly [-lam, +lam].
        """
        if n < 1 or n > self.n_max or not self.has_stats[n]:
            return 0.0

        g = self.avg_gap[n]
        a = self.age[n]
        if g <= 0:
            return 0.0

        ratio = a / g

        dist = abs(ratio - 1.0)
        if dist >= self.spread:
            base = 0.0
        else:
            base = 1.0 - (dist / self.spread)

        score = self.lam * (2.0 * (base - 0.5))
        return float(score)


# =========================
# C. HYBRID MOMENTUM MODULE
# =========================

class MomentumModule:
    """
    Hybrid 'momentum' model with fast + slow curves.
    - Fast curve = aggressive / jackpot hunter (reacts quickly to recent draws)
    - Slow curve = conservative / long-term stability
    - ev_intensity (0..1) shifts weight toward fast curve when jackpot is big.
    Output: small log-scale bonuses/penalties per number.
    """

    def __init__(
        self,
        n_max: int,
        fast_half_life: int = 10,   # ~how many draws before fast signal halves
        slow_half_life: int = 60,   # slow memory
        lam: float = 0.35,          # overall strength of the effect
        base_fast_weight: float = 0.45,  # default split fast vs slow at EV=0.5
    ):
        self.n_max = n_max
        self.fast_half_life = fast_half_life
        self.slow_half_life = slow_half_life
        self.lam = lam
        self.base_fast_weight = base_fast_weight

        self.momentum = np.zeros(n_max + 1, dtype=float)
        self.mean_momentum = 0.0

    def _decay_factor(self, half_life: int) -> float:
        # Each step multiplies by this; after half_life steps, value ~0.5
        if half_life <= 0:
            return 1.0
        return 0.5 ** (1.0 / float(half_life))

    def fit(self, draws: List[Draw], ev_intensity: float = 0.5):
        """
        Build fast + slow momentum curves, mix them based on ev_intensity.

        ev_intensity in [0,1]:
          ~0.2 -> quite conservative
          ~0.5 -> balanced
          ~0.8+ -> very aggressive / jackpot hunter
        """
        if not draws:
            return

        fast_decay = self._decay_factor(self.fast_half_life)
        slow_decay = self._decay_factor(self.slow_half_life)

        fast = np.zeros(self.n_max + 1, dtype=float)
        slow = np.zeros(self.n_max + 1, dtype=float)

        # Use all available draws (they're not huge), oldest -> newest
        for d in draws:
            fast *= fast_decay
            slow *= slow_decay
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    fast[n] += 1.0
                    slow[n] += 1.0

        # Normalise each curve to [0,1] (per curve) to avoid scale issues
        if fast.max() > 0:
            fast /= fast.max()
        if slow.max() > 0:
            slow /= slow.max()

        # EV-controlled mixing: shift toward fast when jackpot is big
        # ev_intensity ~0.0 => almost all slow
        # ev_intensity ~1.0 => strongly fast
        ev_intensity = float(max(0.0, min(1.0, ev_intensity)))  # clamp

        # This line is the "jackpot hunter dial":
        # at EV=0.5, fast ~ base_fast_weight
        # push +/- 0.3 depending on EV; clamp to [0.2, 0.8]
        fast_w = self.base_fast_weight + (ev_intensity - 0.5) * 0.6
        fast_w = max(0.2, min(0.8, fast_w))
        slow_w = 1.0 - fast_w

        # Combine into hybrid momentum
        for n in range(1, self.n_max + 1):
            self.momentum[n] = fast_w * fast[n] + slow_w * slow[n]

        # Centre around mean so bonuses sum ~0
        self.mean_momentum = float(self.momentum[1:].mean() if self.n_max >= 1 else 0.0)

    def number_bonus(self, n: int) -> float:
        """
        Small log-scale bonus/penalty for number n.
        Rough range about [-lam/2, +lam/2].
        """
        if n < 1 or n > self.n_max:
            return 0.0
        delta = self.momentum[n] - self.mean_momentum
        return float(self.lam * delta)


# =========================
# C. TEMPORAL HOT/COLD MOMENTUM MODULE
# =========================

class TemporalMomentumModule:
    """
    Advanced temporal modeling that combines:
      1. Forward-weighted recency (exponential decay on recent hits)
      2. Long-term vs short-term momentum overlap detection
      3. Entropy windows (measure randomness in recent draws)
      4. Cluster decay (detect and decay clustered appearances)
      5. Frequency wave patterns (sinusoidal trend detection)

    This module should start outperforming random in a measurable way.
    """

    def __init__(
        self,
        n_max: int,
        short_window: int = 20,
        long_window: int = 80,
        decay_rate: float = 0.92,
        momentum_weight: float = 0.35,
        entropy_weight: float = 0.15,
        cluster_weight: float = 0.20,
        wave_weight: float = 0.15,
    ):
        self.n_max = n_max
        self.short_window = short_window
        self.long_window = long_window
        self.decay_rate = decay_rate
        self.momentum_weight = momentum_weight
        self.entropy_weight = entropy_weight
        self.cluster_weight = cluster_weight
        self.wave_weight = wave_weight

        self.recency_score = np.zeros(n_max + 1, dtype=float)
        self.momentum_score = np.zeros(n_max + 1, dtype=float)
        self.entropy_score = np.zeros(n_max + 1, dtype=float)
        self.cluster_score = np.zeros(n_max + 1, dtype=float)
        self.wave_score = np.zeros(n_max + 1, dtype=float)
        self.combined_score = np.zeros(n_max + 1, dtype=float)

    def fit(self, draws: List[Draw]):
        if len(draws) < 10:
            return

        n_max = self.n_max
        long_w = min(self.long_window, len(draws))
        short_w = min(self.short_window, len(draws))

        recent_long = draws[-long_w:]
        recent_short = draws[-short_w:]

        self._compute_recency(recent_long)
        self._compute_momentum(recent_short, recent_long)
        self._compute_entropy(recent_short)
        self._compute_cluster_decay(recent_long)
        self._compute_wave_pattern(recent_long)
        self._combine_scores()

    def _compute_recency(self, draws: List[Draw]):
        """Forward-weighted recency with exponential decay."""
        self.recency_score.fill(0.0)
        n_draws = len(draws)
        for idx, d in enumerate(draws):
            age = n_draws - idx - 1
            weight = self.decay_rate ** age
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    self.recency_score[n] += weight

        if self.recency_score.max() > 0:
            self.recency_score /= self.recency_score.max()

    def _compute_momentum(self, short_draws: List[Draw], long_draws: List[Draw]):
        """
        Compare short-term frequency vs long-term frequency.
        Momentum = (short_rate - long_rate) normalized.
        Positive momentum = number is 'heating up'.
        """
        self.momentum_score.fill(0.0)

        short_count = np.zeros(self.n_max + 1, dtype=float)
        long_count = np.zeros(self.n_max + 1, dtype=float)

        for d in short_draws:
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    short_count[n] += 1.0

        for d in long_draws:
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    long_count[n] += 1.0

        short_rate = short_count / max(len(short_draws), 1)
        long_rate = long_count / max(len(long_draws), 1)

        self.momentum_score[1:] = short_rate[1:] - long_rate[1:]

        max_abs = np.abs(self.momentum_score[1:]).max()
        if max_abs > 0:
            self.momentum_score[1:] /= max_abs

    def _compute_entropy(self, draws: List[Draw]):
        """
        Measure local randomness/predictability in recent window.
        Lower entropy for a number => more predictable pattern => slight bonus.
        """
        self.entropy_score.fill(0.0)

        appearances = {n: [] for n in range(1, self.n_max + 1)}
        for idx, d in enumerate(draws):
            for n in d.numbers:
                if 1 <= n <= self.n_max:
                    appearances[n].append(idx)

        for n in range(1, self.n_max + 1):
            gaps = []
            app = appearances[n]
            if len(app) >= 2:
                for i in range(1, len(app)):
                    gaps.append(app[i] - app[i - 1])

            if len(gaps) >= 2:
                gap_std = float(np.std(gaps))
                gap_mean = float(np.mean(gaps))
                if gap_mean > 0:
                    cv = gap_std / gap_mean
                    self.entropy_score[n] = 1.0 - min(cv, 2.0) / 2.0
            elif len(app) >= 1:
                self.entropy_score[n] = 0.5

    def _compute_cluster_decay(self, draws: List[Draw]):
        """
        Detect numbers that appeared in clusters and apply decay.
        Clustered numbers may be 'cooling off'.
        """
        self.cluster_score.fill(0.0)
        n_draws = len(draws)

        for n in range(1, self.n_max + 1):
            indices = []
            for idx, d in enumerate(draws):
                if n in d.numbers:
                    indices.append(idx)

            if len(indices) < 2:
                self.cluster_score[n] = 0.0
                continue

            cluster_count = 0
            for i in range(1, len(indices)):
                if indices[i] - indices[i - 1] <= 3:
                    cluster_count += 1

            cluster_ratio = cluster_count / (len(indices) - 1)

            if len(indices) > 0:
                recency = n_draws - indices[-1] - 1
                if cluster_ratio > 0.4 and recency < 5:
                    self.cluster_score[n] = -0.3 * cluster_ratio
                elif cluster_ratio < 0.2:
                    self.cluster_score[n] = 0.1

    def _compute_wave_pattern(self, draws: List[Draw]):
        """
        Detect sinusoidal frequency waves in number appearances.
        Some numbers may show periodic hot/cold cycles.
        """
        self.wave_score.fill(0.0)
        n_draws = len(draws)

        if n_draws < 20:
            return

        for n in range(1, self.n_max + 1):
            signal = np.zeros(n_draws, dtype=float)
            for idx, d in enumerate(draws):
                if n in d.numbers:
                    signal[idx] = 1.0

            if signal.sum() < 3:
                continue

            window_size = min(10, n_draws // 3)
            smoothed = np.convolve(
                signal, np.ones(window_size) / window_size, mode='valid'
            )

            if len(smoothed) < 5:
                continue

            trend = smoothed[-1] - smoothed[0]
            recent_slope = smoothed[-1] - smoothed[-min(3, len(smoothed))]

            if recent_slope > 0.02:
                self.wave_score[n] = min(recent_slope * 5, 0.5)
            elif recent_slope < -0.02:
                self.wave_score[n] = max(recent_slope * 3, -0.3)

    def _combine_scores(self):
        """Weighted combination of all temporal signals."""
        self.combined_score.fill(0.0)

        w_rec = 1.0 - (
            self.momentum_weight + self.entropy_weight +
            self.cluster_weight + self.wave_weight
        )
        w_rec = max(w_rec, 0.15)

        self.combined_score[1:] = (
            w_rec * self.recency_score[1:] +
            self.momentum_weight * self.momentum_score[1:] +
            self.entropy_weight * self.entropy_score[1:] +
            self.cluster_weight * self.cluster_score[1:] +
            self.wave_weight * self.wave_score[1:]
        )

        max_abs = np.abs(self.combined_score[1:]).max()
        if max_abs > 0:
            self.combined_score[1:] /= max_abs

    def number_bonus(self, n: int, scale: float = 0.3) -> float:
        """
        Returns a bonus/penalty for number n based on temporal momentum.
        Range roughly [-scale, +scale].
        """
        if n < 1 or n > self.n_max:
            return 0.0
        return float(scale * self.combined_score[n])

    def get_hot_numbers(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return the top-k 'hottest' numbers by combined score."""
        scores = [(n, self.combined_score[n]) for n in range(1, self.n_max + 1)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_cold_numbers(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return the top-k 'coldest' numbers by combined score."""
        scores = [(n, self.combined_score[n]) for n in range(1, self.n_max + 1)]
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


# =========================
# 8. META-AI MODULE (UPGRADED)
# =========================

class MetaAIModule:
    """
    Per-number classifier using the full history with 12 engineered features.

    Upgraded from 2 features (freq, age) to 12 rich features per number:
      freq, age, parity, decade band, avg gap, gap variance, cycle phase,
      short-term rate, long-term rate, momentum, recent streak, last-draw flag.

    Uses LogisticRegression with StandardScaler and class_weight='balanced'
    to handle the extreme class imbalance (6 positives per 52 numbers per draw).
    """
    def __init__(self, n_max: int):
        self.n_max = n_max
        self.enabled = LogisticRegression is not None
        self.model = None
        self.scaler = None
        self._current_feats: Optional[np.ndarray] = None

    def fit(self, draws: List[Draw]):
        if not self.enabled or len(draws) < 15:
            self.model = None
            return

        n_max = self.n_max
        SHORT_W = 10
        LONG_W = 40

        counts = np.zeros(n_max + 1, dtype=float)
        last_seen = np.full(n_max + 1, fill_value=-1, dtype=int)
        gap_sums = np.zeros(n_max + 1, dtype=float)
        gap_counts = np.zeros(n_max + 1, dtype=float)
        gap_sq_sums = np.zeros(n_max + 1, dtype=float)

        # Rolling window counts using sliding totals
        from collections import deque
        short_q: deque = deque()  # stores (draw_index, number) pairs in window
        long_q: deque = deque()
        short_counts = np.zeros(n_max + 1, dtype=float)
        long_counts = np.zeros(n_max + 1, dtype=float)

        X: List[List[float]] = []
        y: List[int] = []

        for t in range(len(draws) - 1):
            d = draws[t]

            # --- slide rolling windows ---
            while short_q and short_q[0][0] <= t - SHORT_W:
                _, old_n = short_q.popleft()
                if 1 <= old_n <= n_max:
                    short_counts[old_n] -= 1.0
            while long_q and long_q[0][0] <= t - LONG_W:
                _, old_n = long_q.popleft()
                if 1 <= old_n <= n_max:
                    long_counts[old_n] -= 1.0

            # --- update running stats with draw t ---
            for num in d.numbers:
                if 1 <= num <= n_max:
                    if last_seen[num] >= 0:
                        g = t - last_seen[num]
                        gap_sums[num] += g
                        gap_sq_sums[num] += g * g
                        gap_counts[num] += 1.0
                    counts[num] += 1.0
                    last_seen[num] = t
                    short_counts[num] += 1.0
                    long_counts[num] += 1.0
                    short_q.append((t, num))
                    long_q.append((t, num))

            total_hits = counts.sum()
            if total_hits == 0:
                continue

            next_nums = set(draws[t + 1].numbers)

            for i in range(1, n_max + 1):
                freq = counts[i] / total_hits
                age = float((t - last_seen[i]) if last_seen[i] >= 0 else (t + 1))
                parity = float(i % 2)
                decade = float(i // 10) / 5.0

                gc = gap_counts[i]
                if gc >= 2:
                    gm = gap_sums[i] / gc
                    gv = max(0.0, gap_sq_sums[i] / gc - gm * gm) ** 0.5
                    cycle_phase = min(age / max(gm, 1.0), 3.0)
                elif gc == 1:
                    gm = gap_sums[i]
                    gv = 0.0
                    cycle_phase = min(age / max(gm, 1.0), 3.0)
                else:
                    gm = float(t + 1)
                    gv = 0.0
                    cycle_phase = 1.0

                short_rate = short_counts[i] / SHORT_W
                long_rate = long_counts[i] / LONG_W
                momentum = float(np.tanh((short_rate - long_rate) * 10.0))

                streak = sum(1 for td in range(max(0, t - 4), t + 1)
                             if i in draws[td].numbers)

                feat = [
                    float(freq),
                    min(age, 100.0) / 100.0,
                    parity,
                    decade,
                    min(gm, 50.0) / 50.0,
                    min(gv, 20.0) / 20.0,
                    cycle_phase / 3.0,
                    min(short_rate, 1.0),
                    min(long_rate, 1.0),
                    momentum,
                    streak / 5.0,
                    1.0 if age == 0.0 else 0.0,
                ]
                X.append(feat)
                y.append(1 if i in next_nums else 0)

        if not X:
            self.model = None
            return

        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=int)

        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(Xarr)
            model = LogisticRegression(
                max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
            )
            model.fit(Xs, yarr)
            self.model = model

            # Pre-compute features for the CURRENT state (last draw) for get_weights()
            self._precompute_current_feats(draws)
        except Exception:
            self.model = None
            self.scaler = None

    def _precompute_current_feats(self, draws: List[Draw]):
        """Build feature vectors representing each number's current state."""
        n_max = self.n_max
        SHORT_W = 10
        LONG_W = 40
        t = len(draws) - 1

        counts = np.zeros(n_max + 1, dtype=float)
        last_seen = np.full(n_max + 1, fill_value=-1, dtype=int)
        gap_sums = np.zeros(n_max + 1, dtype=float)
        gap_counts = np.zeros(n_max + 1, dtype=float)
        gap_sq_sums = np.zeros(n_max + 1, dtype=float)
        short_counts = np.zeros(n_max + 1, dtype=float)
        long_counts = np.zeros(n_max + 1, dtype=float)

        for idx, d in enumerate(draws):
            for num in d.numbers:
                if 1 <= num <= n_max:
                    if last_seen[num] >= 0:
                        g = idx - last_seen[num]
                        gap_sums[num] += g
                        gap_sq_sums[num] += g * g
                        gap_counts[num] += 1.0
                    counts[num] += 1.0
                    last_seen[num] = idx
                    if idx >= t - SHORT_W + 1:
                        short_counts[num] += 1.0
                    if idx >= t - LONG_W + 1:
                        long_counts[num] += 1.0

        total_hits = counts.sum()
        feats = []
        for i in range(1, n_max + 1):
            freq = counts[i] / max(total_hits, 1.0)
            age = float((t - last_seen[i]) if last_seen[i] >= 0 else (t + 1))
            parity = float(i % 2)
            decade = float(i // 10) / 5.0
            gc = gap_counts[i]
            if gc >= 2:
                gm = gap_sums[i] / gc
                gv = max(0.0, gap_sq_sums[i] / gc - gm * gm) ** 0.5
                cycle_phase = min(age / max(gm, 1.0), 3.0)
            elif gc == 1:
                gm = gap_sums[i]; gv = 0.0
                cycle_phase = min(age / max(gm, 1.0), 3.0)
            else:
                gm = float(t + 1); gv = 0.0; cycle_phase = 1.0

            short_rate = short_counts[i] / SHORT_W
            long_rate = long_counts[i] / LONG_W
            momentum = float(np.tanh((short_rate - long_rate) * 10.0))
            streak = sum(1 for td in range(max(0, t - 4), t + 1)
                         if i in draws[td].numbers)
            feats.append([
                float(freq), min(age, 100.0) / 100.0, parity, decade,
                min(gm, 50.0) / 50.0, min(gv, 20.0) / 20.0, cycle_phase / 3.0,
                min(short_rate, 1.0), min(long_rate, 1.0), momentum,
                streak / 5.0, 1.0 if age == 0.0 else 0.0,
            ])
        self._current_feats = np.array(feats, dtype=float)

    def get_weights(self, freq_mod: FrequencyRecencyModule) -> np.ndarray:
        """Returns per-number multiplier from the trained classifier."""
        n_max = self.n_max
        if self.model is None or self._current_feats is None:
            return np.ones(n_max + 1, dtype=float)
        try:
            feats = self._current_feats
            if self.scaler is not None:
                feats = self.scaler.transform(feats)
            probs = self.model.predict_proba(feats)[:, 1]
            mean_p = probs.mean() if probs.mean() > 0 else 1.0
            weights = probs / mean_p
            return np.concatenate([[1.0], weights])
        except Exception:
            return np.ones(n_max + 1, dtype=float)


# =========================
# 9. CROWDING / HUMAN PSYCHOLOGY
# =========================

def birthday_bias_score(ticket: List[int]) -> float:
    return sum(1 for n in ticket if n <= 31) / len(ticket)


def pattern_score(ticket: List[int]) -> float:
    t = sorted(ticket)
    if len(t) <= 2:
        return 0.0
    diffs = [t[i+1] - t[i] for i in range(len(t)-1)]
    var = np.var(diffs)
    # small variance (e.g. 1,1,1) -> strong pattern
    return float(math.exp(-var))


def crowding_penalty(ticket: List[int], weight: float = 0.5) -> float:
    """
    Penalty for human-looking crowded patterns:
    - long consecutive runs (e.g. 10,11,12,13)
    - too many numbers in the same decade (10s, 20s, 30s...)

    'weight' scales how strong the penalty is.
    Returns a non-positive value (<= 0).
    """
    if not ticket:
        return 0.0

    ticket = sorted(ticket)
    penalty = 0.0

    # ---- consecutive run penalty ----
    run_len = 1
    for i in range(1, len(ticket)):
        if ticket[i] == ticket[i - 1] + 1:
            run_len += 1
        else:
            if run_len >= 3:
                penalty -= weight * 0.3 * (run_len - 2)
            run_len = 1
    if run_len >= 3:
        penalty -= weight * 0.3 * (run_len - 2)

    # ---- same-decade crowding penalty ----
    decades: Dict[int, int] = {}
    for n in ticket:
        d = n // 10
        decades[d] = decades.get(d, 0) + 1

    max_decade_count = max(decades.values())
    if max_decade_count >= 4:
        penalty -= weight * 0.2 * (max_decade_count - 3)

    return float(penalty)


# =========================
# 10. NUMBER PROBABILITY COMBINER
# =========================

class NumberCombiner:
    def __init__(self, n_max: int, config: Dict[str, float]):
        self.n_max = n_max
        self.config = config  # weights for each module

    def build_probs(
        self,
        freq_mod: FrequencyRecencyModule,
        bias_mod: BiasModule,
        voodoo_mod: VoodooModule,
        meta_ai_mod: MetaAIModule,
        momentum_mod: Optional[MomentumModule] = None,
    ) -> np.ndarray:
        n = self.n_max
        scores = freq_mod.get_scores()
        freq = scores["freq"]
        hot = scores["hot"]
        cold = scores["cold"]

        bias_w = bias_mod.get_weights()
        voodoo_w = voodoo_mod.get_weights()
        ai_w = meta_ai_mod.get_weights(freq_mod)

        logp = np.zeros(n + 1, dtype=float)

        for i in range(1, n + 1):
            val = 1.0 / n

            val *= freq[i] ** self.config.get("freq", 0.5)
            val *= hot[i] ** self.config.get("hot", 0.2)
            val *= cold[i] ** self.config.get("cold", 0.0)
            val *= bias_w[i] ** self.config.get("bias", 0.3)
            val *= voodoo_w[i] ** self.config.get("voodoo", 0.1)
            val *= ai_w[i] ** self.config.get("ai", 0.1)

            logp[i] = math.log(val + 1e-15)

            # NEW: hybrid momentum bonus (fast+slow, EV-controlled)
            if momentum_mod is not None:
                logp[i] += momentum_mod.number_bonus(i)

        max_log = logp[1:].max()
        probs = np.exp(logp - max_log)
        probs[0] = 0.0
        probs /= probs.sum()
        return probs


# =========================
# 11. TICKET SCORER & GENERATOR
# =========================

def score_ticket(
    ticket: List[int],
    num_probs: np.ndarray,
    pair_mod: PairTripletModule,
    crowd_weight: float = 0.5,
    use_triplets: bool = False
) -> float:
    s = 0.0
    for n in ticket:
        s += math.log(num_probs[n] + 1e-15)

    for i in range(len(ticket)):
        for j in range(i + 1, len(ticket)):
            s += pair_mod.pair_bonus(ticket[i], ticket[j])

    if use_triplets:
        for i in range(len(ticket)):
            for j in range(i + 1, len(ticket)):
                for k in range(j + 1, len(ticket)):
                    s += pair_mod.triplet_bonus(ticket[i], ticket[j], ticket[k])

    cp = crowding_penalty(ticket)
    s -= crowd_weight * cp
    return s


def sample_ticket(num_probs: np.ndarray, k: int) -> List[int]:
    numbers = list(range(1, len(num_probs)))
    probs = num_probs[1:].copy()
    chosen = []

    for _ in range(k):
        probs = probs / probs.sum()
        x = random.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if x <= cum:
                pick = numbers[idx]
                chosen.append(pick)
                del numbers[idx]
                probs = np.delete(probs, idx)
                break

    return sorted(chosen)


def sample_uniform_ticket(n_max: int, k: int) -> List[int]:
    """
    Sample a ticket uniformly at random from all C(n_max, k) combos.
    Used as a baseline to compare the model against.
    """
    return sorted(random.sample(range(1, n_max + 1), k))


def generate_top_tickets(
    num_probs: np.ndarray,
    pair_mod: PairTripletModule,
    k: int,
    n_samples: int,
    top_n: int,
    crowd_weight: float = 0.5,
    use_triplets: bool = False,
    pattern_mod: Optional[PatternBalanceModule] = None,
    repeat_mod: Optional[RepeatSkipModule] = None,
    temporal_mod: Optional[TemporalMomentumModule] = None,
    markov_mod: Optional[MarkovModule] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[int]]]:
    """
    Monte Carlo sampler + greedy beam merger.

    Generates tickets via random sampling, then merges with greedy beam search
    results to ensure the top candidates include both diverse random discoveries
    and the deterministically optimal high-scoring combinations.
    """
    if weights is None:
        weights = BASE_WEIGHTS

    freq_weight = weights.get("freq", 1.0)
    recency_weight = weights.get("recency", 0.7)
    pattern_weight = weights.get("pattern", 0.6)
    crowd_w = weights.get("crowd", 0.9)
    spacing_weight = weights.get("spacing", 0.5)

    best: List[Tuple[float, List[int]]] = []
    log_probs = np.log(num_probs + 1e-15)

    for _ in range(n_samples):
        ticket = sample_ticket(num_probs, k)
        ticket.sort()
        score = 0.0

        for n in ticket:
            score += freq_weight * log_probs[n]
            if repeat_mod is not None:
                score += repeat_mod.number_bonus(n)
            if temporal_mod is not None:
                score += recency_weight * temporal_mod.number_bonus(n)
            if markov_mod is not None:
                score += markov_mod.number_bonus(n)

        for a, b in combinations(ticket, 2):
            score += pair_mod.pair_bonus(a, b)

        if use_triplets:
            for a, b, c in combinations(ticket, 3):
                score += pair_mod.triplet_bonus(a, b, c)

        score += crowding_penalty(ticket, crowd_w)

        gaps = [ticket[i+1] - ticket[i] for i in range(len(ticket) - 1)]
        if gaps:
            min_gap = min(gaps)
            if min_gap <= 2:
                score -= spacing_weight * (3 - min_gap) * 0.1

        if pattern_mod is not None:
            score += pattern_weight * pattern_mod.ticket_penalty(ticket)

        if len(best) < top_n:
            best.append((score, ticket))
            best.sort(reverse=True, key=lambda x: x[0])
        else:
            if score > best[-1][0]:
                best[-1] = (score, ticket)
                best.sort(reverse=True, key=lambda x: x[0])

    # --- Merge greedy beam results ---
    # Use a smaller beam width so it stays fast; deduplicate against Monte Carlo results
    beam_n = max(20, top_n)
    try:
        beam_tickets = greedy_beam_tickets(
            num_probs, pair_mod, k,
            beam_width=beam_n,
            pattern_mod=pattern_mod,
            repeat_mod=repeat_mod,
            temporal_mod=temporal_mod,
            markov_mod=markov_mod,
            weights=weights,
        )
        existing_sets = {frozenset(t) for _, t in best}
        for score, ticket in beam_tickets:
            if frozenset(ticket) not in existing_sets:
                best.append((score, ticket))
                existing_sets.add(frozenset(ticket))

        best.sort(reverse=True, key=lambda x: x[0])
        best = best[:top_n]
    except Exception:
        pass

    return best


# =========================
# 12. BACKTESTER
# =========================

class Backtester:
    def __init__(self, n_max: int, k: int):
        self.n_max = n_max
        self.k = k

    def evaluate(
        self,
        draws: List[Draw],
        window: int = 100,
        n_samples: int = 2000,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Simple walk-forward backtest:
        For each draw t >= window:
          - fit model on draws[t-window : t]
          - generate top_n tickets
          - measure how many matches with draws[t]
        """
        results = []
        for t in range(window, len(draws)):
            hist = draws[t-window : t]
            target = draws[t]

            freq_mod = FrequencyRecencyModule(self.n_max)
            freq_mod.fit(hist)

            pair_mod = PairTripletModule(self.n_max)
            pair_mod.fit(hist)

            bias_mod = BiasModule(self.n_max)
            bias_mod.fit(hist)

            voodoo_mod = VoodooModule(self.n_max)
            voodoo_mod.fit(hist)

            ai_mod = MetaAIModule(self.n_max)
            ai_mod.fit(hist)

            comb = NumberCombiner(
                self.n_max,
                config={
                    "freq": 0.5,
                    "hot": 0.2,
                    "cold": 0.0,
                    "bias": 0.3,
                    "voodoo": 0.1,
                    "ai": 0.1,
                }
            )
            num_probs = comb.build_probs(freq_mod, bias_mod, voodoo_mod, ai_mod)

            tickets = generate_top_tickets(
                num_probs,
                pair_mod,
                k=self.k,
                n_samples=n_samples,
                top_n=top_n
            )

            # evaluate hits
            target_set = set(target.numbers)
            hit_counts = []
            for _, ticket in tickets:
                hits = len(target_set.intersection(ticket))
                hit_counts.append(hits)

            results.append({
                "t": t,
                "draw_id": target.draw_id,
                "max_hits": max(hit_counts) if hit_counts else 0,
                "avg_hits": float(sum(hit_counts) / len(hit_counts)) if hit_counts else 0.0
            })

        return {
            "per_step": results,
            "avg_max_hits": float(sum(r["max_hits"] for r in results) / len(results)) if results else 0.0,
            "avg_avg_hits": float(sum(r["avg_hits"] for r in results) / len(results)) if results else 0.0,
        }


# =========================
# 12b. WEIGHT CALIBRATOR
# =========================

class WeightCalibrator:
    """
    Auto-tunes module weights by measuring how well each module's per-number
    scores correlate with actual outcomes over a recent walk-forward window.

    For each module, computes Spearman-like rank correlation between its
    score vector and the actual hit vector (1 = appeared, 0 = didn't) over
    a validation window. Higher correlation → higher weight.

    Returns a blended weight dict that merges the calibrated values with the
    strategy profile weights (BASE or JACKPOT), so the calibration always
    stays within sensible bounds.
    """

    def __init__(self, n_max: int, validation_window: int = 60, min_weight: float = 0.1):
        self.n_max = n_max
        self.validation_window = validation_window
        self.min_weight = min_weight
        self.calibrated_weights: Dict[str, float] = {}

    def _rank_corr(self, scores: np.ndarray, hits: np.ndarray) -> float:
        """Simple Pearson correlation between score vector and hit vector."""
        if scores.std() < 1e-8:
            return 0.0
        n = len(scores)
        sm = scores.mean(); hm = hits.mean()
        cov = float(np.mean((scores - sm) * (hits - hm)))
        denom = float(scores.std() * hits.std())
        return cov / denom if denom > 1e-8 else 0.0

    def calibrate(
        self,
        draws: List[Draw],
        strategy_weights: Dict[str, float],
        blend: float = 0.45,
    ) -> Dict[str, float]:
        """
        Run calibration over validation_window most-recent draws.
        blend=0.45 → 45% calibrated, 55% strategy profile.
        """
        n = self.n_max
        val = draws[-self.validation_window:] if len(draws) > self.validation_window else draws
        if len(val) < 10:
            return strategy_weights

        freq_corrs, recency_corrs, pattern_corrs = [], [], []

        for t in range(1, len(val)):
            hist = val[:t]
            target_set = set(val[t].numbers)
            hit_vec = np.array([1.0 if i in target_set else 0.0 for i in range(1, n + 1)])

            # Frequency module signal
            fmod = FrequencyRecencyModule(n)
            fmod.fit(hist)
            freq_corrs.append(self._rank_corr(fmod.freq[1:], hit_vec))

            # Recency/temporal signal
            tmod = TemporalMomentumModule(n)
            tmod.fit(hist)
            recency_corrs.append(self._rank_corr(tmod.recency_score[1:], hit_vec))

            # Pattern balance acts as penalty — measure how well balanced tickets hit
            # (we approximate with a neutral 0 since it's a ticket-level not number-level signal)

        def safe_mean(lst):
            valid = [x for x in lst if not math.isnan(x)]
            return float(np.mean(valid)) if valid else 0.0

        avg_freq = max(0.0, safe_mean(freq_corrs))
        avg_rec = max(0.0, safe_mean(recency_corrs))

        # Normalise so they sum to the same total as strategy weights
        total_signal = avg_freq + avg_rec + 1e-6
        scale = (strategy_weights.get("freq", 1.0) + strategy_weights.get("recency", 0.7)) / total_signal

        cal = dict(strategy_weights)
        cal["freq"] = avg_freq * scale
        cal["recency"] = avg_rec * scale

        # Blend calibrated with strategy profile
        blended = {}
        for k in strategy_weights:
            sw = strategy_weights[k]
            cv = cal.get(k, sw)
            blended[k] = max(self.min_weight, (1 - blend) * sw + blend * cv)

        self.calibrated_weights = blended
        return blended


# =========================
# 12c. GREEDY BEAM TICKET SEARCH
# =========================

def greedy_beam_tickets(
    num_probs: np.ndarray,
    pair_mod: PairTripletModule,
    k: int,
    beam_width: int = 80,
    pattern_mod: Optional[PatternBalanceModule] = None,
    repeat_mod: Optional[RepeatSkipModule] = None,
    temporal_mod: Optional[TemporalMomentumModule] = None,
    markov_mod: Optional["MarkovModule"] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[int]]]:
    """
    Greedy beam search for high-scoring tickets.

    Rather than randomly sampling tickets (Monte Carlo), this deterministically
    constructs tickets by expanding the best partial solutions at each step.
    At each step we extend every beam member with every remaining number,
    keeping only the top beam_width partial tickets.

    This is far more efficient at finding high-scoring combinations because
    it focuses compute on the most promising regions of the search space.
    Returns up to beam_width complete tickets.
    """
    if weights is None:
        weights = BASE_WEIGHTS

    freq_w = weights.get("freq", 1.0)
    recency_w = weights.get("recency", 0.7)
    pattern_w = weights.get("pattern", 0.6)
    crowd_w = weights.get("crowd", 0.9)

    log_probs = np.log(num_probs + 1e-15)
    n_max = len(num_probs) - 1
    all_numbers = list(range(1, n_max + 1))

    # Seed beam: top beam_width single-number partial tickets
    base_scores = []
    for n in all_numbers:
        s = freq_w * log_probs[n]
        if repeat_mod is not None:
            s += repeat_mod.number_bonus(n)
        if temporal_mod is not None:
            s += recency_w * temporal_mod.number_bonus(n)
        if markov_mod is not None:
            s += markov_mod.number_bonus(n)
        base_scores.append((s, [n]))

    base_scores.sort(key=lambda x: x[0], reverse=True)
    beam: List[Tuple[float, List[int]]] = base_scores[:beam_width]

    for step in range(k - 1):
        candidates: List[Tuple[float, List[int]]] = []
        for score, partial in beam:
            partial_set = set(partial)
            remaining = [n for n in all_numbers if n not in partial_set]
            for n in remaining:
                new_score = score + freq_w * log_probs[n]
                if repeat_mod is not None:
                    new_score += repeat_mod.number_bonus(n)
                if temporal_mod is not None:
                    new_score += recency_w * temporal_mod.number_bonus(n)
                if markov_mod is not None:
                    new_score += markov_mod.number_bonus(n)
                # Add pair/triplet bonuses for the new number with all partial members
                for p in partial:
                    new_score += pair_mod.pair_bonus(p, n)
                candidates.append((new_score, partial + [n]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    # Final scoring pass: add pattern penalty and crowding penalty
    results: List[Tuple[float, List[int]]] = []
    for raw_score, ticket in beam:
        final_score = raw_score
        final_score += crowding_penalty(ticket, crowd_w)
        if pattern_mod is not None:
            final_score += pattern_w * pattern_mod.ticket_penalty(ticket)
        results.append((final_score, sorted(ticket)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# =========================
# 13. HIGH-LEVEL PIPELINE
# =========================

def run_full_machine(
    draws: List[Draw],
    n_max: int,
    k: int,
    n_samples: int = 50000,
    top_n: int = 50,
    jackpot: float | None = None,
    run_backtest: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates:
      1) probability calculation
      2) ticket generation
      3) backtest
      4) EV + bankroll sim (Module D)
    """
    if not draws:
        raise ValueError("No draws provided")

    # ---------- JACKPOT HANDLING ----------
    if jackpot is None:
        print("Fetching current Lotto jackpot...")
        scraped = fetch_current_lotto_jackpot(default=None)

        if scraped is None:
            print("\n[!] Could not detect jackpot automatically.")
            print("Please type the current Lotto jackpot amount in Rands.")
            print("Example: 35000000 for R35 million")

            user_input = input("Enter jackpot amount (Rands): ").strip()
            try:
                jackpot = float(user_input.replace(",", ""))
            except:
                print("Invalid input. Defaulting to R5,000,000.")
                jackpot = 5_000_000.0
        else:
            jackpot = scraped

    print(f"Using Lotto jackpot for EV: R{jackpot:,.2f}")

    # --------- CORE MODULES ---------
    freq_mod = FrequencyRecencyModule(n_max)
    freq_mod.fit(draws)

    pair_mod = PairTripletModule(n_max)
    pair_mod.fit(draws)

    bias_mod = BiasModule(n_max)
    bias_mod.fit(draws)

    voodoo_mod = VoodooModule(n_max)
    voodoo_mod.fit(draws)

    ai_mod = MetaAIModule(n_max)
    ai_mod.fit(draws)

    pattern_mod = PatternBalanceModule(n_max)
    pattern_mod.fit(draws)

    repeat_mod = RepeatSkipModule(n_max)
    repeat_mod.fit(draws)

    temporal_mod = TemporalMomentumModule(n_max)
    temporal_mod.fit(draws)

    # --------- EV MODULE ---------
    prize_table = {
        3: 50.0,
        4: 500.0,
        5: 50_000.0,
    }
    ticket_price = 5.0
    ev_mod = EVModule(n_max, k, ticket_price, prize_table)
    intensity = ev_mod.play_intensity(jackpot)

    effective_samples = int(max(5_000, min(n_samples * intensity, 100_000)))

    print(f"EV intensity factor: {intensity:.3f}")
    print(f"Effective samples used: {effective_samples}")

    # --------- MODULE E+: CHOOSE STRATEGY PROFILE ---------
    mode, alpha, weights = choose_strategy_weights(intensity)
    print()
    print("=== Strategy profile (Module E+) ===")
    print(f"Mode: {mode}")
    print(f"Blend alpha (0=Hybrid, 1=Jackpot): {alpha:.3f}")
    print("Weights used:", weights)

    # --------- HYBRID MOMENTUM (EV-controlled) ---------
    momentum_mod = MomentumModule(n_max)
    momentum_mod.fit(draws, ev_intensity=intensity)

    # --------- MARKOV CHAIN (new) ---------
    markov_mod = MarkovModule(n_max)
    markov_mod.fit(draws)

    # --------- REGIME DETECTION (activated) ---------
    regime_mod = RegimeModule(n_max, n_clusters=3)
    regime_mod.fit(draws)
    current_regime = regime_mod.current_regime
    print(f"Current draw regime cluster: {current_regime}")

    # --------- WEIGHT CALIBRATOR (new) ---------
    calibrator = WeightCalibrator(n_max, validation_window=60)
    weights = calibrator.calibrate(draws, weights, blend=0.45)
    print("Calibrated weights:", {k: round(v, 3) for k, v in weights.items()})

    # --------- NUMBER PROBABILITIES ---------
    comb = NumberCombiner(
        n_max,
        config={
            "freq": 0.5,
            "hot": 0.2,
            "cold": 0.0,
            "bias": 0.3,
            "voodoo": 0.1,
            "ai": 0.1,
        },
    )
    num_probs = comb.build_probs(freq_mod, bias_mod, voodoo_mod, ai_mod, momentum_mod=momentum_mod)

    # --------- TICKET GENERATION (using calibrated E+ weights + beam search) ---------
    top_tickets = generate_top_tickets(
        num_probs,
        pair_mod,
        k=k,
        n_samples=effective_samples,
        top_n=top_n,
        crowd_weight=weights.get("crowd", 0.9),
        use_triplets=True,
        pattern_mod=pattern_mod,
        repeat_mod=repeat_mod,
        temporal_mod=temporal_mod,
        markov_mod=markov_mod,
        weights=weights,
    )

    print("\nTOP GENERATED TICKETS:")
    for score, ticket in top_tickets[:10]:
        print(f"  {ticket} (score={score:.4f})")

    # --------- BACKTEST (A/B/C) ---------
    bt = None
    ev_data = None
    sim_data = None

    if run_backtest:
        bt = backtest_lotto(
            draws,
            n_max=n_max,
            k=k,
            window=150,
            n_samples_per_step=2000,
            top_n_per_step=20,
        )

        # --------- MODULE D: EV + BANKROLL SIMULATION ---------
        assumed_jackpot = jackpot if jackpot else 5_000_000.0

        payouts = {
            3: 50.0,
            4: 1_000.0,
            5: 50_000.0,
            6: assumed_jackpot,
        }

        print("\n=== Module D: EV per ticket (rough, based on hit histograms) ===")
        print(f"Ticket price: R{ticket_price:.2f}")
        print("Assumed payouts (hits -> R):")
        for h in sorted(payouts.keys()):
            print(f"  {h} hits: R{payouts[h]:,.2f}")
        print("NOTE: These are placeholders - update to match real typical payouts.\n")

        model_ev_net, model_ev_gross, model_probs = compute_ev_per_ticket(
            bt["model_hist"], payouts, ticket_price
        )
        rand_ev_net, rand_ev_gross, rand_probs = compute_ev_per_ticket(
            bt["random_hist"], payouts, ticket_price
        )

        print(f"Model gross EV per ticket:  R{model_ev_gross:,.2f}")
        print(f"Model NET EV per ticket:    R{model_ev_net:,.2f}")
        print(f"Random gross EV per ticket: R{rand_ev_gross:,.2f}")
        print(f"Random NET EV per ticket:   R{rand_ev_net:,.2f}")
        print("Reminder: real lotteries are designed to be negative EV; "
              "this is mostly useful for comparing model vs random.\n")

        # --------- Bankroll sim over 1 year (2 draws/week -> 104 draws) ---------
        starting_bankroll = 1000.0
        tickets_per_draw = 1
        draws_per_year = 104

        sim_model = simulate_bankroll(
            model_probs,
            payouts,
            ticket_price,
            starting_bankroll=starting_bankroll,
            tickets_per_draw=tickets_per_draw,
            n_draws=draws_per_year,
            n_runs=2000,
        )

        sim_random = simulate_bankroll(
            rand_probs,
            payouts,
            ticket_price,
            starting_bankroll=starting_bankroll,
            tickets_per_draw=tickets_per_draw,
            n_draws=draws_per_year,
            n_runs=2000,
        )

        def print_sim(name, s):
            print(f"{name}:")
            print(f"  Starting bankroll: R{s['starting_bankroll']:,.2f}")
            print(f"  Draws: {s['n_draws']} | Tickets/draw: {s['tickets_per_draw']}")
            print(f"  Avg final bankroll: R{s['avg']:,.2f}")
            print(f"  10th percentile:    R{s['p10']:,.2f}")
            print(f"  Median:             R{s['p50']:,.2f}")
            print(f"  90th percentile:    R{s['p90']:,.2f}")
            print(f"  Min / Max:          R{s['min']:,.2f} .. R{s['max']:,.2f}\n")

        print("\n=== Module D: 1-year bankroll simulation (very rough) ===")
        print_sim("Model strategy", sim_model)
        print_sim("Random strategy", sim_random)

        print("IMPORTANT: This is a toy simulation based on historical hit patterns "
              "and rough prize assumptions. It cannot predict future draws or "
              "guarantee profitability.\n")

        ev_data = {
            "model": {"net": model_ev_net, "gross": model_ev_gross},
            "random": {"net": rand_ev_net, "gross": rand_ev_gross},
        }
        sim_data = {
            "model": sim_model,
            "random": sim_random,
        }

    return {
        "num_probs": num_probs,
        "top_tickets": top_tickets,
        "ev_intensity": intensity,
        "n_samples_effective": effective_samples,
        "jackpot_used": jackpot,
        "hot_numbers": temporal_mod.get_hot_numbers(10),
        "cold_numbers": temporal_mod.get_cold_numbers(10),
        "backtest": bt,
        "ev": ev_data,
        "sim": sim_data,
    }


# =========================
# 14. BACKTEST FUNCTION
# =========================

def backtest_lotto(
    draws: List[Draw],
    n_max: int,
    k: int,
    window: int = 150,
    n_samples_per_step: int = 2000,
    top_n_per_step: int = 20,
) -> Dict[str, Any]:
    """
    Backtest: for each historical draw after the initial window,
    train on the previous `window` draws, generate tickets, and
    compare our best ticket vs a random generator.

    Returns a dict with histograms so Module D can do EV + bankroll sims.
    """
    total_draws = len(draws)
    print("\n=== Lotto Backtest ===")
    print(f"Total draws: {total_draws}, using window: {window}")
    print(f"Evaluating draws {window} .. {total_draws - 1}")
    print(f"n_samples per step: {n_samples_per_step}, top_n tickets per step: {top_n_per_step}\n")

    model_best_hits_list = []
    random_best_hits_list = []
    top_hit_examples = []

    for idx in range(window, total_draws):
        window_draws = draws[idx - window : idx]
        target = draws[idx]
        actual = set(target.numbers)

        freq_mod = FrequencyRecencyModule(n_max)
        freq_mod.fit(window_draws)

        pair_mod = PairTripletModule(n_max)
        pair_mod.fit(window_draws)

        bias_mod = BiasModule(n_max)
        bias_mod.fit(window_draws)

        voodoo_mod = VoodooModule(n_max)
        voodoo_mod.fit(window_draws)

        ai_mod = MetaAIModule(n_max)
        ai_mod.fit(window_draws)

        pattern_mod = PatternBalanceModule(n_max)
        pattern_mod.fit(window_draws)

        repeat_mod = RepeatSkipModule(n_max)
        repeat_mod.fit(window_draws)

        temporal_mod = TemporalMomentumModule(n_max)
        temporal_mod.fit(window_draws)

        momentum_mod = MomentumModule(n_max)
        momentum_mod.fit(window_draws, ev_intensity=0.5)

        comb = NumberCombiner(
            n_max,
            config={
                "freq": 0.5,
                "hot": 0.2,
                "cold": 0.0,
                "bias": 0.3,
                "voodoo": 0.1,
                "ai": 0.1,
            },
        )
        probs = comb.build_probs(freq_mod, bias_mod, voodoo_mod, ai_mod, momentum_mod=momentum_mod)

        model_tickets = generate_top_tickets(
            probs,
            pair_mod,
            k=k,
            n_samples=n_samples_per_step,
            top_n=top_n_per_step,
            crowd_weight=0.5,
            use_triplets=True,
            pattern_mod=pattern_mod,
            repeat_mod=repeat_mod,
            temporal_mod=temporal_mod,
        )

        random_tickets = [sample_uniform_ticket(n_max, k) for _ in range(top_n_per_step)]

        def best_hits_detail(tickets, is_model=False):
            best = 0
            best_ticket = []
            best_matched = set()
            for t in tickets:
                if is_model:
                    _, nums = t
                else:
                    nums = t
                matched = actual.intersection(nums)
                hits = len(matched)
                if hits > best:
                    best = hits
                    best_ticket = sorted(nums)
                    best_matched = matched
            return best, best_ticket, best_matched

        mb, mb_ticket, mb_matched = best_hits_detail(model_tickets, is_model=True)
        rb, _, _ = best_hits_detail(random_tickets, is_model=False)
        model_best_hits_list.append(mb)
        random_best_hits_list.append(rb)

        if mb >= 3:
            top_hit_examples.append({
                "hits": mb,
                "ticket": mb_ticket,
                "actual": sorted(actual),
                "matched": sorted(mb_matched),
                "draw_id": target.draw_id,
                "date": target.date,
            })

        if (idx - window) % 25 == 0:
            print(f"Draw index {idx}/{total_draws - 1} | model best hits: {mb} | random best: {rb}")

    n_eval = len(model_best_hits_list)
    avg_model = sum(model_best_hits_list) / n_eval if n_eval else 0.0
    avg_random = sum(random_best_hits_list) / n_eval if n_eval else 0.0

    def make_hist(hit_list):
        hist = [0] * 7
        for h in hit_list:
            if 0 <= h <= 6:
                hist[h] += 1
        return hist

    model_hist = make_hist(model_best_hits_list)
    random_hist = make_hist(random_best_hits_list)

    print("\n=== Backtest Summary (Model vs Random) ===")
    print(f"Draws evaluated: {n_eval}")
    print(f"Average BEST hits per draw (model):  {avg_model:.3f}")
    print(f"Average BEST hits per draw (random): {avg_random:.3f}\n")

    def print_hist(name, hist):
        print(f"Hit histogram ({name}) [0..6 hits]:")
        line = []
        for h, c in enumerate(hist):
            line.append(f"{h}:{c}")
        print("  " + ", ".join(line))

    print_hist("model", model_hist)
    print_hist("random", random_hist)

    def fraction_at_least(hist, threshold):
        n = sum(hist)
        if n == 0:
            return 0.0
        cnt = sum(hist[threshold:])
        return cnt / n

    print("\nFraction of draws with at least N hits (model):")
    for N in [3, 4, 5]:
        print(f"  >={N} hits: {fraction_at_least(model_hist, N):.4f}")

    print("\nFraction of draws with at least N hits (random):")
    for N in [3, 4, 5]:
        print(f"  >={N} hits: {fraction_at_least(random_hist, N):.4f}")

    print("\n=== End of Backtest ===\n")

    top_hit_examples.sort(key=lambda x: (-x["hits"], -x["draw_id"]))

    return {
        "n_eval": n_eval,
        "model_hist": model_hist,
        "random_hist": random_hist,
        "avg_model": avg_model,
        "avg_random": avg_random,
        "top_hit_examples": top_hit_examples[:20],
    }


# =========================
# D. BANKROLL + EV SIMULATION HELPERS
# =========================

def hist_to_probs(hist: List[int]) -> List[float]:
    """Convert hit histogram [0..6] to probability list [0..6]."""
    total = sum(hist)
    if total == 0:
        return [0.0] * len(hist)
    return [c / total for c in hist]


def compute_ev_per_ticket(
    hit_hist: List[int],
    payouts: Dict[int, float],
    ticket_price: float,
) -> Tuple[float, float, List[float]]:
    """
    hit_hist: list length 7 with counts for 0..6 hits (for 'best ticket')
    payouts: dict {hits: payout_in_R}
    ticket_price: cost in R per ticket

    Returns (net_ev, gross_ev, probs_list)
    """
    probs = hist_to_probs(hit_hist)
    gross = 0.0
    for hits, payout in payouts.items():
        if 0 <= hits < len(probs):
            gross += probs[hits] * float(payout)
    net = gross - float(ticket_price)
    return net, gross, probs


def simulate_bankroll(
    hit_probs: List[float],
    payouts: Dict[int, float],
    ticket_price: float,
    starting_bankroll: float = 1000.0,
    tickets_per_draw: int = 1,
    n_draws: int = 104,
    n_runs: int = 2000,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Very simple Monte Carlo bankroll simulator for one ticket strategy.

    hit_probs: list of length 7, probability of 0..6 hits for your ticket.
    payouts:  dict {hits: payout_R}
    ticket_price: float
    starting_bankroll: initial R
    tickets_per_draw: how many tickets you play each draw
    n_draws: number of draws in the horizon (104 ~ 2 draws/week for a year)
    n_runs: number of Monte Carlo runs
    """
    rng = random.Random(seed)
    ending: List[float] = []

    cumulative: List[float] = []
    c = 0.0
    for p in hit_probs:
        c += p
        cumulative.append(c)
    cumulative[-1] = 1.0

    for _ in range(n_runs):
        bank = float(starting_bankroll)
        for _ in range(n_draws):
            cost = tickets_per_draw * ticket_price
            bank -= cost

            for _ in range(tickets_per_draw):
                u = rng.random()
                hits = 0
                for h, cutoff in enumerate(cumulative):
                    if u <= cutoff:
                        hits = h
                        break
                bank += payouts.get(hits, 0.0)

        ending.append(bank)

    ending.sort()
    n = len(ending)
    avg = sum(ending) / n
    p10 = ending[int(0.10 * (n - 1))]
    p50 = ending[int(0.50 * (n - 1))]
    p90 = ending[int(0.90 * (n - 1))]

    return {
        "avg": avg,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "min": ending[0],
        "max": ending[-1],
        "n_runs": n_runs,
        "n_draws": n_draws,
        "tickets_per_draw": tickets_per_draw,
        "starting_bankroll": starting_bankroll,
    }


# =========================
# 15. MAIN (EXAMPLE USAGE)
# =========================

if __name__ == "__main__":
    print("Fetching SA Lotto historical data...")
    csv_file = fetch_sa_lotto_history("draws.csv")
    print("Download complete:", csv_file)

    draws = load_draws_from_csv("draws.csv", n_per_draw=6)

    if not draws:
        print("Still no draws - check the CSV format!")
    else:
        print(f"Loaded {len(draws)} historical draws.")

        n_max = max(max(d.numbers) for d in draws)
        k = len(draws[0].numbers)

        RUN_BACKTEST = True

        result = run_full_machine(
            draws,
            n_max,
            k,
            n_samples=20000,
            top_n=50,
            run_backtest=RUN_BACKTEST,
        )

        print("\n" + "=" * 50)
        print("LOTTO-LAB RUN COMPLETE")
        print("=" * 50)
