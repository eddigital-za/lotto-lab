"""
Lotto Engine - Service layer for Lotto-Lab dashboard
Extracts core functionality from main.py for use by the Dash UI
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from main import (
    Draw,
    load_draws_from_csv,
    FrequencyRecencyModule,
    PairTripletModule,
    BiasModule,
    BayesianPriorModule,
    VoodooModule,
    MarkovModule,
    RegimeModule,
    MetaAIModule,
    PatternBalanceModule,
    RepeatSkipModule,
    TemporalMomentumModule,
    MomentumModule,
    EVModule,
    NumberCombiner,
    WeightCalibrator,
    generate_top_tickets,
    greedy_beam_tickets,
    backtest_lotto,
    hist_to_probs,
    compute_ev_per_ticket,
    simulate_bankroll,
    choose_strategy_weights,
    BASE_WEIGHTS,
    JACKPOT_WEIGHTS,
    EV_INTENSITY_LOW,
    EV_INTENSITY_HIGH,
)

from fetch_sa_lotto import fetch_sa_lotto_history, fetch_current_lotto_jackpot


def load_data(csv_path: str = "draws.csv") -> Tuple[List[Draw], int, int]:
    """Load draws and return (draws, n_max, k)"""
    try:
        fetch_sa_lotto_history(csv_path)
    except:
        pass
    
    draws = load_draws_from_csv(csv_path, n_per_draw=6)
    if not draws:
        return [], 52, 6
    
    draws.sort(key=lambda d: d.draw_id, reverse=True)
    
    n_max = max(max(d.numbers) for d in draws)
    k = len(draws[0].numbers)
    return draws, n_max, k


def get_jackpot_info(jackpot: Optional[float] = None) -> Tuple[float, float, str, float, Dict[str, float]]:
    """
    Get jackpot and compute EV intensity and strategy weights.
    Returns: (jackpot, intensity, mode, alpha, weights)
    """
    if jackpot is None:
        fetched = fetch_current_lotto_jackpot(default=5_000_000.0)
        jackpot = fetched if fetched is not None else 5_000_000.0
    
    prize_table = {3: 50.0, 4: 500.0, 5: 50_000.0}
    ticket_price = 5.0
    ev_mod = EVModule(52, 6, ticket_price, prize_table)
    intensity = ev_mod.play_intensity(float(jackpot))
    
    mode, alpha, weights = choose_strategy_weights(intensity)
    
    return jackpot, intensity, mode, alpha, weights


def generate_predictions(
    draws: List[Draw],
    n_max: int,
    k: int,
    jackpot: float,
    n_samples: int = 20000,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Generate ticket predictions with all module analysis.

    Upgraded pipeline includes:
      - BayesianPriorModule (replaces random VoodooModule)
      - MarkovModule (first-order sequential carry-over)
      - RegimeModule (draw cluster detection)
      - WeightCalibrator (auto-tunes module weights from recent data)
      - greedy beam search merged with Monte Carlo sampling
      - MetaAIModule with 12 engineered features + StandardScaler
    """
    if not draws:
        return {"error": "No draws available"}

    jackpot, intensity, mode, alpha, weights = get_jackpot_info(jackpot)
    effective_samples = int(max(5_000, min(n_samples * intensity, 100_000)))

    # --- Core statistical modules ---
    freq_mod = FrequencyRecencyModule(n_max)
    freq_mod.fit(draws)

    pair_mod = PairTripletModule(n_max)
    pair_mod.fit(draws)

    bias_mod = BiasModule(n_max)
    bias_mod.fit(draws)

    # Bayesian prior replaces VoodooModule
    bayesian_mod = BayesianPriorModule(n_max)
    bayesian_mod.fit(draws)

    # Enriched 12-feature ML classifier
    ai_mod = MetaAIModule(n_max)
    ai_mod.fit(draws)

    pattern_mod = PatternBalanceModule(n_max)
    pattern_mod.fit(draws)

    repeat_mod = RepeatSkipModule(n_max)
    repeat_mod.fit(draws)

    temporal_mod = TemporalMomentumModule(n_max)
    temporal_mod.fit(draws)

    momentum_mod = MomentumModule(n_max)
    momentum_mod.fit(draws, ev_intensity=intensity)

    # --- New modules ---
    markov_mod = MarkovModule(n_max)
    markov_mod.fit(draws)

    regime_mod = RegimeModule(n_max, n_clusters=3)
    regime_mod.fit(draws)
    current_regime = regime_mod.current_regime

    # Auto-calibrate weights from recent data
    calibrator = WeightCalibrator(n_max, validation_window=60)
    calibrated_weights = calibrator.calibrate(draws, weights, blend=0.45)

    # --- Build number probabilities ---
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
    num_probs = comb.build_probs(
        freq_mod, bias_mod, bayesian_mod, ai_mod, momentum_mod=momentum_mod
    )

    # --- Ticket generation: Monte Carlo + greedy beam merged ---
    top_tickets = generate_top_tickets(
        num_probs,
        pair_mod,
        k=k,
        n_samples=effective_samples,
        top_n=top_n,
        crowd_weight=calibrated_weights.get("crowd", 0.9),
        use_triplets=True,
        pattern_mod=pattern_mod,
        repeat_mod=repeat_mod,
        temporal_mod=temporal_mod,
        markov_mod=markov_mod,
        weights=calibrated_weights,
    )

    return {
        "tickets": [(score, ticket) for score, ticket in top_tickets],
        "num_probs": {i: float(num_probs[i]) for i in range(1, n_max + 1)},
        "jackpot": jackpot,
        "intensity": intensity,
        "mode": mode,
        "alpha": alpha,
        "weights": calibrated_weights,
        "calibrated_weights": calibrated_weights,
        "effective_samples": effective_samples,
        "current_regime": current_regime,
        "hot_numbers": [n for n, _ in temporal_mod.get_hot_numbers(10)],
        "cold_numbers": [n for n, _ in temporal_mod.get_cold_numbers(10)],
        "freq_scores": {i: float(freq_mod.freq[i]) for i in range(1, n_max + 1)},
        "recency_scores": {i: float(temporal_mod.recency_score[i]) for i in range(1, n_max + 1)},
        "markov_bonuses": {i: float(markov_mod.number_bonus(i)) for i in range(1, n_max + 1)},
    }


def run_backtest_analysis(
    draws: List[Draw],
    n_max: int,
    k: int,
    window: int = 150,
    n_samples_per_step: int = 2000,
    top_n_per_step: int = 20,
) -> Dict[str, Any]:
    """
    Run backtest and return results with EV/bankroll analysis.
    """
    if len(draws) < window + 10:
        return {"error": "Not enough draws for backtest"}
    
    bt = backtest_lotto(
        draws,
        n_max=n_max,
        k=k,
        window=window,
        n_samples_per_step=n_samples_per_step,
        top_n_per_step=top_n_per_step,
    )
    
    payouts = {
        3: 50.0,
        4: 1_000.0,
        5: 50_000.0,
        6: 5_000_000.0,
    }
    ticket_price = 5.0
    
    model_ev_net, model_ev_gross, model_probs = compute_ev_per_ticket(
        bt["model_hist"], payouts, ticket_price
    )
    rand_ev_net, rand_ev_gross, rand_probs = compute_ev_per_ticket(
        bt["random_hist"], payouts, ticket_price
    )
    
    sim_model = simulate_bankroll(
        model_probs, payouts, ticket_price,
        starting_bankroll=1000.0,
        tickets_per_draw=1,
        n_draws=104,
        n_runs=500,
    )
    
    sim_random = simulate_bankroll(
        rand_probs, payouts, ticket_price,
        starting_bankroll=1000.0,
        tickets_per_draw=1,
        n_draws=104,
        n_runs=500,
    )
    
    return {
        "backtest": bt,
        "model_hist": bt["model_hist"],
        "random_hist": bt["random_hist"],
        "model_ev": {"net": model_ev_net, "gross": model_ev_gross},
        "random_ev": {"net": rand_ev_net, "gross": rand_ev_gross},
        "model_sim": sim_model,
        "random_sim": sim_random,
        "top_hit_examples": bt.get("top_hit_examples", []),
    }


def get_frequency_data(draws: List[Draw], n_max: int) -> Dict[str, Any]:
    """Get frequency analysis data for charts."""
    if not draws:
        return {}
    
    freq_mod = FrequencyRecencyModule(n_max)
    freq_mod.fit(draws)
    
    temporal_mod = TemporalMomentumModule(n_max)
    temporal_mod.fit(draws)
    
    return {
        "frequency": {i: float(freq_mod.freq[i]) for i in range(1, n_max + 1)},
        "recency": {i: float(temporal_mod.recency_score[i]) for i in range(1, n_max + 1)},
        "momentum": {i: float(temporal_mod.momentum_score[i]) for i in range(1, n_max + 1)},
        "hot": [n for n, _ in temporal_mod.get_hot_numbers(15)],
        "cold": [n for n, _ in temporal_mod.get_cold_numbers(15)],
    }


def get_recent_draws(draws: List[Draw], count: int = 10) -> List[Dict[str, Any]]:
    """Get recent draw history (newest first)."""
    recent = draws[:count] if len(draws) >= count else draws
    return [
        {
            "draw_id": d.draw_id,
            "date": d.date,
            "numbers": d.numbers,
            "jackpot": d.jackpot,
        }
        for d in recent
    ]
