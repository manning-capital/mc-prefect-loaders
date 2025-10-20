"""
Behavioral tests for Geometric Brownian Motion.

These tests verify that GBM exhibits the correct qualitative behavior:
- Positive mu → upward drift
- Negative mu → downward drift
- Zero mu → no drift (martingale)
- Higher sigma → higher volatility
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import numpy as np
from scipy import stats

from src.attributes.stochastic_models import DELTA_T, GBMParams, GeometricBrownianMotion

from tests.utils import assert_within_tolerance, TOLERANCE, set_random_seed

# Test configuration
N_POINTS = 100  # Number of time steps
N_SIMULATED = 10000  # Number of paths for averaging
INITIAL_PRICE = 100.0

# ============================================================================
# Drift Direction Tests
# ============================================================================


def test_gbm_positive_drift():
    """
    Test that GBM with positive mu drifts upward on average.

    With mu > 0, E[S_t] = S_0 * exp(mu*t) grows exponentially.
    """
    set_random_seed(42)

    mu = 0.05  # 5% drift per time unit (more reasonable)
    sigma = 0.2

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate many paths
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Calculate average final price across all paths
    average_final_price = np.mean(simulated_prices[:, -1])

    # Expected final value: S_0 * exp(mu*T)
    T = N_POINTS * DELTA_T
    expected_final_price = INITIAL_PRICE * np.exp(mu * T)

    # Average should drift up from initial
    assert average_final_price > INITIAL_PRICE, (
        f"With positive mu={mu}, prices should drift up. "
        f"Initial: {INITIAL_PRICE}, Final avg: {average_final_price}"
    )

    # Check it's in the right ballpark (±40% tolerance for stochastic process with finite samples)
    assert (
        0.6 * expected_final_price < average_final_price < 1.4 * expected_final_price
    ), (
        f"Average final price {average_final_price:.2f} too far from expected {expected_final_price:.2f}"
    )


def test_gbm_negative_drift():
    """
    Test that GBM with negative mu drifts downward on average.

    With mu < 0, E[S_t] = S_0 * exp(mu*t) decays exponentially.
    """
    set_random_seed(43)

    mu = -0.02  # -2% drift per time unit (more moderate decay)
    sigma = 0.15

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate many paths
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Calculate average final price
    average_final_price = np.mean(simulated_prices[:, -1])

    # Expected final value
    T = N_POINTS * DELTA_T
    expected_final_price = INITIAL_PRICE * np.exp(mu * T)

    # Average should drift down from initial
    assert average_final_price < INITIAL_PRICE, (
        f"With negative mu={mu}, prices should drift down. "
        f"Initial: {INITIAL_PRICE}, Final avg: {average_final_price}"
    )

    # Check it's in the right ballpark (±40% tolerance for stochastic process with finite samples)
    assert (
        0.6 * expected_final_price < average_final_price < 1.4 * expected_final_price
    ), (
        f"Average final price {average_final_price:.2f} too far from expected {expected_final_price:.2f}"
    )


def test_gbm_zero_drift():
    """
    Test that GBM with mu=0 stays near initial price on average.

    With mu = 0, E[S_t] = S_0 for all t (martingale property).
    """
    set_random_seed(44)

    mu = 0.0
    sigma = 0.2

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate many paths
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Calculate average final price
    average_final_price = np.mean(simulated_prices[:, -1])

    # With mu=0, expected value should stay at initial price
    # Allow ±40% tolerance due to finite sample and volatility effects
    assert 0.6 * INITIAL_PRICE < average_final_price < 1.4 * INITIAL_PRICE, (
        f"With mu=0, average final price should be near initial. "
        f"Initial: {INITIAL_PRICE}, Final avg: {average_final_price}"
    )


# ============================================================================
# Volatility Tests
# ============================================================================


def test_gbm_higher_sigma_increases_volatility():
    """
    Test that higher sigma leads to higher realized volatility.

    Compare two GBM processes with same mu but different sigma values.
    The one with higher sigma should have higher standard deviation of log-returns.
    """
    set_random_seed(45)

    mu = 0.05
    sigma_low = 0.1
    sigma_high = 0.4  # 4x higher volatility

    # Simulate with low sigma
    params_low = GBMParams(mu=mu, sigma=sigma_low)
    gbm_low = GeometricBrownianMotion(params=params_low)
    prices_low = gbm_low.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Simulate with high sigma
    params_high = GBMParams(mu=mu, sigma=sigma_high)
    gbm_high = GeometricBrownianMotion(params=params_high)
    prices_high = gbm_high.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Calculate realized volatility (std of log-returns) for each path, then average
    vol_low = []
    vol_high = []

    for i in range(N_SIMULATED):
        log_returns_low = np.diff(np.log(prices_low[i, :]))
        log_returns_high = np.diff(np.log(prices_high[i, :]))

        vol_low.append(np.std(log_returns_low))
        vol_high.append(np.std(log_returns_high))

    avg_vol_low = np.mean(vol_low)
    avg_vol_high = np.mean(vol_high)

    # Higher sigma should produce significantly higher realized volatility
    assert avg_vol_high > avg_vol_low, (
        f"Higher sigma should produce higher volatility. "
        f"Low: {avg_vol_low:.4f}, High: {avg_vol_high:.4f}"
    )

    # The ratio should be approximately sigma_high/sigma_low
    vol_ratio = avg_vol_high / avg_vol_low
    expected_ratio = sigma_high / sigma_low

    # Allow ±30% tolerance on the ratio
    assert 0.7 * expected_ratio < vol_ratio < 1.3 * expected_ratio, (
        f"Volatility ratio {vol_ratio:.2f} too far from expected {expected_ratio:.2f}"
    )


def test_gbm_volatility_scales_with_sigma():
    """
    Test that realized volatility scales approximately linearly with sigma.

    Test three different sigma values and verify the scaling relationship.
    """
    set_random_seed(46)

    mu = 0.0  # Zero drift to isolate volatility effect
    sigmas = [0.1, 0.2, 0.3]
    realized_vols = []

    for sigma in sigmas:
        params = GBMParams(mu=mu, sigma=sigma)
        gbm = GeometricBrownianMotion(params=params)
        prices = gbm.simulate(N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE)

        # Calculate average realized volatility across paths
        vols = []
        for i in range(N_SIMULATED):
            log_returns = np.diff(np.log(prices[i, :]))
            vols.append(np.std(log_returns))

        realized_vols.append(np.mean(vols))

    # Check that realized volatility increases with sigma
    assert realized_vols[1] > realized_vols[0], "Volatility should increase with sigma"
    assert realized_vols[2] > realized_vols[1], "Volatility should increase with sigma"

    # Check approximate linear scaling: vol should be proportional to sigma
    # Calculate ratios
    ratio_01 = realized_vols[1] / realized_vols[0]
    ratio_02 = realized_vols[2] / realized_vols[0]

    expected_ratio_01 = sigmas[1] / sigmas[0]  # Should be 2.0
    expected_ratio_02 = sigmas[2] / sigmas[0]  # Should be 3.0

    # Allow ±25% tolerance
    assert 0.75 * expected_ratio_01 < ratio_01 < 1.25 * expected_ratio_01, (
        f"Volatility ratio {ratio_01:.2f} too far from expected {expected_ratio_01:.2f}"
    )
    assert 0.75 * expected_ratio_02 < ratio_02 < 1.25 * expected_ratio_02, (
        f"Volatility ratio {ratio_02:.2f} too far from expected {expected_ratio_02:.2f}"
    )


# ============================================================================
# Sanity Tests
# ============================================================================


def test_gbm_positive_prices():
    """
    Ensure that GBM maintains positive prices at all times.

    GBM is designed to keep prices positive by construction.
    """
    set_random_seed(47)

    mu = 0.1
    sigma = 0.5  # High volatility stress test

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Assert all prices are positive
    assert np.all(simulated_prices > 0), (
        f"GBM produced non-positive prices. Min: {np.min(simulated_prices)}"
    )

    # Assert no NaN or Inf values
    assert np.all(np.isfinite(simulated_prices)), "GBM produced NaN or Inf values"


# ============================================================================
# Statistical Property Tests
# ============================================================================


def test_gbm_log_returns_distribution():
    """
    Verify that log-returns follow a normal distribution.

    For GBM, log-returns should be normally distributed with:
    - Mean: (mu - 0.5*sigma²)*dt
    - Variance: sigma²*dt
    """
    set_random_seed(48)

    mu = 0.05
    sigma = 0.2

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate a single long path
    simulated_prices = gbm.simulate(N=10000, N_simulated=1, X_0=INITIAL_PRICE)[0]

    # Calculate log-returns
    log_returns = np.diff(np.log(simulated_prices))

    # Theoretical parameters
    expected_mean = (mu - 0.5 * sigma**2) * DELTA_T
    expected_std = sigma * np.sqrt(DELTA_T)

    # Test normality using Kolmogorov-Smirnov test
    # Standardize the log-returns
    standardized = (log_returns - expected_mean) / expected_std
    ks_statistic, p_value = stats.kstest(standardized, "norm")

    # p-value > 0.05 suggests data is consistent with normal distribution
    assert p_value > 0.05, (
        f"Log-returns do not follow normal distribution. KS test p-value: {p_value:.4f}"
    )

    # Also check that mean and std are close to theoretical
    observed_mean = np.mean(log_returns)
    observed_std = np.std(log_returns, ddof=1)

    assert abs(observed_mean - expected_mean) < 3 * expected_std / np.sqrt(
        len(log_returns)
    ), (
        f"Mean of log-returns {observed_mean:.6f} differs from expected {expected_mean:.6f}"
    )

    assert 0.9 * expected_std < observed_std < 1.1 * expected_std, (
        f"Std of log-returns {observed_std:.6f} differs from expected {expected_std:.6f}"
    )


def test_gbm_variance_grows_linearly():
    """
    Test that variance of log-prices grows linearly with time.

    For GBM: Var[log(S_t)] = sigma²*t
    """
    set_random_seed(49)

    mu = 0.03
    sigma = 0.25

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate many paths
    N = 200
    simulated_prices = gbm.simulate(N=N, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE)

    # Calculate log-prices
    log_prices = np.log(simulated_prices)

    # Check variance at different time points
    time_points = [50, 100, 150, 200]
    variances = []

    for t in time_points:
        # Variance of log(S_t) across paths
        variance = np.var(log_prices[:, t - 1], ddof=1)
        variances.append(variance)

    # Theoretical variances
    theoretical_variances = [sigma**2 * (t - 1) * DELTA_T for t in time_points]

    # Check that observed variances are close to theoretical (±30% tolerance)
    for obs, theo, t in zip(variances, theoretical_variances, time_points):
        assert 0.7 * theo < obs < 1.3 * theo, (
            f"At time {t}, variance {obs:.4f} differs from theoretical {theo:.4f}"
        )


def test_gbm_price_lognormal():
    """
    Verify that prices at a fixed time follow a log-normal distribution.

    For GBM, S_t follows a log-normal distribution with:
    - log(S_t) ~ N(log(S_0) + (mu - 0.5*sigma²)*t, sigma²*t)
    """
    set_random_seed(50)

    mu = 0.05
    sigma = 0.3
    T = 50  # Time point to test

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate many paths
    simulated_prices = gbm.simulate(N=T + 1, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE)

    # Get prices at time T
    prices_at_T = simulated_prices[:, T]

    # For log-normal distribution, log(S_t) should be normal
    log_prices = np.log(prices_at_T)

    # Theoretical parameters for log(S_T)
    expected_mean = np.log(INITIAL_PRICE) + (mu - 0.5 * sigma**2) * T * DELTA_T
    expected_std = sigma * np.sqrt(T * DELTA_T)

    # Test log-normality using KS test on log-transformed data
    standardized = (log_prices - expected_mean) / expected_std
    ks_statistic, p_value = stats.kstest(standardized, "norm")

    assert p_value > 0.05, (
        f"Log-prices do not follow normal distribution (prices not log-normal). "
        f"KS test p-value: {p_value:.4f}"
    )


# ============================================================================
# Fit Method Accuracy Tests
# ============================================================================


def test_gbm_parameter_recovery_standard():
    """
    Test GBM parameter recovery with typical parameters.

    Parameters:
    - mu = 0.05 (moderate positive drift)
    - sigma = 0.2 (moderate volatility)
    """
    set_random_seed(51)

    # True parameters
    mu_true = 0.05
    sigma_true = 0.2

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_parameter_recovery_zero_drift():
    """
    Test GBM parameter recovery with zero drift (martingale case).

    Parameters:
    - mu = 0.0 (no drift)
    - sigma = 0.2 (moderate volatility)
    """
    set_random_seed(52)

    # True parameters
    mu_true = 0.0
    sigma_true = 0.2

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_parameter_recovery_negative_drift():
    """
    Test GBM parameter recovery with negative drift.

    Parameters:
    - mu = -0.05 (negative drift)
    - sigma = 0.2 (moderate volatility)
    """
    set_random_seed(53)

    # True parameters
    mu_true = -0.05
    sigma_true = 0.2

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_parameter_recovery_high_volatility():
    """
    Test GBM parameter recovery with high volatility.

    Parameters:
    - mu = 0.05 (moderate positive drift)
    - sigma = 0.5 (high volatility)
    """
    set_random_seed(54)

    # True parameters
    mu_true = 0.05
    sigma_true = 0.5

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_parameter_recovery_low_volatility():
    """
    Test GBM parameter recovery with low volatility.

    Parameters:
    - mu = 0.05 (moderate positive drift)
    - sigma = 0.05 (low volatility)
    """
    set_random_seed(55)

    # True parameters
    mu_true = 0.05
    sigma_true = 0.05

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_parameter_recovery_high_drift():
    """
    Test GBM parameter recovery with high positive drift.

    Parameters:
    - mu = 0.20 (high positive drift)
    - sigma = 0.2 (moderate volatility)
    """
    set_random_seed(56)

    # True parameters
    mu_true = 0.20
    sigma_true = 0.2

    # Create model and simulate
    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE
    )

    # Store the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
        fitted_params = gbm_fit.fit(simulated_prices[i, :])
        mu_fitted[i] = fitted_params.mu
        sigma_fitted[i] = fitted_params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted_mean, sigma_true, tolerance=TOLERANCE)


def test_gbm_fit_different_sample_sizes():
    """
    Verify that fit accuracy improves with larger sample sizes.

    Fit error should decrease as sample size increases.
    """
    set_random_seed(52)

    mu_true = 0.06
    sigma_true = 0.2

    params = GBMParams(mu=mu_true, sigma=sigma_true)
    gbm = GeometricBrownianMotion(params=params)

    sample_sizes = [500, 2000, 5000]
    sigma_errors = []

    for n in sample_sizes:
        # Simulate and fit multiple times
        errors = []
        for i in range(20):
            simulated_prices = gbm.simulate(N=n, N_simulated=1, X_0=INITIAL_PRICE)[0]

            gbm_fit = GeometricBrownianMotion(params=GBMParams(mu=0, sigma=0))
            fitted_params = gbm_fit.fit(simulated_prices)

            # Calculate relative error for sigma
            error = abs(fitted_params.sigma - sigma_true) / sigma_true
            errors.append(error)

        sigma_errors.append(np.mean(errors))

    # Errors should generally decrease with sample size
    # Check that largest sample has smaller error than smallest
    assert sigma_errors[2] < sigma_errors[0], (
        f"Fit error should decrease with sample size. Errors: {sigma_errors}"
    )


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_gbm_very_low_volatility():
    """
    Test behavior with sigma close to zero.

    Should behave like deterministic exponential growth: S_t ≈ S_0 * exp(mu*t)
    """
    set_random_seed(53)

    mu = 0.05
    sigma = 0.001  # Very low volatility

    params = GBMParams(mu=mu, sigma=sigma)
    gbm = GeometricBrownianMotion(params=params)

    # Simulate paths
    simulated_prices = gbm.simulate(N=N_POINTS, N_simulated=100, X_0=INITIAL_PRICE)

    # Expected deterministic value
    T = N_POINTS * DELTA_T
    expected_final = INITIAL_PRICE * np.exp(mu * T)

    # All paths should be very close to deterministic path
    final_prices = simulated_prices[:, -1]
    relative_deviations = np.abs(final_prices - expected_final) / expected_final

    # With very low volatility, deviations should be small
    # Even with sigma=0.001, over 100 steps we accumulate some variance
    assert np.max(relative_deviations) < 0.10, (
        f"With low volatility, max deviation {np.max(relative_deviations):.4f} should be < 10%"
    )


def test_gbm_extreme_drift():
    """
    Test with very high positive and negative mu values.

    Should still produce valid simulations without numerical issues.
    """
    set_random_seed(54)

    # Test high positive drift
    params_high = GBMParams(mu=0.5, sigma=0.2)
    gbm_high = GeometricBrownianMotion(params=params_high)

    prices_high = gbm_high.simulate(N=50, N_simulated=100, X_0=INITIAL_PRICE)

    assert np.all(np.isfinite(prices_high)), "High mu produced invalid values"
    assert np.all(prices_high > 0), "High mu produced non-positive prices"
    assert np.mean(prices_high[:, -1]) > INITIAL_PRICE, "High mu should drift up"

    # Test high negative drift
    params_low = GBMParams(mu=-0.3, sigma=0.2)
    gbm_low = GeometricBrownianMotion(params=params_low)

    prices_low = gbm_low.simulate(N=50, N_simulated=100, X_0=INITIAL_PRICE)

    assert np.all(np.isfinite(prices_low)), "Low mu produced invalid values"
    assert np.all(prices_low > 0), "Low mu produced non-positive prices"
    assert np.mean(prices_low[:, -1]) < INITIAL_PRICE, "Low mu should drift down"


def test_gbm_different_initial_prices():
    """
    Verify model works correctly with various initial price levels.

    GBM should scale appropriately with different starting values.
    """
    set_random_seed(55)

    mu = 0.05
    sigma = 0.2
    params = GBMParams(mu=mu, sigma=sigma)

    initial_prices = [0.01, 1.0, 100.0, 10000.0]

    # Use shorter time horizon to reduce variance accumulation
    n_steps = 50

    for X_0 in initial_prices:
        gbm = GeometricBrownianMotion(params=params)
        simulated_prices = gbm.simulate(N=n_steps, N_simulated=100, X_0=X_0)

        # Check basic properties
        assert np.all(simulated_prices > 0), f"Non-positive prices with X_0={X_0}"
        assert np.all(np.isfinite(simulated_prices)), f"Invalid values with X_0={X_0}"

        # Check that scaling is appropriate
        # Expected final value
        T = n_steps * DELTA_T
        expected_final = X_0 * np.exp(mu * T)
        avg_final = np.mean(simulated_prices[:, -1])

        # Should be within reasonable range (wider tolerance due to stochastic nature)
        assert 0.4 * expected_final < avg_final < 2.5 * expected_final, (
            f"With X_0={X_0}, avg final {avg_final:.4f} too far from expected {expected_final:.4f}"
        )
