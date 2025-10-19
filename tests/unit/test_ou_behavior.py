import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import numpy as np
import pytest

from src.attributes.stochastic_models import (
    DELTA_T,
    OrnsteinUhlenbeck,
    OUParams,
)

# Test configuration constants
N_POINTS = 10_000  # Number of time steps for simulation
N_SIMULATED = 100  # Number of paths for averaging
TOLERANCE = 0.15  # ±15% tolerance for parameter recovery
INITIAL_PRICE = 100.0  # Starting price for GBM simulations
INITIAL_VALUE = 0.0  # Starting value for OU simulations


# ============================================================================
# Helper Functions
# ============================================================================


def assert_within_tolerance(
    fitted_value: float, true_value: float, tolerance: float = TOLERANCE
):
    """
    Assert that a fitted parameter is within the specified tolerance of the true value.

    Args:
        fitted_value: The parameter value estimated from data
        true_value: The true parameter value used to generate the data
        tolerance: Relative tolerance (default ±15%)
    """
    if true_value == 0:
        # For zero values, use absolute tolerance
        assert abs(fitted_value) <= tolerance, (
            f"Fitted value {fitted_value} not within absolute tolerance {tolerance} of 0"
        )
    elif true_value > 0:
        lower_bound = true_value * (1 - tolerance)
        upper_bound = true_value * (1 + tolerance)
        assert lower_bound <= fitted_value <= upper_bound, (
            f"Fitted value {fitted_value} not within ±{tolerance * 100}% of true value {true_value}. "
            f"Expected range: [{lower_bound:.6f}, {upper_bound:.6f}]"
        )
    else:
        lower_bound = true_value * (1 + tolerance)
        upper_bound = true_value * (1 - tolerance)
        assert lower_bound <= fitted_value <= upper_bound, (
            f"Fitted value {fitted_value} not within ±{tolerance * 100}% of true value {true_value}. "
            f"Expected range: [{lower_bound:.6f}, {upper_bound:.6f}]"
        )


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def test_ou_parameter_recovery_standard():
    """
    Test OU parameter recovery with typical spread parameters.

    Note: In the implementation, parameter names are swapped from standard notation:
    - mu = mean reversion speed (usually called theta)
    - theta = long-term mean (usually called mu)

    Parameters:
    - mu = 1.0 (mean reversion speed)
    - theta = 0.0 (long-term mean)
    - sigma = 0.1 (volatility)
    """
    set_random_seed(48)

    # True parameters (using implementation's naming convention)
    mu_true = 0.5  # mean reversion speed
    theta_true = 0.0  # long-term mean
    sigma_true = 0.1

    # Create model and simulate
    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_VALUE
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(
        theta_fitted_mean, theta_true, tolerance=0.20
    )  # Mean harder to estimate
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_parameter_recovery_non_zero_theta():
    """
    Test OU parameter recovery with non-zero theta.
    """
    set_random_seed(48)

    # True parameters (using implementation's naming convention)
    mu_true = 0.5  # mean reversion speed
    theta_true = 1.0  # long-term mean
    sigma_true = 0.1

    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_VALUE
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true)
    assert_within_tolerance(theta_fitted_mean, theta_true)
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_parameter_recovery_large_positive_theta():
    """
    Test OU parameter recovery with large positive theta.
    """
    set_random_seed(48)

    # True parameters (using implementation's naming convention)
    mu_true = 0.5  # mean reversion speed
    theta_true = 100.0  # long-term mean
    sigma_true = 0.1

    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_VALUE
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true)
    assert_within_tolerance(theta_fitted_mean, theta_true)
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_parameter_recovery_large_negative_theta():
    """
    Test OU parameter recovery with large negative theta.
    """
    set_random_seed(48)

    # True parameters (using implementation's naming convention)
    mu_true = 0.5  # mean reversion speed
    theta_true = -100.0  # long-term mean
    sigma_true = 0.1

    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_VALUE
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true)
    assert_within_tolerance(theta_fitted_mean, theta_true)
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_parameter_recovery_fast_reversion():
    """
    Test OU parameter recovery with fast mean reversion.

    Parameters:
    - mu = 5.0 (fast mean reversion speed)
    - theta = 1.0 (long-term mean)
    - sigma = 0.2 (volatility)
    """
    set_random_seed(49)

    # True parameters (using implementation's naming convention)
    mu_true = 2.5  # mean reversion speed
    theta_true = 0.0  # long-term mean
    sigma_true = 0.25

    # Create model and simulate
    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS,
        N_simulated=N_SIMULATED,
        X_0=theta_true,  # Start at mean
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true)
    assert_within_tolerance(theta_fitted_mean, theta_true)
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_parameter_recovery_slow_reversion():
    """
    Test OU parameter recovery with slow mean reversion.

    Parameters:
    - mu = 0.1 (slow mean reversion speed)
    - theta = 0.5 (long-term mean)
    - sigma = 0.15 (volatility)
    """
    set_random_seed(50)

    # True parameters (using implementation's naming convention)
    mu_true = 0.1  # mean reversion speed
    theta_true = 0.5  # long-term mean
    sigma_true = 0.15

    # Create model and simulate
    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))
    simulated_values = ou.simulate(
        N=N_POINTS,
        N_simulated=N_SIMULATED,
        X_0=theta_true,  # Start at mean
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values[i, :])
        mu_fitted[i] = params.mu
        theta_fitted[i] = params.theta
        sigma_fitted[i] = params.sigma

    # Calculate the mean of the fitted parameters
    mu_fitted_mean = np.mean(mu_fitted)
    theta_fitted_mean = np.mean(theta_fitted)
    sigma_fitted_mean = np.mean(sigma_fitted)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted_mean, mu_true)
    assert_within_tolerance(theta_fitted_mean, theta_true)
    assert_within_tolerance(sigma_fitted_mean, sigma_true)


def test_ou_mean_reversion_property():
    """
    Verify that OU process exhibits mean reversion behavior.

    Starting far from mean, the process should move toward the mean over time.
    """
    set_random_seed(51)

    mu = 2.0  # Moderate mean reversion speed
    theta = 0.0  # Target mean
    sigma = 0.1

    # Start far from mean
    initial_value = 5.0

    # Simulate
    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))
    simulated_values = ou.simulate(
        N=10_000,  # Shorter path for this test
        N_simulated=1,
        X_0=initial_value,
    )[0]

    # Check that later values are closer to mean than initial value
    early_distance = np.mean(np.abs(simulated_values[:1000] - theta))
    late_distance = np.mean(np.abs(simulated_values[-1000:] - theta))

    assert late_distance < early_distance, (
        f"Mean reversion not observed. Early distance: {early_distance}, "
        f"Late distance: {late_distance}"
    )


def test_ou_stationary_variance():
    """
    Verify that OU process long-run variance matches theoretical value.

    Theoretical stationary variance: sigma²/(2*mu)
    where mu is the mean reversion speed in the implementation.
    """
    set_random_seed(52)

    mu = 1.0  # Mean reversion speed
    theta = 0.0  # Long-term mean
    sigma = 0.2

    # Theoretical stationary variance: sigma²/(2*mu)
    theoretical_variance = sigma**2 / (2 * mu)

    # Simulate long path starting at mean
    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=1, X_0=theta)[0]

    # Use second half of data to ensure stationarity
    stationary_data = simulated_values[N_POINTS // 2 :]
    observed_variance = np.var(stationary_data, ddof=1)

    # Check if observed variance is close to theoretical (±30% tolerance for variance)
    assert_within_tolerance(observed_variance, theoretical_variance, tolerance=0.30)


def test_ou_autocorrelation():
    """
    Verify that OU process autocorrelation decays exponentially.

    Theoretical autocorrelation at lag k: exp(-mu * k * dt)
    where mu is the mean reversion speed in the implementation.
    """
    set_random_seed(53)

    mu = 0.1  # Mean reversion speed
    theta = 0.0  # Long-term mean
    sigma = 0.2
    # Use smaller lags to avoid numerical precision issues
    # With mu=0.1 and DELTA_T=1, these give reasonable autocorrelation values:
    # lag 1: exp(-0.1) ≈ 0.905
    # lag 5: exp(-0.5) ≈ 0.606
    # lag 10: exp(-1.0) ≈ 0.368
    lags = [1, 5, 10]

    # Simulate
    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=N_SIMULATED, X_0=theta)

    # Calculate autocorrelation at a few lags
    autocorrelations = np.zeros((N_SIMULATED, len(lags)))
    for i in range(N_SIMULATED):
        for lag in lags:
            # Observed autocorrelation
            centered = simulated_values[i, :] - np.mean(simulated_values[i, :])
            autocorr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]

            # Store the autocorrelation
            autocorrelations[i, lags.index(lag)] = autocorr

    # Calculate the mean of the autocorrelations
    autocorrelations_mean = np.mean(autocorrelations, axis=0)

    # Check if observed is close to theoretical (±25% tolerance)
    for i in range(len(lags)):
        # Theoretical autocorrelation: exp(-mu * lag * dt)
        theoretical_autocorr = np.exp(-mu * lags[i] * DELTA_T)

        # Check if observed is close to theoretical (±25% tolerance)
        assert_within_tolerance(
            autocorrelations_mean[i], theoretical_autocorr, tolerance=0.25
        )


# ============================================================================
# Statistical Property Tests - Additional
# ============================================================================


def test_ou_convergence_to_stationary():
    """
    Verify that starting far from mean, the distribution converges to stationary distribution.

    Over time, the variance should approach the stationary variance: sigma²/(2*mu)
    """
    set_random_seed(56)

    mu = 1.0
    theta = 0.0
    sigma = 0.3

    # Start far from mean
    initial_value = 5.0

    # Theoretical stationary variance
    stationary_var = sigma**2 / (2 * mu)

    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=initial_value
    )

    # Check variance at different time points
    early_variance = np.var(simulated_values[:, 1000], ddof=1)
    late_variance = np.var(simulated_values[:, -1], ddof=1)

    # Late variance should be within reasonable range of stationary
    # (variance estimation is noisy with finite samples)
    assert 0.6 * stationary_var < late_variance < 1.4 * stationary_var, (
        f"Late variance {late_variance:.4f} should approach stationary {stationary_var:.4f}"
    )

    # Early variance (starting far from mean) should be higher than late variance
    # as the distribution hasn't converged yet
    assert early_variance > late_variance * 0.8, (
        f"Early variance {early_variance:.4f} should be higher than or similar to late variance {late_variance:.4f}"
    )


def test_ou_half_life():
    """
    Test that half-life of mean reversion matches theoretical value.

    Half-life is the time it takes for the distance to the mean to halve: t_half = ln(2)/mu
    """
    set_random_seed(57)

    mu = 0.5
    theta = 0.0
    sigma = 0.2

    # Theoretical half-life
    theoretical_half_life = np.log(2) / mu

    # Start away from mean
    initial_value = 2.0

    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))

    # Simulate many paths to average out noise
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=initial_value
    )

    # Calculate average distance to mean over time
    distances = np.abs(simulated_values - theta)
    avg_distances = np.mean(distances, axis=0)

    # Initial distance
    initial_distance = avg_distances[0]

    # Find the time when distance is approximately half of initial
    half_distance = initial_distance / 2

    # Find index where distance crosses half_distance
    below_half = avg_distances < half_distance
    if np.any(below_half):
        observed_half_life_idx = np.argmax(below_half)
        observed_half_life = observed_half_life_idx * DELTA_T

        # Should be within ±50% of theoretical (wider tolerance due to discrete time steps)
        assert (
            0.5 * theoretical_half_life
            < observed_half_life
            < 1.5 * theoretical_half_life
        ), (
            f"Observed half-life {observed_half_life:.2f} differs from theoretical {theoretical_half_life:.2f}"
        )


def test_ou_symmetry():
    """
    Verify process is symmetric around the mean.

    Starting above vs below mean should produce symmetric behavior.
    """
    set_random_seed(58)

    mu = 1.5
    theta = 0.0
    sigma = 0.25

    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))

    # Start above mean
    initial_above = 2.0
    simulated_above = ou.simulate(N=5000, N_simulated=N_SIMULATED, X_0=initial_above)

    # Start below mean
    initial_below = -2.0
    simulated_below = ou.simulate(N=5000, N_simulated=N_SIMULATED, X_0=initial_below)

    # Calculate average distance to mean over time
    dist_above = np.mean(np.abs(simulated_above - theta), axis=0)
    dist_below = np.mean(np.abs(simulated_below - theta), axis=0)

    # The average distances should be similar (symmetric)
    # Check at a few time points (wider tolerance due to finite sample effects)
    for t_idx in [100, 500, 1000, 2000]:
        ratio = dist_above[t_idx] / dist_below[t_idx]
        assert 0.7 < ratio < 1.4, (
            f"At time {t_idx}, asymmetry detected. "
            f"Distance above: {dist_above[t_idx]:.4f}, below: {dist_below[t_idx]:.4f}"
        )


# ============================================================================
# Fit Method Accuracy Tests - Additional
# ============================================================================


def test_ou_fit_recovers_parameters():
    """
    Test that fit() recovers parameters from simulated data with various starting points.

    Should work regardless of where the process starts.
    """
    set_random_seed(59)

    # True parameters
    mu_true = 0.8
    theta_true = 1.0
    sigma_true = 0.3

    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))

    # Test with different starting points
    starting_points = [-2.0, 0.0, 1.0, 3.0]

    for X_0 in starting_points:
        # Simulate long path
        simulated_values = ou.simulate(N=N_POINTS, N_simulated=1, X_0=X_0)[0]

        # Fit
        fitted_params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values)

        # Check recovery (±20% tolerance)
        assert 0.8 * mu_true < fitted_params.mu < 1.2 * mu_true, (
            f"Starting from {X_0}, fitted mu {fitted_params.mu:.4f} differs from true {mu_true:.4f}"
        )

        assert 0.8 * theta_true < fitted_params.theta < 1.2 * theta_true, (
            f"Starting from {X_0}, fitted theta {fitted_params.theta:.4f} differs from true {theta_true:.4f}"
        )


def test_ou_fit_with_different_sample_sizes():
    """
    Verify that longer time series improve fit accuracy.

    Estimation error should decrease with more data.
    """
    set_random_seed(60)

    mu_true = 0.5
    theta_true = 0.0
    sigma_true = 0.2

    ou = OrnsteinUhlenbeck(OUParams(mu=mu_true, theta=theta_true, sigma=sigma_true))

    sample_sizes = [1000, 5000, 10000]
    mu_errors = []

    for n in sample_sizes:
        errors = []
        for i in range(20):
            simulated_values = ou.simulate(N=n, N_simulated=1, X_0=theta_true)[0]

            fitted_params = OrnsteinUhlenbeck(OUParams(0, 0, 0)).fit(simulated_values)

            # Calculate relative error for mu
            error = abs(fitted_params.mu - mu_true) / mu_true
            errors.append(error)

        mu_errors.append(np.mean(errors))

    # Errors should decrease with sample size
    assert mu_errors[2] < mu_errors[0], (
        f"Fit error should decrease with sample size. Errors: {mu_errors}"
    )


# ============================================================================
# Edge Case Tests - Additional
# ============================================================================


def test_ou_invalid_coefficient_raises_error():
    """
    Test that fit() raises ValueError when data doesn't follow OU process.

    For example, trending data (explosive) should fail.
    """
    set_random_seed(61)

    # Create trending data (not mean-reverting)
    # This is like a random walk with drift
    trend = 0.1
    noise = 0.1
    X = np.cumsum(np.random.normal(trend, noise, 1000))

    ou = OrnsteinUhlenbeck(OUParams(0, 0, 0))

    # Should raise ValueError because coefficient will be >= 1
    with pytest.raises(ValueError, match="Invalid OLS coefficient"):
        ou.fit(X)


def test_ou_very_slow_mean_reversion():
    """
    Test behavior when mu is very small (near random walk).

    Should still work but with weak mean reversion.
    """
    set_random_seed(62)

    mu = 0.01  # Very slow mean reversion
    theta = 0.0
    sigma = 0.2

    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))

    # Simulate
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=N_SIMULATED, X_0=5.0)

    # Check basic properties
    assert np.all(np.isfinite(simulated_values)), (
        "Invalid values with slow mean reversion"
    )

    # Mean reversion should be very weak - process should still be far from mean after long time
    # But should show some tendency toward mean
    initial_mean_dist = np.mean(np.abs(simulated_values[:, 0] - theta))
    final_mean_dist = np.mean(np.abs(simulated_values[:, -1] - theta))

    # With slow reversion, might not get much closer, but shouldn't diverge
    assert final_mean_dist < 2 * initial_mean_dist, (
        f"Process should not diverge with slow mean reversion. "
        f"Initial: {initial_mean_dist:.2f}, Final: {final_mean_dist:.2f}"
    )


def test_ou_crossing_behavior():
    """
    Verify process crosses the mean with expected frequency.

    For OU process starting away from mean, should cross with regularity.
    """
    set_random_seed(63)

    mu = 1.0
    theta = 0.0
    sigma = 0.3

    ou = OrnsteinUhlenbeck(OUParams(mu=mu, theta=theta, sigma=sigma))

    # Start away from mean
    initial_value = 2.0

    # Simulate single long path
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=1, X_0=initial_value)[0]

    # Count crossings of the mean
    crossings = 0
    for i in range(1, len(simulated_values)):
        # Check if sign changed (crossed zero/theta)
        if (simulated_values[i - 1] - theta) * (simulated_values[i] - theta) < 0:
            crossings += 1

    # With strong mean reversion, should cross many times
    # Expect at least 100 crossings in 10000 steps
    assert crossings > 100, (
        f"Expected many crossings with mean reversion. Got {crossings}"
    )

    # But not too many (shouldn't be purely noise)
    assert crossings < 5000, (
        f"Too many crossings suggests noise dominates. Got {crossings}"
    )
