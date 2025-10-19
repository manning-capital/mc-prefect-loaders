import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from src.attributes.stochastic_models import (
    DELTA_T,
    OrnsteinUhlenbeck,
    GeometricBrownianMotion,
)

# Test configuration constants
N_POINTS = 100_000  # Large dataset for reliable parameter estimation
N_SIMULATED = 50  # Number of simulated paths
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
    else:
        lower_bound = true_value * (1 - tolerance)
        upper_bound = true_value * (1 + tolerance)
        assert lower_bound <= fitted_value <= upper_bound, (
            f"Fitted value {fitted_value} not within ±{tolerance * 100}% of true value {true_value}. "
            f"Expected range: [{lower_bound:.6f}, {upper_bound:.6f}]"
        )


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


# ============================================================================
# Geometric Brownian Motion Tests
# ============================================================================


def test_gbm_parameter_recovery_standard():
    """
    Test GBM parameter recovery with typical market parameters.

    Typical annual parameters converted to per-second:
    - mu = 0.05 (5% annual drift)
    - sigma = 0.2 (20% annual volatility)
    """
    set_random_seed(42)

    # True parameters
    mu_true = 0.05
    sigma_true = 0.2

    # Create model and simulate
    gbm = GeometricBrownianMotion(mu=mu_true, sigma=sigma_true)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE, dt=DELTA_T
    )

    # Fit the simulated data
    gbm_fit = GeometricBrownianMotion()
    mu_fitted, sigma_fitted, log_likelihood = gbm_fit.fit(simulated_prices, DELTA_T)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted, mu_true)
    assert_within_tolerance(sigma_fitted, sigma_true)

    # Assert log likelihood is finite and reasonable
    assert np.isfinite(log_likelihood)
    assert log_likelihood < 0  # Log likelihood should be negative


def test_gbm_parameter_recovery_high_volatility():
    """
    Test GBM parameter recovery with high volatility.

    Parameters:
    - mu = 0.1 (10% annual drift)
    - sigma = 0.5 (50% annual volatility - crypto-like)
    """
    set_random_seed(43)

    # True parameters
    mu_true = 0.1
    sigma_true = 0.5

    # Create model and simulate
    gbm = GeometricBrownianMotion(mu=mu_true, sigma=sigma_true)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_PRICE, dt=DELTA_T
    )

    # Initialize the fitted parameters
    mu_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)
    log_likelihood = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        gbm_fit = GeometricBrownianMotion()
        mu_fitted[i], sigma_fitted[i], log_likelihood[i] = gbm_fit.fit(
            simulated_prices, DELTA_T
        )

    # Calculate the mean of the fitted parameters
    mu_fitted = np.mean(mu_fitted)
    sigma_fitted = np.mean(sigma_fitted)
    log_likelihood = np.mean(log_likelihood)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted, mu_true, tolerance=TOLERANCE)
    assert_within_tolerance(sigma_fitted, sigma_true, tolerance=TOLERANCE)
    assert np.isfinite(log_likelihood), "Log likelihood is not finite"


def test_gbm_parameter_recovery_low_drift():
    """
    Test GBM parameter recovery with low drift.

    Parameters:
    - mu = 0.01 (1% annual drift)
    - sigma = 0.15 (15% annual volatility)
    """
    set_random_seed(44)

    # True parameters
    mu_true = 0.01
    sigma_true = 0.15

    # Create model and simulate
    gbm = GeometricBrownianMotion(mu=mu_true, sigma=sigma_true)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=1, X_0=INITIAL_PRICE, dt=DELTA_T
    )[0]

    # Fit the simulated data
    gbm_fit = GeometricBrownianMotion()
    mu_fitted, sigma_fitted, log_likelihood = gbm_fit.fit(simulated_prices, DELTA_T)

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted, mu_true)
    assert_within_tolerance(sigma_fitted, sigma_true)
    assert np.isfinite(log_likelihood)


def test_gbm_simulated_mean_properties():
    """
    Verify that GBM simulated log-returns have approximately correct mean.

    For GBM, log returns should have mean ≈ (mu - 0.5*sigma²)*dt
    """
    set_random_seed(45)

    mu = 0.05
    sigma = 0.2

    # Simulate
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=1, X_0=INITIAL_PRICE, dt=DELTA_T
    )[0]

    # Calculate log returns
    log_returns = np.diff(np.log(simulated_prices))

    # Expected mean of log returns
    expected_mean = (mu - 0.5 * sigma**2) * DELTA_T
    observed_mean = np.mean(log_returns)

    # Check if observed mean is close to expected (with some statistical tolerance)
    # Using 3 standard errors as tolerance
    std_error = sigma * np.sqrt(DELTA_T) / np.sqrt(len(log_returns))
    assert abs(observed_mean - expected_mean) < 3 * std_error, (
        f"Log returns mean {observed_mean} too far from expected {expected_mean}"
    )


def test_gbm_simulated_volatility_properties():
    """
    Verify that GBM simulated log-returns have approximately correct volatility.

    For GBM, log returns should have std dev ≈ sigma*sqrt(dt)
    """
    set_random_seed(46)

    mu = 0.05
    sigma = 0.2

    # Simulate
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=1, X_0=INITIAL_PRICE, dt=DELTA_T
    )[0]

    # Calculate log returns
    log_returns = np.diff(np.log(simulated_prices))

    # Expected std dev of log returns
    expected_std = sigma * np.sqrt(DELTA_T)
    observed_std = np.std(log_returns, ddof=1)

    # Check if observed std is within reasonable range (±20% for std estimation)
    assert_within_tolerance(observed_std, expected_std, tolerance=0.20)


def test_gbm_positive_prices():
    """
    Ensure that GBM simulation maintains positive prices at all times.

    GBM is designed to keep prices positive by construction.
    """
    set_random_seed(47)

    mu = 0.1
    sigma = 0.5

    # Simulate with high volatility to stress test
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
    simulated_prices = gbm.simulate(
        N=N_POINTS, N_simulated=1, X_0=INITIAL_PRICE, dt=DELTA_T
    )[0]

    # Assert all prices are positive
    assert np.all(simulated_prices > 0), (
        f"GBM produced non-positive prices. Min: {np.min(simulated_prices)}"
    )

    # Assert no NaN or Inf values
    assert np.all(np.isfinite(simulated_prices)), "GBM produced NaN or Inf values"


# ============================================================================
# Ornstein-Uhlenbeck Tests
# ============================================================================


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
    mu_true = 0.1  # mean reversion speed
    theta_true = 0.0  # long-term mean
    sigma_true = 0.1

    # Create model and simulate
    ou = OrnsteinUhlenbeck(mu=mu_true, theta=theta_true, sigma=sigma_true)
    simulated_values = ou.simulate(
        N=N_POINTS, N_simulated=N_SIMULATED, X_0=INITIAL_VALUE, dt=DELTA_T
    )

    # Store the fitted parameters.
    mu_fitted = np.zeros(N_SIMULATED)
    theta_fitted = np.zeros(N_SIMULATED)
    sigma_fitted = np.zeros(N_SIMULATED)

    # Fit the simulated data
    for i in range(N_SIMULATED):
        params = OrnsteinUhlenbeck().fit(simulated_values[i, :])
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
    mu_true = 5.0  # mean reversion speed
    theta_true = 1.0  # long-term mean
    sigma_true = 0.2

    # Create model and simulate
    ou = OrnsteinUhlenbeck(mu=mu_true, theta=theta_true, sigma=sigma_true)
    simulated_values = ou.simulate(
        N=N_POINTS,
        N_simulated=1,
        X_0=theta_true,  # Start at mean
        dt=DELTA_T,
    )[0]

    # Fit the simulated data
    ou_fit = OrnsteinUhlenbeck()
    mu_fitted, theta_fitted, sigma_fitted, log_likelihood = ou_fit.fit(
        simulated_values, DELTA_T
    )

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted, mu_true)
    assert_within_tolerance(theta_fitted, theta_true)
    assert_within_tolerance(sigma_fitted, sigma_true)
    assert np.isfinite(log_likelihood)


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
    ou = OrnsteinUhlenbeck(mu=mu_true, theta=theta_true, sigma=sigma_true)
    simulated_values = ou.simulate(
        N=N_POINTS,
        N_simulated=1,
        X_0=theta_true,  # Start at mean
        dt=DELTA_T,
    )[0]

    # Fit the simulated data
    ou_fit = OrnsteinUhlenbeck()
    mu_fitted, theta_fitted, sigma_fitted, log_likelihood = ou_fit.fit(
        simulated_values, DELTA_T
    )

    # Assert parameters are recovered within tolerance
    assert_within_tolerance(mu_fitted, mu_true)
    assert_within_tolerance(theta_fitted, theta_true)
    assert_within_tolerance(sigma_fitted, sigma_true)
    assert np.isfinite(log_likelihood)


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
    ou = OrnsteinUhlenbeck(mu=mu, theta=theta, sigma=sigma)
    simulated_values = ou.simulate(
        N=10_000,  # Shorter path for this test
        N_simulated=1,
        X_0=initial_value,
        dt=DELTA_T,
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
    ou = OrnsteinUhlenbeck(mu=mu, theta=theta, sigma=sigma)
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=1, X_0=theta, dt=DELTA_T)[0]

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

    mu = 2.0  # Mean reversion speed
    theta = 0.0  # Long-term mean
    sigma = 0.2

    # Simulate
    ou = OrnsteinUhlenbeck(mu=mu, theta=theta, sigma=sigma)
    simulated_values = ou.simulate(N=N_POINTS, N_simulated=1, X_0=theta, dt=DELTA_T)[0]

    # Calculate autocorrelation at a few lags
    lags = [100, 500, 1000]

    for lag in lags:
        # Theoretical autocorrelation: exp(-mu * lag * dt)
        theoretical_autocorr = np.exp(-mu * lag * DELTA_T)

        # Observed autocorrelation
        centered = simulated_values - np.mean(simulated_values)
        autocorr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]

        # Check if observed is close to theoretical (±25% tolerance)
        assert_within_tolerance(autocorr, theoretical_autocorr, tolerance=0.25)
