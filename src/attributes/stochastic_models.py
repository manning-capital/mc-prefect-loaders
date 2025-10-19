import numpy as np
import scipy.optimize as so


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion process.
    """
    X: np.ndarray
    dt: float
    mu: float
    sigma: float
    
    def __init__(self, mu: float = None, sigma: float = None):
        self.mu = mu
        self.sigma = sigma

    @staticmethod
    def __log_likelihood(
        params: tuple[float, float], X: np.ndarray, dt: float
    ) -> float:
        """
        Computes the log likelihood of the GBM process.
        """
        """
        Calculates the log-likelihood for a Geometric Brownian Motion model.

        The GBM is defined by dS_t = mu*S_t*dt + sigma*S_t*dW_t.
        The log-returns follow a normal distribution.

        Args:
            params (list or tuple): A list containing the GBM parameters [mu, sigma].
            prices (np.ndarray): A 1D NumPy array of asset prices.
            dt (float): The constant time step between price observations.

        Returns:
            float: The negative log-likelihood value. This is typically used for
                minimization routines, which is why the negative is returned.
        """
        mu, sigma = params

        if sigma <= 0:
            return np.inf  # Negative sigma is not valid

        # Calculate log-returns from the prices
        log_returns = np.diff(np.log(X))

        # Number of observations (log-return increments)
        n = len(log_returns)

        # Expected mean of the log-returns over time step dt
        expected_mean = (mu - 0.5 * sigma**2) * dt

        # Calculate the log-likelihood terms
        term1 = -0.5 * n * np.log(2 * np.pi)
        term2 = -0.5 * n * np.log(sigma**2 * dt)
        term3 = -np.sum((log_returns - expected_mean)**2) / (2 * sigma**2 * dt)

        log_likelihood = term1 + term2 + term3

        # For minimization, return the negative log-likelihood
        return -log_likelihood


    def simulate(self, N: int, N_simulated: int, X_0: float, dt: float = None) -> np.ndarray:
        """
        Simulates the GBM process.
        """
        if dt is None:
            dt = self.dt
            
        # Initialize the simulated process.
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0  # initial value
        
        # Simulate the process.
        for i in range(1, N):
            X_simulated[:, i] = X_simulated[:, i - 1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * np.random.normal(0, 1, N_simulated))
            
        return X_simulated

    def fit(self, X: np.ndarray, dt: float) -> tuple[float, float, float, float]:
        """
        Estimates Geometric Brownian Motion coefficients (µ, σ) of the given array
        using the Maximum Likelihood Estimation method

        input: X - array-like data to be fit as a GBM process
        returns: µ, σ, Total Log Likelihood
        """
        # Set the parameters.
        self.X = X
        self.dt = dt
        
        # Set the small bound.
        small_bound = 1e-9
        
        # Set the bounds.
        bounds = (
            (small_bound, None),
            (small_bound, None),
        )

        # Initialize the initial values using log-returns
        log_returns = np.diff(np.log(self.X))
        sigma_init = np.std(log_returns) / np.sqrt(dt)
        mu_init = np.mean(log_returns) / dt + 0.5 * sigma_init**2

        # Minimize the log likelihood.
        result = so.minimize(
            GeometricBrownianMotion.__log_likelihood,
            (mu_init, sigma_init),
            args=(self.X, self.dt),
            method="L-BFGS-B",
            bounds=bounds,
            tol=1e-10,
        )

        # Get the parameters.
        mu, sigma = result.x
        max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood

        # Set the parameters.
        self.mu = mu
        self.sigma = sigma

        # Return the parameters.
        return mu, sigma, max_log_likelihood

class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is defined by:

    dX_t = theta (mu - X_t) dt + sigma dW_t

    where $W_t$ is a Wiener process.
    """

    X: np.ndarray
    dt: float
    mu: float
    theta: float
    sigma: float

    def __init__(self, mu: float = None, theta: float = None, sigma: float = None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    @staticmethod
    def __log_likelihood(
        params: tuple[float, float, float], X: np.ndarray, dt: float
    ) -> float:
        """
        Computes the log likelihood of the OU process.
        """

        # Get the parameters.
        mu, theta, sigma = params

        # Get the number of observations.
        n = len(X)

        # Get the lag and next values.
        X_lag = X[:-1]
        X_next = X[1:]

        # Get the tilde sigma.
        tilde_sigma = sigma * np.sqrt((1 - np.exp(-2 * mu * dt)) / (2 * mu))

        # Compute the log likelihood.
        log_likelihood = (
            -0.5 * np.log(2 * np.pi)
            - np.log(tilde_sigma)
            - 1
            / (2 * n * tilde_sigma**2)
            * np.sum(
                (X_next - X_lag * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt)))
                ** 2
            )
        )

        return -log_likelihood

    def fit(self, X: np.ndarray, dt: float) -> tuple[float, float, float, float]:
        """
        Estimates Ornstein-Uhlenbeck coefficients (θ, µ, σ) of the given array
        using the Maximum Likelihood Estimation method

        input: X - array-like data to be fit as an OU process
        returns: θ, µ, σ, Total Log Likelihood
        """
        # Set the parameters.
        self.X = X
        self.dt = dt

        # Set the small bound.
        small_bound = 1e-9

        # Set the bounds.
        bounds = (
            (small_bound, None),
            (None, None),
            (small_bound, None),
        )  # mu > 0, theta ∈ ℝ, sigma > 0

        # Initialize the initial values.
        mu_init = small_bound
        sigma_init = np.std(self.X)
        theta_init = np.mean(self.X)

        # Minimize the log likelihood.
        result = so.minimize(
            OrnsteinUhlenbeck.__log_likelihood,
            (mu_init, theta_init, sigma_init),
            args=(self.X, self.dt),
            method="L-BFGS-B",
            bounds=bounds,
            tol=1e-10,
        )

        # Get the parameters.
        mu, theta, sigma = result.x
        max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood

        # Set the parameters.
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        # Return the parameters.
        return mu, theta, sigma, max_log_likelihood

    def simulate(self, N: int, N_simulated: int, X_0: float, dt: float = None) -> np.ndarray:
        """
        Simulates the OU process.
        """
        if dt is None:
            dt = self.dt

        # Initialize the simulated process.
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0  # initial value

        # Simulate the process.
        for i in range(1, N):
            X_simulated[:, i] = (
                X_simulated[:, i - 1] * np.exp(-self.mu * dt)
                + self.theta * (1 - np.exp(-self.mu * dt))
                + self.sigma
                * np.sqrt((1 - np.exp(-2 * self.mu * dt)) / (2 * self.mu))
                * np.random.normal(0, 1, N_simulated)
            )

        return X_simulated