import numpy as np
import scipy.optimize as so


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

    def simulate(self, N: int, N_simulated: int, dt: float = None) -> np.ndarray:
        """
        Simulates the OU process.
        """
        if dt is None:
            dt = self.dt

        # Initialize the simulated process.
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = self.theta  # initial value

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
