from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

# Constants: These are the parameters that are used to estimate the GBM and OU processes they should never be changed.
DELTA_T = 1


class StochasticModelParams(ABC):
    """
    Base class for stochastic model parameters.
    """

    @abstractmethod
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the parameters to a dictionary.
        """
        pass


@dataclass
class GBMParams(StochasticModelParams):
    mu: float  # drift parameter
    sigma: float  # volatility parameter

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}


@dataclass
class OUParams(StochasticModelParams):
    mu: float  # mean reversion parameter
    theta: float  # asymptotic mean
    sigma: float  # Brownian motion scale (standard deviation)

    def __init__(self, mu: float, theta: float, sigma: float):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def to_dict(self) -> dict:
        return {"mu": self.mu, "theta": self.theta, "sigma": self.sigma}


class StochasticModel(ABC):
    """
    Base class for stochastic models.
    """

    __params: StochasticModelParams

    def __init__(self, params: StochasticModelParams = None):
        self.__params = params

    @property
    def params(self) -> StochasticModelParams:
        """
        Get the parameters of the model.
        """
        if self.__params is None:
            raise ValueError("Parameters are not set for the model.")
        return self.__params

    @params.setter
    def params(self, params: StochasticModelParams):
        """
        Set the parameters of the model.
        """
        self.__params = params

    @abstractmethod
    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log likelihood of the model.
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> tuple[float, float, float, float]:
        """
        Fit the model to the data.
        """
        pass

    @abstractmethod
    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulate the model.
        """
        pass


class GeometricBrownianMotion(StochasticModel):
    """
    Geometric Brownian Motion process.
    """

    def __init__(self, params: GBMParams = None):
        super().__init__(params)

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Calculates the log-likelihood for a Geometric Brownian Motion model.

        The GBM SDE: dS_t = mu*S_t*dt + sigma*S_t*dW_t
        Log-returns follow: r ~ N((mu - 0.5*sigma²)*dt, sigma²*dt)

        Uses the global DELTA_T constant for time step.

        Args:
            X (np.ndarray): Array of asset prices

        Returns:
            float: The negative log-likelihood (for minimization)
        """
        # Calculate log-returns from prices
        log_returns = np.diff(np.log(X))
        n = len(log_returns)

        # Expected mean of log-returns
        expected_mean = (self.params.mu - 0.5 * self.params.sigma**2) * DELTA_T

        # Calculate log-likelihood terms
        term1 = -0.5 * n * np.log(2 * np.pi)
        term2 = -0.5 * n * np.log(self.params.sigma**2 * DELTA_T)
        term3 = -np.sum((log_returns - expected_mean) ** 2) / (
            2 * self.params.sigma**2 * DELTA_T
        )

        log_likelihood = term1 + term2 + term3

        # Return negative for minimization
        return -log_likelihood

    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulates GBM paths using the Euler-Maruyama scheme.

        Following the approach from:
        https://towardsdatascience.com/stochastic-processes-simulation-the-ornstein-uhlenbeck-process-e8bff820f3/

        GBM SDE: dS = mu*S*dt + sigma*S*dW
        Solution: S_t = S_0 * exp((mu - 0.5*sigma²)*t + sigma*W_t)

        Uses the global DELTA_T constant for time step.

        input: N - number of time steps
               N_simulated - number of paths to simulate
               X_0 - initial value
        returns: np.ndarray of shape (N_simulated, N) with simulated paths
        """
        # Initialize the simulated paths
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0

        # Simulate using the exact solution at each time step
        for i in range(1, N):
            X_simulated[:, i] = X_simulated[:, i - 1] * np.exp(
                (self.params.mu - 0.5 * self.params.sigma**2) * DELTA_T
                + self.params.sigma
                * np.sqrt(DELTA_T)
                * np.random.normal(0, 1, N_simulated)
            )

        return X_simulated

    def fit(self, X: np.ndarray) -> GBMParams:
        """
        Estimates Geometric Brownian Motion parameters from price data.

        Following the moment matching approach similar to:
        https://towardsdatascience.com/stochastic-processes-simulation-the-ornstein-uhlenbeck-process-e8bff820f3/

        The GBM SDE is: dS = mu*S*dt + sigma*S*dW
        Taking logs: d(log S) = (mu - 0.5*sigma²)*dt + sigma*dW

        Log-returns follow: r = log(S_t/S_{t-1}) ~ N((mu - 0.5*sigma²)*dt, sigma²*dt)

        Uses the global DELTA_T constant for time step.

        Parameters estimated by moment matching:
        - Var(r) = sigma²*DELTA_T  =>  sigma = sqrt(Var(r)/DELTA_T)
        - E[r] = (mu - 0.5*sigma²)*DELTA_T  =>  mu = E[r]/DELTA_T + 0.5*sigma²

        input: X - array-like price data
        returns: GBMParams with estimated mu and sigma
        """
        # Calculate log-returns
        log_returns = np.diff(np.log(X))

        # Estimate sigma from variance of log-returns
        # Var(r) = sigma²*DELTA_T
        sigma = np.sqrt(np.var(log_returns, ddof=1) / DELTA_T)

        # Estimate mu from mean of log-returns
        # E[r] = (mu - 0.5*sigma²)*DELTA_T
        mu = np.mean(log_returns) / DELTA_T + 0.5 * sigma**2

        # Update model parameters
        self.params = GBMParams(mu=mu, sigma=sigma)

        return self.params


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is defined by:

    dX_t = theta (mu - X_t) dt + sigma dW_t

    where $W_t$ is a Wiener process.
    """

    def __init__(self, params: OUParams = None):
        super().__init__(params)

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Computes the log likelihood of the OU process.

        Uses the global DELTA_T constant for time step.
        """
        # Get the number of observations.
        n = len(X)

        # Get the lag and next values.
        X_lag = X[:-1]
        X_next = X[1:]

        # Get the tilde sigma.
        tilde_sigma = self.params.sigma * np.sqrt(
            (1 - np.exp(-2 * self.params.mu * DELTA_T)) / (2 * self.params.mu)
        )

        # Compute the log likelihood.
        log_likelihood = (
            -0.5 * np.log(2 * np.pi)
            - np.log(tilde_sigma)
            - 1
            / (2 * n * tilde_sigma**2)
            * np.sum(
                (
                    X_next
                    - X_lag * np.exp(-self.params.mu * DELTA_T)
                    - self.params.theta * (1 - np.exp(-self.params.mu * DELTA_T))
                )
                ** 2
            )
        )

        return -log_likelihood

    def fit(self, X: np.ndarray) -> OUParams:
        """
        Estimates Ornstein-Uhlenbeck parameters from the given array using OLS regression
        on the exact discrete-time solution (not the Euler approximation).

        The exact OU discrete transition is:
        X_{t+dt} = theta*(1 - exp(-mu*dt)) + X_t*exp(-mu*dt) + sigma*sqrt((1-exp(-2*mu*dt))/(2*mu))*noise

        Letting a = exp(-mu*dt), we can write:
        X_{t+dt} = theta*(1 - a) + X_t*a + noise

        OLS regression of X_{t+1} on X_t gives:
        - intercept = theta*(1 - a)
        - coef = a = exp(-mu*dt)

        Therefore:
        - mu = -log(coef) / dt
        - theta = intercept / (1 - coef)

        input: X - array-like data to be fit as an OU process
        returns: OUParams
        """
        # Regress X_{t+1} on X_t (not differences!)
        X_next = X[1:]
        X_lag = X[:-1]
        X_with_const = sm.add_constant(X_lag)

        # Fit OLS regression: X_{t+1} = intercept + coef*X_t
        model = sm.OLS(X_next, X_with_const)
        results = model.fit()

        # Extract coefficients: [intercept, coef]
        intercept = results.params[0]
        coef = results.params[1]

        # Extract OU parameters from exact solution
        # coef = exp(-mu*DELTA_T), which must be in (0, 1) for a valid mean-reverting OU process
        if not (0 < coef < 1):
            raise ValueError(
                f"Invalid OLS coefficient {coef:.6f}. For a mean-reverting OU process, "
                f"the coefficient must be in (0, 1) since it equals exp(-mu*DELTA_T). "
                f"This indicates the data does not follow an OU process or has insufficient variation."
            )

        mu = -np.log(coef) / DELTA_T
        theta = intercept / (1 - coef)

        # Get residual standard deviation
        # residuals = X_{t+1} - (theta*(1-a) + X_t*a)
        # Theoretical: sigma_residual = sigma * sqrt((1 - exp(-2*mu*dt)) / (2*mu))
        residual_std = np.sqrt(results.mse_resid)

        # Back out sigma from residual_std
        # residual_std^2 = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
        sigma = residual_std * np.sqrt(2 * mu / (1 - np.exp(-2 * mu * DELTA_T)))

        # Update the parameters.
        self.params = OUParams(mu=mu, theta=theta, sigma=sigma)

        return self.params

    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulates the OU process.

        Uses the global DELTA_T constant for time step.
        """
        # Initialize the simulated process.
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0  # initial value

        # Simulate the process.
        for i in range(1, N):
            X_simulated[:, i] = (
                X_simulated[:, i - 1] * np.exp(-self.params.mu * DELTA_T)
                + self.params.theta * (1 - np.exp(-self.params.mu * DELTA_T))
                + self.params.sigma
                * np.sqrt(
                    (1 - np.exp(-2 * self.params.mu * DELTA_T)) / (2 * self.params.mu)
                )
                * np.random.normal(0, 1, N_simulated)
            )

        return X_simulated
