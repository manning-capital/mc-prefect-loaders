import datetime as dt
from typing import List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import polars as pl
import mc_postgres_db.models as models
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from src.attributes.stochastic_models import (
    OUParams,
    GBMParams,
    OrnsteinUhlenbeck,
    GeometricBrownianMotion,
)

TOLERANCE = 0.15  # ±15% tolerance for parameter recovery


async def sample_provider_data(
    engine: Engine,
) -> tuple[models.ProviderType, models.Provider]:
    # Create the exchange provider type.
    with Session(engine) as session:
        exchange_provider_type = models.ProviderType(
            name="Exchange", description="Exchange", is_active=True
        )
        session.add(exchange_provider_type)
        session.commit()
        session.refresh(exchange_provider_type)

    # Create the exchange provider.
    with Session(engine) as session:
        kraken_provider = models.Provider(
            name="Kraken",
            description="Kraken",
            provider_type_id=exchange_provider_type.id,
            is_active=True,
        )
        session.add(kraken_provider)
        session.commit()
        session.refresh(kraken_provider)

    return exchange_provider_type, kraken_provider


async def sample_asset_data(
    engine: Engine,
) -> tuple[models.ProviderType, models.AssetType, models.AssetType]:
    # Create the crypto asset type.
    with Session(engine) as session:
        crypto_asset_type = models.AssetType(
            name="CryptoCurrency", description="CryptoCurrency", is_active=True
        )
        session.add(crypto_asset_type)
        session.commit()
        session.refresh(crypto_asset_type)

    # Create the fiat asset type.
    with Session(engine) as session:
        fiat_asset_type = models.AssetType(
            name="FiatCurrency", description="FiatCurrency", is_active=True
        )
        session.add(fiat_asset_type)
        session.commit()
        session.refresh(fiat_asset_type)

    # Create BTC asset.
    with Session(engine) as session:
        btc_asset = models.Asset(
            name="BTC", description="BTC", asset_type_id=crypto_asset_type.id
        )
        session.add(btc_asset)
        session.commit()
        session.refresh(btc_asset)

    # Create ETH asset.
    with Session(engine) as session:
        eth_asset = models.Asset(
            name="ETH", description="ETH", asset_type_id=crypto_asset_type.id
        )
        session.add(eth_asset)
        session.commit()
        session.refresh(eth_asset)

    # Create USD asset.
    with Session(engine) as session:
        usd_asset = models.Asset(
            name="USD", description="USD", asset_type_id=fiat_asset_type.id
        )
        session.add(usd_asset)
        session.commit()
        session.refresh(usd_asset)

    return crypto_asset_type, fiat_asset_type, btc_asset, eth_asset, usd_asset


async def create_base_data(
    engine: Engine,
) -> tuple[
    models.ProviderType, models.Provider, models.ContentType, models.SentimentType
]:
    # Create the provider type.
    with Session(engine) as session:
        provider_type = models.ProviderType(
            name="NEWS_PROVIDER", description="News provider", is_active=True
        )
        session.add(provider_type)
        session.commit()

        # Create the provider.
        provider = models.Provider(
            name="Coindesk", provider_type_id=provider_type.id, is_active=True
        )
        session.add(provider)
        session.commit()

        # Create the content type.
        content_type = models.ContentType(
            name="NEWS", description="News", is_active=True
        )
        session.add(content_type)
        session.commit()

        # Create the sentiment type.
        sentiment_type = models.SentimentType(
            name="NLTKVader", description="NLTKVader", is_active=True
        )
        session.add(sentiment_type)
        session.commit()

        # Get the ORM objects for all of the above.
        provider_type = session.execute(
            select(models.ProviderType).where(
                models.ProviderType.name == "NEWS_PROVIDER"
            )
        ).scalar_one()
        provider = session.execute(
            select(models.Provider).where(models.Provider.name == "Coindesk")
        ).scalar_one()
        content_type = session.execute(
            select(models.ContentType).where(models.ContentType.name == "NEWS")
        ).scalar_one()
        sentiment_type = session.execute(
            select(models.SentimentType).where(models.SentimentType.name == "NLTKVader")
        ).scalar_one()

        return provider_type, provider, content_type, sentiment_type


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


def generate_cointegrated_pair(
    n_points: int = 1000,
    alpha: float = 10.0,
    beta: float = 1.5,
    drift: float = 0.05,
    volatility: float = 0.2,
    theta: float = 0.5,
    mu: float = 0.1,
    sigma: float = 2.0,
    start_price: float = 100.0,
    resolution: dt.timedelta = dt.timedelta(minutes=1),
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate synthetic cointegrated pair data for testing.

    Creates two price series where:
    - close_1 follows GBM
    - close_2 = alpha + beta * close_1 + residuals
    - residuals follow OU process with known parameters

    Args:
        n_points: Number of data points
        alpha: Linear relationship intercept
        beta: Linear relationship slope
        drift: GBM drift
        volatility: GBM volatility
        theta: OU process asymptotic mean
        mu: OU process mean reversion speed
        sigma: OU process volatility
        start_price: Starting price for close_1
        resolution: Time resolution
        seed: Random seed for reproducibility

    Returns:
        pl.DataFrame with columns: timestamp, close_1, close_2
    """
    set_random_seed(seed)

    # Generate base price series using GBM
    gbm_params = GBMParams(mu=drift, sigma=volatility)
    gbm = GeometricBrownianMotion(params=gbm_params)
    close_1_prices = gbm.simulate(N=n_points, N_simulated=1, X_0=start_price)[0]

    # Generate OU residuals with proper parameters for mean reversion
    ou_params = OUParams(mu=mu, theta=theta, sigma=sigma)
    ou = OrnsteinUhlenbeck(params=ou_params)
    residuals = ou.simulate(N=n_points, N_simulated=1, X_0=theta)[0]

    # Create cointegrated close_2
    close_2_prices = alpha + beta * close_1_prices + residuals

    # Create timestamps
    start_time = dt.datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [start_time + i * resolution for i in range(n_points)]

    return pl.DataFrame(
        {"timestamp": timestamps, "close_1": close_1_prices, "close_2": close_2_prices}
    )


def generate_non_cointegrated_pair(
    n_points: int = 1000,
    resolution: dt.timedelta = dt.timedelta(minutes=1),
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate two independent price series that are NOT cointegrated.

    Args:
        n_points: Number of data points
        resolution: Time resolution
        seed: Random seed for reproducibility

    Returns:
        pl.DataFrame with columns: timestamp, close_1, close_2
    """
    set_random_seed(seed)

    # Generate two independent GBM processes
    gbm_params_1 = GBMParams(mu=0.05, sigma=0.2)
    gbm_params_2 = GBMParams(mu=0.03, sigma=0.15)

    gbm_1 = GeometricBrownianMotion(params=gbm_params_1)
    gbm_2 = GeometricBrownianMotion(params=gbm_params_2)

    close_1_prices = gbm_1.simulate(N=n_points, N_simulated=1, X_0=100.0)[0]
    close_2_prices = gbm_2.simulate(N=n_points, N_simulated=1, X_0=200.0)[0]

    # Create timestamps
    start_time = dt.datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [start_time + i * resolution for i in range(n_points)]

    return pl.DataFrame(
        {"timestamp": timestamps, "close_1": close_1_prices, "close_2": close_2_prices}
    )


def generate_trending_pair(
    n_points: int = 1000,
    trend_strength: float = 0.1,
    resolution: dt.timedelta = dt.timedelta(minutes=1),
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate price series with strong trending behavior.

    Args:
        n_points: Number of data points
        trend_strength: Strength of the trend (0.1 = 10% per time unit)
        resolution: Time resolution
        seed: Random seed for reproducibility

    Returns:
        pl.DataFrame with columns: timestamp, close_1, close_2
    """
    set_random_seed(seed)

    # Generate trending series
    timestamps = [
        dt.datetime(2024, 1, 1, 12, 0, 0) + i * resolution for i in range(n_points)
    ]

    # Create strong upward trend
    trend_1 = np.linspace(100, 100 * (1 + trend_strength * n_points), n_points)
    trend_2 = np.linspace(200, 200 * (1 + trend_strength * n_points), n_points)

    # Add some noise
    noise_1 = np.random.normal(0, 5, n_points)
    noise_2 = np.random.normal(0, 8, n_points)

    close_1_prices = trend_1 + noise_1
    close_2_prices = trend_2 + noise_2

    return pl.DataFrame(
        {"timestamp": timestamps, "close_1": close_1_prices, "close_2": close_2_prices}
    )


def generate_market_data_dataframe(
    to_asset_ids: List[int],
    n_points: int = 1000,
    n_cointegrated_pairs: Optional[int] = None,
    provider_id: int = 1,
    from_asset_id: int = 1,
    cointegrated_params: dict = None,
    resolution: dt.timedelta = dt.timedelta(minutes=1),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame for ProviderAssetMarket data with configurable assets and cointegrated pairs.

    Args:
        to_asset_ids: List of to asset IDs (required)
        n_points: Number of data points to generate
        n_cointegrated_pairs: Number of cointegrated pairs to generate. If None, all possible pairs will be cointegrated
        provider_id: Provider ID for all market data
        from_asset_id: From asset ID (base currency, e.g., USD)
        cointegrated_params: Dictionary of parameters for cointegrated pairs generation
        resolution: Time resolution for timestamps
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame with columns: timestamp, provider_id, from_asset_id, to_asset_id, close

    Raises:
        ValueError: If n_cointegrated_pairs exceeds the maximum possible pairs (C(n,2) where n = len(to_asset_ids))
    """
    n_assets = len(to_asset_ids)
    max_pairs = n_assets * (n_assets - 1) // 2  # C(n,2) = n choose 2

    # If n_cointegrated_pairs is None, use all possible pairs
    if n_cointegrated_pairs is None:
        n_cointegrated_pairs = max_pairs
    elif n_cointegrated_pairs > max_pairs:
        raise ValueError(
            f"n_cointegrated_pairs ({n_cointegrated_pairs}) cannot exceed maximum possible pairs ({max_pairs}) "
            f"for {n_assets} assets"
        )

    # Default cointegrated parameters
    default_cointegrated_params = {
        "alpha": 10.0,
        "beta": 1.5,
        "drift": 0.05,
        "volatility": 0.2,
        "theta": 0.5,
        "mu": 0.1,
        "sigma": 2.0,
        "start_price": 100.0,
    }

    if cointegrated_params is None:
        cointegrated_params = default_cointegrated_params
    else:
        # Merge with defaults
        cointegrated_params = {**default_cointegrated_params, **cointegrated_params}

    # Generate timestamps
    start_time = dt.datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [start_time + i * resolution for i in range(n_points)]

    # Generate cointegrated pairs
    cointegrated_pairs = []
    all_pairs = list(combinations(range(len(to_asset_ids)), 2))  # All possible pairs

    # Take only the requested number of pairs
    selected_pairs = all_pairs[:n_cointegrated_pairs]

    for i, (asset_idx_1, asset_idx_2) in enumerate(selected_pairs):
        # Generate cointegrated pair data
        cointegrated_df = generate_cointegrated_pair(
            n_points=n_points,
            alpha=cointegrated_params["alpha"],
            beta=cointegrated_params["beta"],
            drift=cointegrated_params["drift"],
            volatility=cointegrated_params["volatility"],
            theta=cointegrated_params["theta"],
            mu=cointegrated_params["mu"],
            sigma=cointegrated_params["sigma"],
            start_price=cointegrated_params["start_price"],
            resolution=resolution,
            seed=seed + i,  # Different seed for each pair
        )

        # Add first asset (close_1)
        cointegrated_pairs.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "provider_id": [provider_id] * n_points,
                    "from_asset_id": [from_asset_id] * n_points,
                    "to_asset_id": [to_asset_ids[asset_idx_1]] * n_points,
                    "close": cointegrated_df["close_1"].to_list(),
                }
            )
        )

        # Add second asset (close_2)
        cointegrated_pairs.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "provider_id": [provider_id] * n_points,
                    "from_asset_id": [from_asset_id] * n_points,
                    "to_asset_id": [to_asset_ids[asset_idx_2]] * n_points,
                    "close": cointegrated_df["close_2"].to_list(),
                }
            )
        )

    # Generate non-cointegrated assets for remaining assets
    non_cointegrated_assets = []
    used_asset_indices = set()
    for asset_idx_1, asset_idx_2 in selected_pairs:
        used_asset_indices.add(asset_idx_1)
        used_asset_indices.add(asset_idx_2)

    remaining_asset_indices = [
        i for i in range(len(to_asset_ids)) if i not in used_asset_indices
    ]

    for i, asset_idx in enumerate(remaining_asset_indices):
        # Generate independent GBM process
        set_random_seed(
            seed + 100 + i
        )  # Different seed for each non-cointegrated asset

        gbm_params = GBMParams(
            mu=0.05 + i * 0.01, sigma=0.2 + i * 0.05
        )  # Vary parameters slightly
        gbm = GeometricBrownianMotion(params=gbm_params)
        prices = gbm.simulate(N=n_points, N_simulated=1, X_0=100.0 + i * 50)[0]

        non_cointegrated_assets.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "provider_id": [provider_id] * n_points,
                    "from_asset_id": [from_asset_id] * n_points,
                    "to_asset_id": [to_asset_ids[asset_idx]] * n_points,
                    "close": prices,
                }
            )
        )

    # Combine all dataframes
    all_dataframes = cointegrated_pairs + non_cointegrated_assets
    result_df = pd.concat(all_dataframes, ignore_index=True)

    return result_df
