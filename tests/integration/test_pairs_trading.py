import os
import sys
import datetime as dt

import pandas as pd
import pytest
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from tests.utils import (
    sample_asset_data,
    sample_provider_data,
    generate_market_data_dataframe,
)
from src.attributes.pairs_trading import refresh_pairs_trading_attribute_data

TOLERANCE = 0.20  # ±20% tolerance for parameter recovery


@pytest.mark.asyncio
async def test_pairs_trading():
    """Test the pairs trading attribute data."""

    # Get the engine.
    engine = await get_engine()

    # Set the lookback window days.
    lookback_window_days = 30

    # Create the provider and asset data.
    _, kraken_provider = await sample_provider_data(engine)
    (
        _,
        _,
        btc_asset,
        eth_asset,
        usd_asset,
    ) = await sample_asset_data(engine)

    # Create EUR asset manually
    with Session(engine) as session:
        eur_asset = models.Asset(
            symbol="EUR",
            name="Euro",
            asset_type_id=usd_asset.asset_type_id,  # Same type as USD
            is_active=True,
        )
        session.add(eur_asset)
        session.commit()
        session.refresh(eur_asset)

    # Create the pairs trading asset group type.
    with Session(engine) as session:
        pairs_trading_asset_group_type = models.AssetGroupType(
            symbol="STATISTICAL_PAIRS_TRADING",
            name="Statistical Pairs Trading",
            description="Group type for pairs trading attributes like the cointegration and hedge ratio.",
            is_active=True,
        )
        session.add(pairs_trading_asset_group_type)
        session.commit()
        session.refresh(pairs_trading_asset_group_type)

    # Generate market data for USD pairs (BTC/USD and ETH/USD)
    date: dt.date = dt.date(2025, 1, 1)
    df_usd = generate_market_data_dataframe(
        to_asset_ids=[btc_asset.id, eth_asset.id],
        n_points=lookback_window_days * 24 * 60,  # lookback window days of data
        n_cointegrated_pairs=1,
        provider_id=kraken_provider.id,
        from_asset_id=usd_asset.id,
        cointegrated_params={
            "alpha": 10.0,
            "beta": 1.5,
            "drift": 0.05,
            "volatility": 0.2,
            "theta": 0.5,
            "mu": 0.1,
            "sigma": 2.0,
            "start_price": 100.0,
        },
        start_time=dt.datetime.combine(
            date - dt.timedelta(days=lookback_window_days), dt.time.min
        ),
    )

    # Generate market data for EUR pairs (BTC/EUR and ETH/EUR)
    df_eur = generate_market_data_dataframe(
        to_asset_ids=[btc_asset.id, eth_asset.id],
        n_points=lookback_window_days * 24 * 60,  # lookback window days of data
        n_cointegrated_pairs=1,
        provider_id=kraken_provider.id,
        from_asset_id=eur_asset.id,
        cointegrated_params={
            "alpha": 10.0,
            "beta": 1.5,
            "drift": 0.05,
            "volatility": 0.2,
            "theta": 0.5,
            "mu": 0.1,
            "sigma": 2.0,
            "start_price": 100.0,
        },
        start_time=dt.datetime.combine(
            date - dt.timedelta(days=lookback_window_days), dt.time.min
        ),
    )

    # Create the provider asset groups.
    with Session(engine) as session:
        btc_usd_eth_usd_provider_asset_group = models.ProviderAssetGroup(
            asset_group_type_id=pairs_trading_asset_group_type.id,
            is_active=True,
            members=[
                models.ProviderAssetGroupMember(
                    provider_id=kraken_provider.id,
                    from_asset_id=usd_asset.id,
                    to_asset_id=btc_asset.id,
                    order=1,
                ),
                models.ProviderAssetGroupMember(
                    provider_id=kraken_provider.id,
                    from_asset_id=usd_asset.id,
                    to_asset_id=eth_asset.id,
                    order=2,
                ),
            ],
        )
        session.add(btc_usd_eth_usd_provider_asset_group)
        session.commit()
        session.refresh(btc_usd_eth_usd_provider_asset_group)
        btc_usd_eth_eur_provider_asset_group = models.ProviderAssetGroup(
            asset_group_type_id=pairs_trading_asset_group_type.id,
            is_active=True,
            members=[
                models.ProviderAssetGroupMember(
                    provider_id=kraken_provider.id,
                    from_asset_id=eur_asset.id,
                    to_asset_id=btc_asset.id,
                    order=1,
                ),
                models.ProviderAssetGroupMember(
                    provider_id=kraken_provider.id,
                    from_asset_id=eur_asset.id,
                    to_asset_id=eth_asset.id,
                    order=2,
                ),
            ],
        )
        session.add(btc_usd_eth_eur_provider_asset_group)
        session.commit()
        session.refresh(btc_usd_eth_eur_provider_asset_group)

    # Adjust EUR prices to be different (multiply by 0.85)
    df_eur["close"] = df_eur["close"] * 0.85

    # Combine both dataframes
    df = pd.concat([df_usd, df_eur], ignore_index=True)

    # Set the data.
    await set_data(models.ProviderAssetMarket.__tablename__, df)

    # Create the provider asset groups.
    await refresh_pairs_trading_attribute_data(
        date=date,
        lookback_window_days=lookback_window_days,
    )

    # Get the pairs trading attribute data.
    pairs_trading_attribute_data = pd.read_sql(
        select(models.ProviderAssetGroupAttribute),
        con=engine,
    )

    # Check the pairs trading attribute data.
    assert len(pairs_trading_attribute_data) == 2
