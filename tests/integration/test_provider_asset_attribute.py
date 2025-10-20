import os
import sys
import datetime as dt
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from tests.utils import sample_asset_data, sample_provider_data, assert_within_tolerance
from src.attributes.stochastic_models import (
    OUParams,
    GBMParams,
    OrnsteinUhlenbeck,
    GeometricBrownianMotion,
)
from src.attributes.provider_asset_attribute_flows import (
    refresh_provider_asset_attribute_data,
)

TOLERANCE = 0.20  # ±20% tolerance for parameter recovery


@pytest.mark.asyncio
async def test_creation_of_members_through_provider_asset_group_orm():
    # Get the engine.
    engine = await get_engine()

    # Create the provider and asset data.
    _, kraken_provider = await sample_provider_data(engine)
    (
        _,
        _,
        btc_asset,
        eth_asset,
        usd_asset,
    ) = await sample_asset_data(engine)

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

    # Create the provider asset group.
    with Session(engine) as session:
        provider_asset_group = models.ProviderAssetGroup(
            asset_group_type_id=pairs_trading_asset_group_type.id,
            name=f"{btc_asset.name}{usd_asset.name}-{eth_asset.name}{usd_asset.name}",
            description=f"{btc_asset.name}-{usd_asset.name} and {eth_asset.name}-{usd_asset.name}",
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
        session.add(provider_asset_group)
        session.commit()

    # Check if the provider asset group was created.
    with Session(engine) as session:
        provider_asset_group = session.execute(
            select(models.ProviderAssetGroup).where(
                models.ProviderAssetGroup.name
                == f"{btc_asset.name}{usd_asset.name}-{eth_asset.name}{usd_asset.name}"
            )
        ).scalar_one()
        assert provider_asset_group is not None
        assert len(provider_asset_group.members) == 2
        assert provider_asset_group.members[0].provider_id == kraken_provider.id
        assert provider_asset_group.members[0].from_asset_id == usd_asset.id
        assert provider_asset_group.members[0].to_asset_id == btc_asset.id
        assert provider_asset_group.members[0].order == 1
        assert provider_asset_group.members[1].provider_id == kraken_provider.id
        assert provider_asset_group.members[1].from_asset_id == usd_asset.id
        assert provider_asset_group.members[1].to_asset_id == eth_asset.id
        assert provider_asset_group.members[1].order == 2

    # Check if the provider asset group member was created.
    with Session(engine) as session:
        provider_asset_group_member_1 = session.execute(
            select(models.ProviderAssetGroupMember).where(
                models.ProviderAssetGroup.id == provider_asset_group.id,
                models.ProviderAssetGroupMember.order == 1,
            )
        ).scalar_one()
        provider_asset_group_member_2 = session.execute(
            select(models.ProviderAssetGroupMember).where(
                models.ProviderAssetGroup.id == provider_asset_group.id,
                models.ProviderAssetGroupMember.order == 2,
            )
        ).scalar_one()
        assert provider_asset_group_member_1 is not None
        assert provider_asset_group_member_2 is not None
        assert provider_asset_group_member_1.provider_id == kraken_provider.id
        assert provider_asset_group_member_1.from_asset_id == usd_asset.id
        assert provider_asset_group_member_1.to_asset_id == btc_asset.id
        assert provider_asset_group_member_1.order == 1
        assert provider_asset_group_member_2.provider_id == kraken_provider.id
        assert provider_asset_group_member_2.from_asset_id == usd_asset.id
        assert provider_asset_group_member_2.to_asset_id == eth_asset.id
        assert provider_asset_group_member_2.order == 2


@pytest.mark.asyncio
async def test_refresh_of_provider_asset_attribute_data():
    # Patch the step property to use 1 day instead of 1 hour for testing
    resolution = dt.timedelta(minutes=1)
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.step",
            property(lambda self: dt.timedelta(days=1)),
        ),
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.resolution",
            property(lambda self: resolution),
        ),
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            _,
            _,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

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

        # Create the time frame.
        start_time = (dt.datetime.now()).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        end_time = start_time + dt.timedelta(days=60)
        tf = pd.date_range(start=start_time, end=end_time, freq=resolution)

        # Create the first asset to be a geometric brownian motion.
        drift = 0.01
        volatility = 0.05
        S_initial_eth_to_usd = 10000
        gb_process = GeometricBrownianMotion(
            params=GBMParams(mu=drift, sigma=volatility)
        )
        S_eth_to_usd = gb_process.simulate(
            N=len(tf), N_simulated=1, X_0=S_initial_eth_to_usd
        )[0]

        # Initialize the parameters for Ornstein-Uhlenbeck process including the linear fit between two assets.
        alpha = 850
        beta = 4.5
        mu = 0.01
        sigma = 0.05
        theta = 0.0
        ou_process = OrnsteinUhlenbeck(params=OUParams(mu=mu, theta=theta, sigma=sigma))
        X_spread = ou_process.simulate(N=len(tf), N_simulated=1, X_0=theta)[0]
        S_btc_to_usd = alpha + beta * S_eth_to_usd + X_spread

        # Create provider asset market data over 30 days.
        df_btc_to_usd = pd.DataFrame(
            {
                "timestamp": tf,
                "provider_id": [kraken_provider.id for _ in tf],
                "from_asset_id": [usd_asset.id for _ in tf],
                "to_asset_id": [btc_asset.id for _ in tf],
                "close": S_btc_to_usd,
            }
        )
        df_eth_to_usd = pd.DataFrame(
            {
                "timestamp": tf,
                "provider_id": [kraken_provider.id for _ in tf],
                "from_asset_id": [usd_asset.id for _ in tf],
                "to_asset_id": [eth_asset.id for _ in tf],
                "close": S_eth_to_usd,
            }
        )
        df = pd.concat([df_btc_to_usd, df_eth_to_usd])

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group.
        await refresh_provider_asset_attribute_data(start=start_time, end=end_time)

        # Check if the provider asset group was created.
        with Session(engine) as session:
            provider_asset_group = (
                session.execute(
                    select(models.ProviderAssetGroup).options(
                        joinedload(models.ProviderAssetGroup.members)
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            assert provider_asset_group is not None
            assert len(provider_asset_group.members) == 2

        # Get provider asset attribute data.
        with Session(engine) as session:
            provider_asset_group_attributes_df = pl.read_database(
                query=select(models.ProviderAssetGroupAttribute).where(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id
                    == provider_asset_group.id
                ),
                connection=engine,
            )
            assert_within_tolerance(
                provider_asset_group_attributes_df["linear_fit_beta"].mean(),
                beta,
                tolerance=TOLERANCE,
            )
            assert_within_tolerance(
                provider_asset_group_attributes_df["linear_fit_alpha"].mean(),
                alpha,
                tolerance=TOLERANCE,
            )
            assert_within_tolerance(
                provider_asset_group_attributes_df["ol_theta"].mean(),
                theta,
                tolerance=TOLERANCE,
            )
            assert_within_tolerance(
                provider_asset_group_attributes_df["ol_mu"].mean(),
                mu,
                tolerance=TOLERANCE,
            )
            assert_within_tolerance(
                provider_asset_group_attributes_df["ol_sigma"].mean(),
                sigma,
                tolerance=TOLERANCE,
            )
            assert (
                provider_asset_group_attributes_df["cointegration_p_value"].mean()
                < 0.0001
            ), "The cointegration p-value is not less than 0.0001"
