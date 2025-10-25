import os
import sys
import datetime as dt
from unittest.mock import patch

import pandas as pd
import pytest
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from tests.utils import (
    set_random_seed,
    sample_asset_data,
    sample_provider_data,
    assert_within_tolerance,
    generate_cointegrated_pair,
)
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
                models.ProviderAssetGroup.asset_group_type_id
                == pairs_trading_asset_group_type.id
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
async def test_creation_of_provider_asset_when_asset_group_does_not_exist():
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(hours=1)],
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

        # Create cointegrated pair data.
        cointegrated_pair_df = generate_cointegrated_pair(
            n_points=1000,
            alpha=10.0,
            beta=1.5,
            theta=0.5,
            mu=0.1,
            sigma=2.0,
            start_price=100.0,
        )

        # Transform the cointegrated pair data to a provider asset market data.
        df = pd.DataFrame(
            {
                "timestamp": cointegrated_pair_df["timestamp"],
                "provider_id": [
                    kraken_provider.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "from_asset_id": [
                    usd_asset.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "to_asset_id": [
                    btc_asset.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "close": cointegrated_pair_df["close_1"],
            }
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timestamp": cointegrated_pair_df["timestamp"],
                        "provider_id": [
                            kraken_provider.id
                            for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "from_asset_id": [
                            usd_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "to_asset_id": [
                            eth_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "close": cointegrated_pair_df["close_2"],
                    }
                ),
            ]
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group.
        await refresh_provider_asset_attribute_data(
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
        )

        # Check if the provider asset group was created.
        with Session(engine) as session:
            provider_asset_group = (
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalar_one_or_none()
            )
            assert provider_asset_group is not None
            assert len(provider_asset_group.members) == 2


@pytest.mark.asyncio
async def test_creation_of_provider_asset_when_asset_group_already_exists():
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(days=1)],
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

        # Create cointegrated pair data.
        cointegrated_pair_df = generate_cointegrated_pair(
            n_points=1000,
            alpha=10.0,
            beta=1.5,
            theta=0.5,
            mu=0.1,
            sigma=2.0,
            start_price=100.0,
        )

        # Transform the cointegrated pair data to a provider asset market data.
        df = pd.DataFrame(
            {
                "timestamp": cointegrated_pair_df["timestamp"],
                "provider_id": [
                    kraken_provider.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "from_asset_id": [
                    usd_asset.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "to_asset_id": [
                    btc_asset.id for _ in cointegrated_pair_df["timestamp"]
                ],
                "close": cointegrated_pair_df["close_1"],
            }
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timestamp": cointegrated_pair_df["timestamp"],
                        "provider_id": [
                            kraken_provider.id
                            for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "from_asset_id": [
                            usd_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "to_asset_id": [
                            eth_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "close": cointegrated_pair_df["close_2"],
                    }
                ),
            ]
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group.
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
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
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
        )

        # Check if the provider asset group was created.
        with Session(engine) as session:
            provider_asset_groups = session.execute(
                select(models.ProviderAssetGroup)
                .where(
                    models.ProviderAssetGroup.asset_group_type_id
                    == pairs_trading_asset_group_type.id
                )
                .options(joinedload(models.ProviderAssetGroup.members))
            ).scalars()
            assert len(provider_asset_groups) == 1
            provider_asset_group = provider_asset_groups[0]
            assert provider_asset_group.id == created_provider_asset_group.id
            assert provider_asset_group.is_active == created_provider_asset_group.is_active
            assert provider_asset_group.members == created_provider_asset_group.members


@pytest.mark.asyncio
async def test_parameter_recovery_of_statistical_pairs_trading_30_day_window():
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(days=30)],
    ):
        # Random seed for the geometric brownian motion and Ornstein-Uhlenbeck process.
        set_random_seed(48)

        # Set the resolution.
        resolution = dt.timedelta(minutes=1)

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
        end_time = start_time + dt.timedelta(days=30, hours=12)
        tf = pd.date_range(start=start_time, end=end_time, freq=resolution)

        # Create the first asset to be a geometric brownian motion.
        drift = 0.000001
        volatility = 0.0001
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

        # Add the asset group.
        with Session(engine) as session:
            provider_asset_group = models.ProviderAssetGroup(
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
            session.add(provider_asset_group)
            session.commit()
            session.refresh(provider_asset_group)

        # Create the provider asset group.
        await refresh_provider_asset_attribute_data(
            start=end_time - dt.timedelta(hours=12), end=end_time
        )

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
            provider_asset_group_attributes_df = pd.read_sql(
                select(models.ProviderAssetGroupAttribute).where(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id
                    == provider_asset_group.id
                ),
                con=engine,
            )
            linear_fit_beta = provider_asset_group_attributes_df[
                "linear_fit_beta"
            ].mean()
            linear_fit_alpha = provider_asset_group_attributes_df[
                "linear_fit_alpha"
            ].mean()
            ou_theta = provider_asset_group_attributes_df["ou_theta"].mean()
            ou_mu = provider_asset_group_attributes_df["ou_mu"].mean()
            ou_sigma = provider_asset_group_attributes_df["ou_sigma"].mean()
            cointegration_p_value = provider_asset_group_attributes_df[
                "cointegration_p_value"
            ].mean()
            assert_within_tolerance(linear_fit_beta, beta, tolerance=TOLERANCE)
            assert_within_tolerance(linear_fit_alpha, alpha, tolerance=TOLERANCE)
            assert_within_tolerance(ou_theta, theta, tolerance=TOLERANCE)
            assert_within_tolerance(ou_mu, mu, tolerance=TOLERANCE)
            assert_within_tolerance(ou_sigma, sigma, tolerance=TOLERANCE)
            assert_within_tolerance(cointegration_p_value, 0.0001, tolerance=TOLERANCE)
            assert_within_tolerance(ou_sigma, sigma, tolerance=TOLERANCE)
            assert_within_tolerance(cointegration_p_value, 0.0001, tolerance=TOLERANCE)
