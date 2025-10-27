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
    generate_market_data_dataframe,
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
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
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
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

            # Store the member IDs while the object is still attached to the session
            created_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in created_provider_asset_group.members
            }

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
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
            assert (
                provider_asset_group.asset_group_type_id
                == pairs_trading_asset_group_type.id
            )
            assert (
                provider_asset_group.is_active == created_provider_asset_group.is_active
            )

            # Compare members by their IDs to avoid DetachedInstanceError
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in provider_asset_group.members
            }
            assert provider_member_ids == created_member_ids


@pytest.mark.asyncio
async def test_group_not_duplicated_with_reversed_order_values():
    """Test that groups with reversed order values are recognized as identical."""
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,  # Added default drift
                "volatility": 0.2,  # Added default volatility
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group with reversed order values (2,1 instead of 1,2).
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,
                        order=2,  # Reversed order
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=eth_asset.id,
                        order=1,  # Reversed order
                    ),
                ],
            )
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that only 1 group exists (no duplicates created).
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            assert len(all_groups) == 1, f"Expected 1 group, found {len(all_groups)}"

            # Verify member IDs match (order-independent)
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in all_groups[0].members
            }
            expected_member_ids = {
                (kraken_provider.id, usd_asset.id, btc_asset.id),
                (kraken_provider.id, usd_asset.id, eth_asset.id),
            }
            assert provider_member_ids == expected_member_ids


@pytest.mark.asyncio
async def test_group_not_duplicated_with_different_order_values():
    """Test that groups with different order values are recognized as identical."""
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,  # Added default drift
                "volatility": 0.2,  # Added default volatility
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group with different order values (5,10 instead of 1,2).
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,
                        order=5,  # Different order
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=eth_asset.id,
                        order=10,  # Different order
                    ),
                ],
            )
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that only 1 group exists (no duplicates created).
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            assert len(all_groups) == 1, f"Expected 1 group, found {len(all_groups)}"

            # Verify member IDs match (order-independent)
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in all_groups[0].members
            }
            expected_member_ids = {
                (kraken_provider.id, usd_asset.id, btc_asset.id),
                (kraken_provider.id, usd_asset.id, eth_asset.id),
            }
            assert provider_member_ids == expected_member_ids


@pytest.mark.asyncio
async def test_group_not_duplicated_with_reversed_member_list():
    """Test that groups with reversed member list order are recognized as identical."""
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,  # Added default drift
                "volatility": 0.2,  # Added default volatility
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group with reversed member list order (ETH first, then BTC).
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=eth_asset.id,  # ETH first
                        order=1,
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,  # BTC second
                        order=2,
                    ),
                ],
            )
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that only 1 group exists (no duplicates created).
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            assert len(all_groups) == 1, f"Expected 1 group, found {len(all_groups)}"

            # Verify member IDs match (order-independent)
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in all_groups[0].members
            }
            expected_member_ids = {
                (kraken_provider.id, usd_asset.id, btc_asset.id),
                (kraken_provider.id, usd_asset.id, eth_asset.id),
            }
            assert provider_member_ids == expected_member_ids


@pytest.mark.asyncio
async def test_group_not_duplicated_with_reversed_list_and_different_orders():
    """Test that groups with both reversed list order and different order values are recognized as identical."""
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,  # Added default drift
                "volatility": 0.2,  # Added default volatility
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group with reversed list order AND different order values.
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=eth_asset.id,  # ETH first
                        order=10,  # Different order
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,  # BTC second
                        order=5,  # Different order
                    ),
                ],
            )
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that only 1 group exists (no duplicates created).
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            assert len(all_groups) == 1, f"Expected 1 group, found {len(all_groups)}"

            # Verify member IDs match (order-independent)
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in all_groups[0].members
            }
            expected_member_ids = {
                (kraken_provider.id, usd_asset.id, btc_asset.id),
                (kraken_provider.id, usd_asset.id, eth_asset.id),
            }
            assert provider_member_ids == expected_member_ids


@pytest.mark.asyncio
async def test_group_not_duplicated_with_three_members_mixed_ordering():
    """Test that pairs trading correctly handles multiple pairs without creating duplicates."""
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(days=1)],
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            crypto_asset_type,
            _,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create a third asset (SOL) for this test
        with Session(engine) as session:
            sol_asset = models.Asset(
                symbol="SOL",
                name="Solana",
                description="Solana cryptocurrency",
                asset_type_id=crypto_asset_type.id,
                is_active=True,
            )
            session.add(sol_asset)
            session.commit()
            session.refresh(sol_asset)

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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD) plus SOL
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id, sol_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,  # Only BTC-ETH are cointegrated
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
        )

        # Adjust SOL prices to be different (multiply by 0.8)
        sol_mask = df["to_asset_id"] == sol_asset.id
        df.loc[sol_mask, "close"] = df.loc[sol_mask, "close"] * 0.8

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create two provider asset groups with different member orderings.
        # Group 1: BTC-ETH pair with order 1,2
        # Group 2: BTC-SOL pair with order 2,1 (reversed)
        with Session(engine) as session:
            # First pair: BTC-ETH with order 1,2
            btc_eth_group = models.ProviderAssetGroup(
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
            session.add(btc_eth_group)

            # Second pair: BTC-SOL with order 2,1 (reversed)
            btc_sol_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,
                        order=2,  # Reversed order
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=sol_asset.id,
                        order=1,  # Reversed order
                    ),
                ],
            )
            session.add(btc_sol_group)
            session.commit()
            session.refresh(btc_eth_group)
            session.refresh(btc_sol_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that the system creates pairs trading groups correctly.
        # For 3 assets, StatisticalPairsTrading should create 3 pairs: BTC-ETH, BTC-SOL, ETH-SOL
        # The system should recognize that our manually created groups are equivalent to the desired pairs
        # and not create duplicates, so we expect exactly 3 groups total.
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            # Should have exactly 3 groups: BTC-ETH, BTC-SOL, ETH-SOL pairs
            assert len(all_groups) == 3, (
                f"Expected 3 groups (all pairs), found {len(all_groups)}"
            )

            # Verify that all expected pairs exist (order-independent)
            expected_pairs = [
                {
                    (kraken_provider.id, usd_asset.id, btc_asset.id),
                    (kraken_provider.id, usd_asset.id, eth_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, btc_asset.id),
                    (kraken_provider.id, usd_asset.id, sol_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, eth_asset.id),
                    (kraken_provider.id, usd_asset.id, sol_asset.id),
                },
            ]

            found_pairs = []
            for group in all_groups:
                member_ids = {
                    (m.provider_id, m.from_asset_id, m.to_asset_id)
                    for m in group.members
                }
                found_pairs.append(member_ids)

            # Check that all expected pairs are present
            for expected_pair in expected_pairs:
                assert expected_pair in found_pairs, (
                    f"Expected pair {expected_pair} not found in {found_pairs}"
                )


@pytest.mark.asyncio
async def test_group_not_duplicated_with_four_assets():
    """Test that pairs trading correctly creates all possible pairs for 4 assets (C(4,2) = 6 pairs)."""
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(days=1)],
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            crypto_asset_type,
            _,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create two additional assets (SOL and ADA) for this test
        with Session(engine) as session:
            sol_asset = models.Asset(
                symbol="SOL",
                name="Solana",
                description="Solana cryptocurrency",
                asset_type_id=crypto_asset_type.id,
                is_active=True,
            )
            ada_asset = models.Asset(
                symbol="ADA",
                name="Cardano",
                description="Cardano cryptocurrency",
                asset_type_id=crypto_asset_type.id,
                is_active=True,
            )
            session.add(sol_asset)
            session.add(ada_asset)
            session.commit()
            session.refresh(sol_asset)
            session.refresh(ada_asset)

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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD) plus SOL and ADA
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id, sol_asset.id, ada_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,  # Only BTC-ETH are cointegrated
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
        )

        # Adjust SOL and ADA prices to be different
        sol_mask = df["to_asset_id"] == sol_asset.id
        df.loc[sol_mask, "close"] = df.loc[sol_mask, "close"] * 0.8

        ada_mask = df["to_asset_id"] == ada_asset.id
        df.loc[ada_mask, "close"] = df.loc[ada_mask, "close"] * 0.6

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that the system creates all possible pairs for 4 assets.
        # For 4 assets, StatisticalPairsTrading should create C(4,2) = 6 pairs:
        # BTC-ETH, BTC-SOL, BTC-ADA, ETH-SOL, ETH-ADA, SOL-ADA
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            # Should have exactly 6 groups: all possible pairs
            assert len(all_groups) == 6, (
                f"Expected 6 groups (C(4,2) pairs), found {len(all_groups)}"
            )

            # Verify that all expected pairs exist (order-independent)
            expected_pairs = [
                {
                    (kraken_provider.id, usd_asset.id, btc_asset.id),
                    (kraken_provider.id, usd_asset.id, eth_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, btc_asset.id),
                    (kraken_provider.id, usd_asset.id, sol_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, btc_asset.id),
                    (kraken_provider.id, usd_asset.id, ada_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, eth_asset.id),
                    (kraken_provider.id, usd_asset.id, sol_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, eth_asset.id),
                    (kraken_provider.id, usd_asset.id, ada_asset.id),
                },
                {
                    (kraken_provider.id, usd_asset.id, sol_asset.id),
                    (kraken_provider.id, usd_asset.id, ada_asset.id),
                },
            ]

            found_pairs = []
            for group in all_groups:
                member_ids = {
                    (m.provider_id, m.from_asset_id, m.to_asset_id)
                    for m in group.members
                }
                found_pairs.append(member_ids)

            # Check that all expected pairs are present
            for expected_pair in expected_pairs:
                assert expected_pair in found_pairs, (
                    f"Expected pair {expected_pair} not found in {found_pairs}"
                )

            # Verify no unexpected pairs exist
            for found_pair in found_pairs:
                assert found_pair in expected_pairs, (
                    f"Unexpected pair {found_pair} found"
                )


@pytest.mark.asyncio
async def test_group_not_duplicated_with_same_order_values():
    """Test that groups where all members have the same order value are handled gracefully by filtering duplicates."""
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,  # Added default drift
                "volatility": 0.2,  # Added default volatility
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset group where all members have the same order value.
        with Session(engine) as session:
            created_provider_asset_group = models.ProviderAssetGroup(
                asset_group_type_id=pairs_trading_asset_group_type.id,
                is_active=True,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=btc_asset.id,
                        order=1,  # Same order
                    ),
                    models.ProviderAssetGroupMember(
                        provider_id=kraken_provider.id,
                        from_asset_id=usd_asset.id,
                        to_asset_id=eth_asset.id,
                        order=1,  # Same order
                    ),
                ],
            )
            session.add(created_provider_asset_group)
            session.commit()
            session.refresh(created_provider_asset_group)

        # Refresh the provider asset attribute data - this should succeed by filtering duplicate order values
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check that only 1 group exists (no duplicates created).
        with Session(engine) as session:
            all_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup)
                    .where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                    .options(joinedload(models.ProviderAssetGroup.members))
                )
                .unique()
                .scalars()
            )

            assert len(all_groups) == 1, f"Expected 1 group, found {len(all_groups)}"

            # Verify member IDs match (order-independent)
            provider_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in all_groups[0].members
            }
            expected_member_ids = {
                (kraken_provider.id, usd_asset.id, btc_asset.id),
                (kraken_provider.id, usd_asset.id, eth_asset.id),
            }
            assert provider_member_ids == expected_member_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "window_days,data_days,seed",
    [
        (30, 31, 48),  # 30-day window with 31 days of data
        (60, 61, 49),  # 60-day window with 61 days of data
        (90, 91, 50),  # 90-day window with 91 days of data
    ],
)
async def test_parameter_recovery_of_statistical_pairs_trading(
    window_days: int, data_days: int, seed: int
):
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(days=window_days)],
    ):
        # Random seed for the geometric brownian motion and Ornstein-Uhlenbeck process.
        set_random_seed(seed)

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

        # Create the group ahead of time so that it's in the right order.
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

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        alpha = 10.0
        beta = 1.5
        drift = 0.00005
        volatility = 0.0005
        theta = 0.0
        mu = 0.01
        sigma = 0.005
        start_price = 100.0

        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=data_days * 24 * 60,  # Sufficient data for the window
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": alpha,
                "beta": beta,
                "drift": drift,
                "volatility": volatility,
                "theta": theta,
                "mu": mu,
                "sigma": sigma,
                "start_price": start_price,
            },
            resolution=resolution,
        )

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
            start=df["timestamp"].max() - dt.timedelta(hours=12),
            end=df["timestamp"].max(),
        )

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
            assert_within_tolerance(ou_sigma, sigma, tolerance=TOLERANCE)
            assert_within_tolerance(ou_mu, mu, tolerance=TOLERANCE)
            assert cointegration_p_value < 0.001, (
                f"Cointegration p-value {cointegration_p_value} should be < 0.001"
            )


@pytest.mark.asyncio
async def test_pairs_only_formed_with_same_from_asset():
    """
    Test that pairs are only formed when both members have the same from_asset.
    This ensures pairs trading logic is meaningful (e.g., BTC/USD and ETH/USD can be paired,
    but BTC/USD and ETH/EUR cannot be paired).
    """
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
        df_usd = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
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
        )

        # Generate market data for EUR pairs (BTC/EUR and ETH/EUR)
        df_eur = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
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
        )

        # Adjust EUR prices to be different (multiply by 0.85)
        df_eur["close"] = df_eur["close"] * 0.85

        # Combine both dataframes
        df = pd.concat([df_usd, df_eur], ignore_index=True)

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset groups.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Verify that exactly 2 pairs were created (one for USD, one for EUR)
        with Session(engine) as session:
            provider_asset_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup).where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                )
                .scalars()
                .unique()
            )

            assert len(provider_asset_groups) == 2, (
                f"Expected exactly 2 pairs (USD and EUR), but got {len(provider_asset_groups)}"
            )

            # Verify each pair has members with the same from_asset
            for group in provider_asset_groups:
                members = group.members
                assert len(members) == 2, f"Each pair should have exactly 2 members"

                # All members in a pair should have the same from_asset_id
                from_asset_ids = [member.from_asset_id for member in members]
                assert len(set(from_asset_ids)) == 1, (
                    f"All members in a pair should have the same from_asset_id, "
                    f"but got {from_asset_ids}"
                )

                # Verify the pair contains BTC and ETH
                to_asset_ids = [member.to_asset_id for member in members]
                assert btc_asset.id in to_asset_ids, "Pair should contain BTC"
                assert eth_asset.id in to_asset_ids, "Pair should contain ETH"

            # Verify we have one USD pair and one EUR pair
            usd_groups = [
                g
                for g in provider_asset_groups
                if g.members[0].from_asset_id == usd_asset.id
            ]
            eur_groups = [
                g
                for g in provider_asset_groups
                if g.members[0].from_asset_id == eur_asset.id
            ]

            assert len(usd_groups) == 1, "Should have exactly one USD pair"
            assert len(eur_groups) == 1, "Should have exactly one EUR pair"


@pytest.mark.asyncio
async def test_no_pairs_formed_with_different_from_assets():
    """
    Test that when assets have different from_assets, no pairs are formed.
    This ensures the constraint is properly enforced.
    """
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

        # Create market data where each asset has a different from_asset
        # BTC/USD, ETH/EUR - these should NOT form a pair
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
                "provider_id": [kraken_provider.id for _ in range(1000)],
                "from_asset_id": [usd_asset.id for _ in range(1000)],
                "to_asset_id": [btc_asset.id for _ in range(1000)],
                "close": [100.0 + i * 0.01 for i in range(1000)],
            }
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            "2024-01-01", periods=1000, freq="1min"
                        ),
                        "provider_id": [kraken_provider.id for _ in range(1000)],
                        "from_asset_id": [eur_asset.id for _ in range(1000)],
                        "to_asset_id": [eth_asset.id for _ in range(1000)],
                        "close": [200.0 + i * 0.02 for i in range(1000)],
                    }
                ),
            ]
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset groups.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Verify that NO pairs were created since BTC/USD and ETH/EUR have different from_assets
        with Session(engine) as session:
            provider_asset_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup).where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                )
                .scalars()
                .unique()
            )

            assert len(provider_asset_groups) == 0, (
                f"Expected no pairs since BTC/USD and ETH/EUR have different from_assets, "
                f"but got {len(provider_asset_groups)} pairs"
            )


@pytest.mark.asyncio
async def test_multiple_providers_with_same_from_asset():
    """
    Test that pairs can be formed across different providers as long as they have the same from_asset.
    This ensures the constraint is based on from_asset, not provider.
    """
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(hours=1)],
    ):
        # Get the engine.
        engine = await get_engine()

        # Create multiple providers and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        _, binance_provider = await sample_provider_data(engine)
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

        # Create market data with different providers but same from_asset (USD)
        # Kraken BTC/USD, Binance ETH/USD - these should form a pair
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
                "provider_id": [kraken_provider.id for _ in range(1000)],
                "from_asset_id": [usd_asset.id for _ in range(1000)],
                "to_asset_id": [btc_asset.id for _ in range(1000)],
                "close": [100.0 + i * 0.01 for i in range(1000)],
            }
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            "2024-01-01", periods=1000, freq="1min"
                        ),
                        "provider_id": [binance_provider.id for _ in range(1000)],
                        "from_asset_id": [usd_asset.id for _ in range(1000)],
                        "to_asset_id": [eth_asset.id for _ in range(1000)],
                        "close": [200.0 + i * 0.02 for i in range(1000)],
                    }
                ),
            ]
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset groups.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Verify that exactly 1 pair was created (Kraken BTC/USD + Binance ETH/USD)
        with Session(engine) as session:
            provider_asset_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup).where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                )
                .scalars()
                .unique()
            )

            assert len(provider_asset_groups) == 1, (
                f"Expected exactly 1 pair (Kraken BTC/USD + Binance ETH/USD), "
                f"but got {len(provider_asset_groups)}"
            )

            # Verify the pair has members from different providers but same from_asset
            group = provider_asset_groups[0]
            members = group.members
            assert len(members) == 2, f"Pair should have exactly 2 members"

            # All members should have the same from_asset_id (USD)
            from_asset_ids = [member.from_asset_id for member in members]
            assert len(set(from_asset_ids)) == 1, (
                f"All members should have the same from_asset_id (USD), "
                f"but got {from_asset_ids}"
            )
            assert from_asset_ids[0] == usd_asset.id, "from_asset_id should be USD"

            # Members should have different provider_ids
            provider_ids = [member.provider_id for member in members]
            assert len(set(provider_ids)) == 2, (
                f"Members should have different provider_ids, but got {provider_ids}"
            )
            assert kraken_provider.id in provider_ids, "Should include Kraken provider"
            assert binance_provider.id in provider_ids, (
                "Should include Binance provider"
            )


@pytest.mark.asyncio
async def test_insufficient_members_for_pairing():
    """
    Test that when there's only one asset with a particular from_asset, no pairs are formed.
    This ensures the system handles edge cases gracefully.
    """
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
            _,
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

        # Create market data with only one asset (BTC/USD)
        # This should result in no pairs since we need at least 2 assets with the same from_asset
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
                "provider_id": [kraken_provider.id for _ in range(1000)],
                "from_asset_id": [usd_asset.id for _ in range(1000)],
                "to_asset_id": [btc_asset.id for _ in range(1000)],
                "close": [100.0 + i * 0.01 for i in range(1000)],
            }
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create the provider asset groups.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Verify that NO pairs were created since there's only one asset
        with Session(engine) as session:
            provider_asset_groups = list(
                session.execute(
                    select(models.ProviderAssetGroup).where(
                        models.ProviderAssetGroup.asset_group_type_id
                        == pairs_trading_asset_group_type.id
                    )
                )
                .scalars()
                .unique()
            )

            assert len(provider_asset_groups) == 0, (
                f"Expected no pairs since there's only one asset (BTC/USD), "
                f"but got {len(provider_asset_groups)} pairs"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step,window,start_time,n_points,validator",
    [
        (
            dt.timedelta(hours=1),
            dt.timedelta(hours=6),
            dt.datetime(2025, 1, 1, 0, 30, 43),
            500,
            lambda ts: (ts.minute == 0, f"Timestamp {ts} should have minute=0"),
        ),
        (
            dt.timedelta(minutes=15),
            dt.timedelta(hours=1),
            dt.datetime(2025, 1, 1, 0, 7, 23),
            200,
            lambda ts: (
                ts.minute in [0, 15, 30, 45],
                f"Timestamp {ts} should have minute in [0, 15, 30, 45], got {ts.minute}",
            ),
        ),
        (
            dt.timedelta(hours=6),
            dt.timedelta(days=1),
            dt.datetime(2025, 1, 1, 2, 15, 47),
            1000,
            lambda ts: (
                ts.hour in [0, 6, 12, 18],
                f"Timestamp {ts} should have hour in [0, 6, 12, 18], got {ts.hour}",
            ),
        ),
        (
            dt.timedelta(days=1),
            dt.timedelta(days=7),
            dt.datetime(2025, 1, 1, 14, 32, 15),
            2000,
            lambda ts: (ts.hour == 0, f"Timestamp {ts} should have hour=0"),
        ),
    ],
)
async def test_timestamp_alignment(step, window, start_time, n_points, validator):
    """Test that timestamps are properly aligned to step boundaries."""
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [window],
        ),
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.step",
            new_callable=lambda: step,
        ),
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.resolution",
            new_callable=lambda: dt.timedelta(minutes=1),
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

        # Generate market data starting at a misaligned time
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=n_points,
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
        )
        df["timestamp"] = pd.date_range(start=start_time, periods=len(df), freq="1min")

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

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
            session.refresh(provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
        )

        # Check the timestamps in ProviderAssetGroupAttribute
        with Session(engine) as session:
            attributes_df = pd.read_sql(
                select(models.ProviderAssetGroupAttribute).where(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id
                    == provider_asset_group.id
                ),
                con=engine,
            )

        # Verify all timestamps are properly aligned
        for timestamp in attributes_df["timestamp"]:
            condition, error_msg = validator(timestamp)
            assert condition, error_msg

            # These checks apply to all timestamps
            assert timestamp.second == 0, f"Timestamp {timestamp} should have second=0"
            assert timestamp.microsecond == 0, (
                f"Timestamp {timestamp} should have microsecond=0"
            )


@pytest.mark.asyncio
async def test_misaligned_input_range_with_1hour_step():
    """Test specific example: start=2025-01-01 00:30:43, end=2025-01-01 04:30:43, step=1 hour
    Verify generated timestamps are 01:00:00, 02:00:00, 03:00:00, 04:00:00."""
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [dt.timedelta(hours=6)],
        ),
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.step",
            new_callable=lambda: dt.timedelta(hours=1),
        ),
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.resolution",
            new_callable=lambda: dt.timedelta(minutes=1),
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

        # Generate market data for the exact scenario: start=00:30:43, end=04:30:43
        start_time = dt.datetime(2025, 1, 1, 0, 30, 43)
        end_time = dt.datetime(2025, 1, 1, 4, 30, 43)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=240,  # 4 hours worth of 1-minute data
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
        )
        df["timestamp"] = pd.date_range(start=start_time, end=end_time, freq="1min")

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

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
            session.refresh(provider_asset_group)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=start_time,
            end=end_time,
        )

        # Check the timestamps in ProviderAssetGroupAttribute
        with Session(engine) as session:
            attributes_df = pd.read_sql(
                select(models.ProviderAssetGroupAttribute).where(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id
                    == provider_asset_group.id
                ),
                con=engine,
            )

        # Verify timestamps are exactly 01:00:00, 02:00:00, 03:00:00, 04:00:00
        expected_timestamps = {
            dt.datetime(2025, 1, 1, 1, 0, 0),
            dt.datetime(2025, 1, 1, 2, 0, 0),
            dt.datetime(2025, 1, 1, 3, 0, 0),
            dt.datetime(2025, 1, 1, 4, 0, 0),
        }

        actual_timestamps = set(attributes_df["timestamp"].dt.to_pydatetime())

        # Check that expected timestamps are present (allowing for more if window captures more data)
        assert expected_timestamps.issubset(actual_timestamps), (
            f"Expected at least {expected_timestamps}, but got {actual_timestamps}"
        )

        # Verify all timestamps are aligned to hour boundaries
        for timestamp in actual_timestamps:
            assert timestamp.minute == 0, f"Timestamp {timestamp} should have minute=0"
            assert timestamp.second == 0, f"Timestamp {timestamp} should have second=0"
            assert timestamp.microsecond == 0, (
                f"Timestamp {timestamp} should have microsecond=0"
            )
