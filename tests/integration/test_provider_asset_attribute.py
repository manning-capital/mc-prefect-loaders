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
            drift=0.05,
            volatility=0.2,
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
            drift=0.05,
            volatility=0.2,
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

            # Store the member IDs while the object is still attached to the session
            created_member_ids = {
                (m.provider_id, m.from_asset_id, m.to_asset_id)
                for m in created_provider_asset_group.members
            }

        # Refresh the provider asset attribute data.
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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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

        # Create cointegrated pair data with three assets.
        cointegrated_pair_df = generate_cointegrated_pair(
            n_points=1000,
            alpha=10.0,
            beta=1.5,
            theta=0.5,
            mu=0.1,
            sigma=2.0,
            start_price=100.0,
        )

        # Transform the cointegrated pair data to a provider asset market data for three assets.
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
                            sol_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "close": cointegrated_pair_df["close_1"]
                        * 0.8,  # Different price for SOL
                    }
                ),
            ]
        )

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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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

        # Create cointegrated pair data with four assets.
        cointegrated_pair_df = generate_cointegrated_pair(
            n_points=1000,
            alpha=10.0,
            beta=1.5,
            theta=0.5,
            mu=0.1,
            sigma=2.0,
            start_price=100.0,
        )

        # Transform the cointegrated pair data to a provider asset market data for four assets.
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
                            sol_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "close": cointegrated_pair_df["close_1"]
                        * 0.8,  # Different price for SOL
                    }
                ),
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
                            ada_asset.id for _ in cointegrated_pair_df["timestamp"]
                        ],
                        "close": cointegrated_pair_df["close_2"]
                        * 0.6,  # Different price for ADA
                    }
                ),
            ]
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Refresh the provider asset attribute data.
        await refresh_provider_asset_attribute_data(
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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
            start=cointegrated_pair_df["timestamp"].min(),
            end=cointegrated_pair_df["timestamp"].max(),
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

        # Create cointegrated pair data.
        alpha = 10.0
        beta = 1.5
        drift = 0.00005
        volatility = 0.0005
        theta = 0.0
        mu = 0.0001
        sigma = 0.005
        start_price = 100.0
        cointegrated_pair_df = generate_cointegrated_pair(
            n_points=31 * 24 * 60,
            alpha=alpha,
            beta=beta,
            drift=drift,
            volatility=volatility,
            theta=theta,
            mu=mu,
            sigma=sigma,
            start_price=start_price,
            resolution=resolution,
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
            start=cointegrated_pair_df["timestamp"].max() - dt.timedelta(hours=12),
            end=cointegrated_pair_df["timestamp"].max(),
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
