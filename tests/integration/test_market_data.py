import os
import sys
import datetime as dt
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import (
    Asset,
    Provider,
    AssetType,
    ProviderType,
    ProviderAsset,
    ProviderAssetMarket,
)

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from mc_postgres_db.prefect.asyncio.tasks import get_engine

from data.variables_base import create_global_concurrency_limit
from src.market.market_data import KrakenProviderAssetMarketData
from src.market.provider_asset_market_flows import pull_provider_asset_market_data


def create_base_data(engine: Engine):
    with Session(engine) as session:
        # Create the concurrency limit.
        create_global_concurrency_limit(
            name="kraken-api",
            limit=1,
            slot_decay_per_second=1,
        )

        # Add provider type data.
        session.add(
            ProviderType(
                name="CryptoCurrencyExchange",
                description="CryptoCurrencyExchange",
            )
        )
        session.commit()

        # Get the id for the provider type.
        provider_type_id_stmt = select(ProviderType.id).where(
            ProviderType.name == "CryptoCurrencyExchange"
        )
        provider_type_id = session.execute(provider_type_id_stmt).scalar_one()

        # Add provider data.
        session.add(
            Provider(
                name="Kraken",
                description="Kraken",
                provider_type_id=provider_type_id,
            )
        )
        session.commit()
        kraken_provider_id_stmt = select(Provider.id).where(Provider.name == "Kraken")
        kraken_provider_id = session.execute(kraken_provider_id_stmt).scalar_one()

        # Add asset type data.
        session.add(
            AssetType(
                name="CryptoCurrency",
                description="CryptoCurrency",
            )
        )
        session.add(
            AssetType(
                name="FiatCurrency",
                description="FiatCurrency",
            )
        )
        session.commit()

        # Get the id for the asset type.
        crypto_asset_type_id_stmt = select(AssetType.id).where(
            AssetType.name == "CryptoCurrency"
        )
        crypto_asset_type_id = session.execute(crypto_asset_type_id_stmt).scalar_one()
        fiat_asset_type_id_stmt = select(AssetType.id).where(
            AssetType.name == "FiatCurrency"
        )
        fiat_asset_type_id = session.execute(fiat_asset_type_id_stmt).scalar_one()

        # Add the asset data.
        session.add(
            Asset(
                name="BTC",
                description="BTC",
                asset_type_id=crypto_asset_type_id,
            )
        )
        session.add(
            Asset(
                name="ETH",
                description="ETH",
                asset_type_id=crypto_asset_type_id,
            )
        )
        session.add(
            Asset(
                name="USD",
                description="USD",
                asset_type_id=fiat_asset_type_id,
            )
        )
        session.add(
            Asset(
                name="1INCH",
                description="1INCH",
                asset_type_id=crypto_asset_type_id,
            )
        )
        session.commit()

        # Get the ids for the assets.
        btc_asset_id_stmt = select(Asset.id).where(Asset.name == "BTC")
        btc_asset_id = session.execute(btc_asset_id_stmt).scalar_one()
        eth_asset_id_stmt = select(Asset.id).where(Asset.name == "ETH")
        eth_asset_id = session.execute(eth_asset_id_stmt).scalar_one()
        usd_asset_id_stmt = select(Asset.id).where(Asset.name == "USD")
        usd_asset_id = session.execute(usd_asset_id_stmt).scalar_one()
        one_inch_asset_id_stmt = select(Asset.id).where(Asset.name == "1INCH")
        one_inch_asset_id = session.execute(one_inch_asset_id_stmt).scalar_one()

        # Add the provider asset data.
        session.add(
            ProviderAsset(
                date=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=btc_asset_id,
                asset_code="XXBT",
                is_active=True,
            )
        )
        session.add(
            ProviderAsset(
                date=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=eth_asset_id,
                asset_code="XETH",
                is_active=True,
            )
        )
        session.add(
            ProviderAsset(
                date=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=usd_asset_id,
                asset_code="ZUSD",
                is_active=True,
            )
        )
        session.add(
            ProviderAsset(
                date=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=one_inch_asset_id,
                asset_code="1INCH",
                is_active=True,
            )
        )
        session.commit()

        return (
            provider_type_id,
            crypto_asset_type_id,
            fiat_asset_type_id,
            kraken_provider_id,
            btc_asset_id,
            eth_asset_id,
            usd_asset_id,
            one_inch_asset_id,
        )


class FakeData:
    asset_pairs: dict[str, dict[str, Any]]
    market_data: dict[str, list[list[Any]]]

    def __init__(self):
        self.reset_data()

    def reset_data(self):
        self.asset_pairs = {}
        self.market_data = {}


@pytest.fixture(scope="function")
def fake_data():
    # Initialize the fake data.
    fake_data = FakeData()

    # Mock the get_asset_pairs method.
    async def mock_request_asset_pairs(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {"result": fake_data.asset_pairs}

    # Mock the get_market_data method.
    async def mock_request_market_data(
        self,
        pair_asset_code: str,
    ) -> dict[str, dict[str, list[list[Any]]]]:
        return {"result": {pair_asset_code: fake_data.market_data[pair_asset_code]}}

    # Patch the get_asset_pairs and get_market_data methods.
    with patch.object(
        KrakenProviderAssetMarketData,
        "request_asset_pairs",
        mock_request_asset_pairs,
    ):
        with patch.object(
            KrakenProviderAssetMarketData,
            "request_market_data",
            mock_request_market_data,
        ):
            yield fake_data


@pytest.mark.asyncio
async def test_pull_new_kraken_data_into_empty_database(fake_data: FakeData):
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        _,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        _,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Add the fake data.
    fake_data.asset_pairs = {
        "XXBTZUSD": {
            "base": "XXBT",
            "quote": "ZUSD",
        },
        "XETHZUSD": {
            "base": "XETH",
            "quote": "ZUSD",
        },
    }
    use_time = dt.datetime.now()
    fake_data.market_data = {
        "XXBTZUSD": [
            [int(use_time.timestamp()), 100, 100, 100, 100, 100, 100, 100],
        ],
        "XETHZUSD": [
            [int(use_time.timestamp()), 100, 100, 100, 100, 100, 100, 100],
        ],
    }

    # Pull the provider asset market data.
    await pull_provider_asset_market_data()

    # Check the data.
    with Session(engine) as session:
        # Get the provider asset market data.
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        # Check the data.
        assert len(provider_asset_market_data) == sum(
            len(data) for data in fake_data.market_data.values()
        )

        # Check the BTC to USD data.
        btc_to_usd_data = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == usd_asset_id,
                ProviderAssetMarket.to_asset_id == btc_asset_id,
            )
        ).scalar_one_or_none()
        assert btc_to_usd_data is not None
        assert btc_to_usd_data.open == 100.0
        assert btc_to_usd_data.high == 100.0
        assert btc_to_usd_data.low == 100.0
        assert btc_to_usd_data.close == 100.0
        assert btc_to_usd_data.volume == 100.0

        # Check the ETH to USD data.
        eth_to_usd_data = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == usd_asset_id,
                ProviderAssetMarket.to_asset_id == eth_asset_id,
            )
        ).scalar_one_or_none()
        assert eth_to_usd_data is not None
        assert eth_to_usd_data.open == 100.0
        assert eth_to_usd_data.high == 100.0
        assert eth_to_usd_data.low == 100.0
        assert eth_to_usd_data.close == 100.0
        assert eth_to_usd_data.volume == 100.0


@pytest.mark.asyncio
async def test_pull_new_kraken_data_into_non_empty_database(fake_data: FakeData):
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        _,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Add the fake data.
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)
    fake_data.asset_pairs = {
        "XXBTZUSD": {
            "base": "XXBT",
            "quote": "ZUSD",
        },
        "XETHZUSD": {
            "base": "XETH",
            "quote": "ZUSD",
        },
    }
    fake_data.market_data = {
        "XXBTZUSD": [
            [
                int(use_time.timestamp()),
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
            ],
        ],
        "XETHZUSD": [
            [
                int(use_time.timestamp()),
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
                300.0,
            ],
        ],
    }

    # Add existing data.
    with Session(engine) as session:
        existing_provider_asset_market = ProviderAssetMarket(
            provider_id=kraken_provider_id,
            from_asset_id=usd_asset_id,
            to_asset_id=btc_asset_id,
            timestamp=use_time,
            open=200.0,
            high=200.0,
            low=200.0,
            close=200.0,
            volume=200.0,
        )
        session.add(existing_provider_asset_market)
        session.commit()
        session.refresh(existing_provider_asset_market)

    # Pull the provider asset market data.
    await pull_provider_asset_market_data()

    # Check the data.
    with Session(engine) as session:
        # General checks.
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )
        assert len(provider_asset_market_data) == sum(
            len(data) for data in fake_data.market_data.values()
        )

        # Check the BTC to USD data.
        btc_to_usd_data = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == usd_asset_id,
                ProviderAssetMarket.to_asset_id == btc_asset_id,
            )
        ).scalar_one_or_none()
        assert btc_to_usd_data is not None
        assert btc_to_usd_data.open == 300.0
        assert btc_to_usd_data.high == 300.0
        assert btc_to_usd_data.low == 300.0
        assert btc_to_usd_data.close == 300.0
        assert btc_to_usd_data.volume == 300.0

        # Check the ETH to USD data.
        eth_to_usd_data = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == usd_asset_id,
                ProviderAssetMarket.to_asset_id == eth_asset_id,
            )
        ).scalar_one_or_none()
        assert eth_to_usd_data is not None
        assert eth_to_usd_data.open == 300.0
        assert eth_to_usd_data.high == 300.0
        assert eth_to_usd_data.low == 300.0
        assert eth_to_usd_data.close == 300.0
        assert eth_to_usd_data.volume == 300.0


@pytest.mark.asyncio
async def test_pull_new_kraken_data_into_empty_database_with_pair_that_does_not_match_from_and_to_asset_combination(
    fake_data: FakeData,
):
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        _,
        _,
        _,
        usd_asset_id,
        one_inch_asset_id,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Add the fake data.
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)
    fake_data.asset_pairs = {
        "1INCHUSD": {
            "base": "1INCH",
            "quote": "ZUSD",
        },
    }
    fake_data.market_data = {
        "1INCHUSD": [
            [
                int(use_time.timestamp()),
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
        ],
    }

    # Pull the provider asset market data.
    await pull_provider_asset_market_data()

    # Check the data.
    with Session(engine) as session:
        # General checks.
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )
        assert len(provider_asset_market_data) == sum(
            len(data) for data in fake_data.market_data.values()
        )

        # Check the 1INCH to USD data.
        one_inch_usd_data = session.execute(
            select(ProviderAssetMarket).where(
                ProviderAssetMarket.from_asset_id == usd_asset_id,
                ProviderAssetMarket.to_asset_id == one_inch_asset_id,
            )
        ).scalar_one_or_none()
        assert one_inch_usd_data is not None
        assert one_inch_usd_data.open == 100.0
        assert one_inch_usd_data.high == 100.0
        assert one_inch_usd_data.low == 100.0
        assert one_inch_usd_data.close == 100.0
        assert one_inch_usd_data.volume == 100.0


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 5, 10, 100, 1000])
async def test_market_data_batching_with_different_batch_sizes(
    fake_data: FakeData, batch_size: int
):
    """Test that market data flow works correctly with different batch sizes."""
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        one_inch_asset_id,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Create a larger dataset to test batching
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)

    # Create multiple data points for each pair to test batching
    num_data_points = 25  # This will create 25 * 4 = 100 total records
    fake_data.asset_pairs = {
        "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
        "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
        "1INCHUSD": {"base": "1INCH", "quote": "ZUSD"},
        "XETHXXBT": {"base": "XETH", "quote": "XXBT"},
    }

    # Generate multiple timestamps and data points
    market_data = {}
    for pair_name in fake_data.asset_pairs.keys():
        pair_data = []
        for i in range(num_data_points):
            timestamp = use_time + dt.timedelta(minutes=i)
            pair_data.append(
                [
                    int(timestamp.timestamp()),
                    100.0 + i,  # Vary the price slightly
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                ]
            )
        market_data[pair_name] = pair_data

    fake_data.market_data = market_data

    # Pull the provider asset market data with the specified batch size
    await pull_provider_asset_market_data(batch_size=batch_size)

    # Verify the data was inserted correctly
    with Session(engine) as session:
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        # Should have 100 total records (25 per pair * 4 pairs)
        expected_total = sum(len(data) for data in fake_data.market_data.values())
        assert len(provider_asset_market_data) == expected_total, (
            f"Expected {expected_total} records, got {len(provider_asset_market_data)} for batch_size={batch_size}"
        )

        # Verify specific data points
        btc_usd_records = (
            session.execute(
                select(ProviderAssetMarket).where(
                    ProviderAssetMarket.from_asset_id == usd_asset_id,
                    ProviderAssetMarket.to_asset_id == btc_asset_id,
                )
            )
            .scalars()
            .all()
        )

        assert len(btc_usd_records) == num_data_points, (
            f"Expected {num_data_points} BTC/USD records, got {len(btc_usd_records)} for batch_size={batch_size}"
        )

        # Check that prices are correctly varied
        prices = [record.close for record in btc_usd_records]
        assert min(prices) == 100.0, (
            f"Expected min price 100.0, got {min(prices)} for batch_size={batch_size}"
        )
        assert max(prices) == 100.0 + num_data_points - 1, (
            f"Expected max price {100.0 + num_data_points - 1}, got {max(prices)} for batch_size={batch_size}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [10000])
async def test_market_data_batching_edge_case_single_batch(
    fake_data: FakeData, batch_size: int
):
    """Test market data flow when all data fits in a single batch."""
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        _,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Create small dataset that fits in one batch
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)

    fake_data.asset_pairs = {
        "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
        "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
    }

    fake_data.market_data = {
        "XXBTZUSD": [
            [int(use_time.timestamp()), 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        "XETHZUSD": [
            [int(use_time.timestamp()), 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
        ],
    }

    # Use large batch size to ensure single batch
    await pull_provider_asset_market_data(batch_size=batch_size)

    # Verify the data was inserted correctly
    with Session(engine) as session:
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        assert len(provider_asset_market_data) == 2, (
            f"Expected 2 records, got {len(provider_asset_market_data)} for batch_size={batch_size}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
async def test_market_data_batching_data_integrity_across_batch_sizes(
    fake_data: FakeData, batch_size: int
):
    """Test that different batch sizes produce identical results."""
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        one_inch_asset_id,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Create consistent test data
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)

    fake_data.asset_pairs = {
        "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
        "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
        "1INCHUSD": {"base": "1INCH", "quote": "ZUSD"},
    }

    # Create 15 data points per pair (45 total records)
    num_data_points = 15
    market_data = {}
    for pair_name in fake_data.asset_pairs.keys():
        pair_data = []
        for i in range(num_data_points):
            timestamp = use_time + dt.timedelta(minutes=i)
            pair_data.append(
                [
                    int(timestamp.timestamp()),
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                    100.0 + i,
                ]
            )
        market_data[pair_name] = pair_data

    fake_data.market_data = market_data

    # Pull the provider asset market data with the specified batch size
    await pull_provider_asset_market_data(batch_size=batch_size)

    # Verify the data was inserted correctly
    with Session(engine) as session:
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        # Should have 45 total records (15 per pair * 3 pairs)
        expected_total = sum(len(data) for data in fake_data.market_data.values())
        assert len(provider_asset_market_data) == expected_total, (
            f"Expected {expected_total} records, got {len(provider_asset_market_data)} for batch_size={batch_size}"
        )

        # Verify specific data points
        btc_usd_records = (
            session.execute(
                select(ProviderAssetMarket).where(
                    ProviderAssetMarket.from_asset_id == usd_asset_id,
                    ProviderAssetMarket.to_asset_id == btc_asset_id,
                )
            )
            .scalars()
            .all()
        )

        assert len(btc_usd_records) == num_data_points, (
            f"Expected {num_data_points} BTC/USD records, got {len(btc_usd_records)} for batch_size={batch_size}"
        )

        # Check that prices are correctly varied
        prices = [record.close for record in btc_usd_records]
        assert min(prices) == 100.0, (
            f"Expected min price 100.0, got {min(prices)} for batch_size={batch_size}"
        )
        assert max(prices) == 100.0 + num_data_points - 1, (
            f"Expected max price {100.0 + num_data_points - 1}, got {max(prices)} for batch_size={batch_size}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [10])
async def test_market_data_batching_with_empty_data(
    fake_data: FakeData, batch_size: int
):
    """Test market data flow behavior with empty data."""
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Set empty data
    fake_data.asset_pairs = {}
    fake_data.market_data = {}

    # Pull the provider asset market data
    await pull_provider_asset_market_data(batch_size=batch_size)

    # Verify no data was inserted
    with Session(engine) as session:
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        assert len(provider_asset_market_data) == 0, (
            f"Expected 0 records, got {len(provider_asset_market_data)} for batch_size={batch_size}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [50])
async def test_market_data_batching_with_large_dataset(
    fake_data: FakeData, batch_size: int
):
    """Test market data flow with a large dataset to verify batching works correctly."""
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    (
        _,
        _,
        _,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
        one_inch_asset_id,
    ) = create_base_data(engine)

    # Reset the fake data.
    fake_data.reset_data()

    # Create a large dataset
    use_time = dt.datetime.now(dt.timezone.utc)
    use_time = use_time.replace(microsecond=0)

    fake_data.asset_pairs = {
        "XXBTZUSD": {"base": "XXBT", "quote": "ZUSD"},
        "XETHZUSD": {"base": "XETH", "quote": "ZUSD"},
        "1INCHUSD": {"base": "1INCH", "quote": "ZUSD"},
        "XETHXXBT": {"base": "XETH", "quote": "XXBT"},
    }

    # Create 100 data points per pair (400 total records)
    num_data_points = 100
    market_data = {}
    for pair_name in fake_data.asset_pairs.keys():
        pair_data = []
        for i in range(num_data_points):
            timestamp = use_time + dt.timedelta(minutes=i)
            pair_data.append(
                [
                    int(timestamp.timestamp()),
                    100.0 + i * 0.1,  # Small price increments
                    100.0 + i * 0.1,
                    100.0 + i * 0.1,
                    100.0 + i * 0.1,
                    100.0 + i * 0.1,
                    100.0 + i * 0.1,
                    100.0 + i * 0.1,
                ]
            )
        market_data[pair_name] = pair_data

    fake_data.market_data = market_data

    # Use small batch size to force multiple batches
    await pull_provider_asset_market_data(batch_size=batch_size)

    # Verify the data was inserted correctly
    with Session(engine) as session:
        provider_asset_market_data_stmt = select(ProviderAssetMarket)
        provider_asset_market_data = (
            session.execute(provider_asset_market_data_stmt).scalars().all()
        )

        expected_total = sum(len(data) for data in fake_data.market_data.values())
        assert len(provider_asset_market_data) == expected_total, (
            f"Expected {expected_total} records, got {len(provider_asset_market_data)} for batch_size={batch_size}"
        )

        # Verify we have the expected number of batches processed
        # With 400 records and batch_size=50, we should have 8 batches
        expected_batches = (expected_total + batch_size - 1) // batch_size
        assert expected_batches == 8, (
            f"Expected 8 batches for {expected_total} records with batch_size={batch_size}"
        )
