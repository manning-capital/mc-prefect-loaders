import os
import sys

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from tests.mock_database import MockDatabase
from mc_postgres_db.testing.utilities import clear_database
from mc_postgres_db.models import (
    ProviderAssetOrder,
    ProviderType,
    Provider,
    Asset,
    AssetType,
)
from mc_postgres_db.prefect.asyncio.tasks import get_engine as get_engine_async
from src.order.kraken_trade_book_flows import pull_kraken_orders


@pytest.fixture(autouse=True, scope="function")
def mock_data_source():
    """
    Mock fixture for PostgreSQL database connection.
    This is a placeholder and should be replaced with actual database mocking logic.
    """
    # Mock implementations of the functions that interact with the database
    db = MockDatabase()

    async def mock_get_kraken_provider_asset_order_data(
        kraken_provider_id: int,
        from_asset_ids: list[int],
        to_asset_ids: list[int],
        count: int = 500,
        lookback_seconds: int = 60,
    ):
        return db.get_provider_asset_order_data(
            provider_ids=[kraken_provider_id],
            from_asset_ids=from_asset_ids,
            to_asset_ids=to_asset_ids,
            start_datetime=datetime.now(timezone.utc)
            - timedelta(seconds=lookback_seconds),
            end_datetime=datetime.now(timezone.utc),
        )

    with patch(
        "src.order.kraken_trade_book_flows.get_kraken_provider_asset_order_data",
        mock_get_kraken_provider_asset_order_data,
    ):
        yield db


def create_sample_data(engine: Engine):
    with Session(engine) as session:
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

        provider_id_stmt = select(Provider.id).where(Provider.name == "Kraken")
        provider_id = session.execute(provider_id_stmt).scalar_one()

        # Add asset type data.
        session.add(
            AssetType(
                name="CryptoCurrency",
                description="CryptoCurrency",
            )
        )
        session.commit()

        # Get the id for the asset type.
        asset_type_id_stmt = select(AssetType.id).where(
            AssetType.name == "CryptoCurrency"
        )
        asset_type_id = session.execute(asset_type_id_stmt).scalar_one()

        # Add the asset data.
        session.add(
            Asset(
                name="BTC",
                description="BTC",
                asset_type_id=asset_type_id,
            )
        )
        session.add(
            Asset(
                name="ETH",
                description="ETH",
                asset_type_id=asset_type_id,
            )
        )
        session.add(
            Asset(
                name="USDT",
                description="USDT",
                asset_type_id=asset_type_id,
            )
        )
        session.add(
            Asset(
                name="USDC",
                description="USDC",
                asset_type_id=asset_type_id,
            )
        )
        session.commit()

        # Get the ids for the assets.
        asset_id_1_stmt = select(Asset.id).where(Asset.name == "BTC")
        asset_id_1 = session.execute(asset_id_1_stmt).scalar_one()
        asset_id_2_stmt = select(Asset.id).where(Asset.name == "ETH")
        asset_id_2 = session.execute(asset_id_2_stmt).scalar_one()
        asset_id_3_stmt = select(Asset.id).where(Asset.name == "USDT")
        asset_id_3 = session.execute(asset_id_3_stmt).scalar_one()
        asset_id_4_stmt = select(Asset.id).where(Asset.name == "USDC")
        asset_id_4 = session.execute(asset_id_4_stmt).scalar_one()

        return (
            provider_type_id,
            asset_type_id,
            provider_id,
            asset_id_1,
            asset_id_2,
            asset_id_3,
            asset_id_4,
        )


@pytest.mark.asyncio
async def test_pull_when_both_database_and_kraken_is_empty(
    mock_data_source: MockDatabase,
):
    # Clear any existing data in the mock data source
    mock_data_source.clear_provider_asset_order_data()

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 0


@pytest.mark.asyncio
async def test_pull_when_database_is_empty(mock_data_source: MockDatabase):
    # Clear any existing data in the mock database
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    mock_data_source.add_provider_asset_order_data(
        timestamp=datetime.now(timezone.utc),
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 1


@pytest.mark.asyncio
async def test_pull_when_database_has_an_existing_record(
    mock_data_source: MockDatabase,
):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)
    mock_data_source.clear_provider_asset_order_data()

    # Create the sample data.
    (
        provider_type_id,
        asset_type_id,
        provider_id,
        asset_id_1,
        asset_id_2,
        asset_id_3,
        asset_id_4,
    ) = create_sample_data(engine)

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )

    # Add the same data to the mock database.
    with Session(engine) as session:
        session.add(
            ProviderAssetOrder(
                provider_id=provider_id,
                from_asset_id=asset_id_1,
                to_asset_id=asset_id_2,
                timestamp=use_time,
                price=100.0,
                volume=10.0,
            )
        )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 1


@pytest.mark.asyncio
async def test_pull_when_database_has_multiple_records(mock_data_source: MockDatabase):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)
    mock_data_source.clear_provider_asset_order_data()

    # Create the sample data.
    (
        provider_type_id,
        asset_type_id,
        provider_id,
        asset_id_1,
        asset_id_2,
        asset_id_3,
        asset_id_4,
    ) = create_sample_data(engine)

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=101.0,
        volume=11.0,
    )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[asset_id_1],
        to_asset_ids=[asset_id_2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 2


@pytest.mark.asyncio
async def test_pull_with_multiple_from_and_to_asset_ids(mock_data_source: MockDatabase):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)
    mock_data_source.clear_provider_asset_order_data()

    # Create the sample data.
    (
        provider_type_id,
        asset_type_id,
        provider_id,
        asset_id_1,
        asset_id_2,
        asset_id_3,
        asset_id_4,
    ) = create_sample_data(engine)

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=provider_id,
        from_asset_id=asset_id_3,
        to_asset_id=asset_id_4,
        price=101.0,
        volume=11.0,
    )

    # Call the pull_kraken_orders function with multiple from and to asset IDs
    await pull_kraken_orders(
        from_asset_ids=[asset_id_1, asset_id_3],
        to_asset_ids=[asset_id_2, asset_id_4],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 2


@pytest.mark.asyncio
async def test_pull_with_multiple_existing_duplicates_with_one_new_duplicate_and_non_duplicates(
    mock_data_source: MockDatabase,
):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)
    mock_data_source.clear_provider_asset_order_data()

    # Create the sample data.
    (
        provider_type_id,
        asset_type_id,
        provider_id,
        asset_id_1,
        asset_id_2,
        asset_id_3,
        asset_id_4,
    ) = create_sample_data(engine)

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=101.0,
        volume=11.0,
    )

    # Add the same data to the mock database.
    with Session(engine) as session:
        session.add(
            ProviderAssetOrder(
                provider_id=provider_id,
                from_asset_id=asset_id_1,
                to_asset_id=asset_id_2,
                timestamp=use_time,
                price=100.0,
                volume=10.0,
            )
        )
        session.commit()

    # Ensure the state of the mock database and data source before pulling.
    assert mock_data_source.get_provider_asset_order_count() == 3

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[asset_id_1],
        to_asset_ids=[asset_id_2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 3


@pytest.mark.asyncio
async def test_pull_with_multiple_new_duplicates_and_non_duplicates(
    mock_data_source: MockDatabase,
):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)
    mock_data_source.clear_provider_asset_order_data()

    # Create the sample data.
    (
        provider_type_id,
        asset_type_id,
        provider_id,
        asset_id_1,
        asset_id_2,
        asset_id_3,
        asset_id_4,
    ) = create_sample_data(engine)

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=provider_id,
        from_asset_id=asset_id_1,
        to_asset_id=asset_id_2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=2),
        provider_id=provider_id,
        from_asset_id=asset_id_3,
        to_asset_id=asset_id_4,
        price=102.0,
        volume=12.0,
    )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[asset_id_1, asset_id_3],
        to_asset_ids=[asset_id_2, asset_id_4],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_data_source.get_provider_asset_order_count() == 3
