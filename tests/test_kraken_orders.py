import os
import sys

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from src.order.kraken_trade_book_flows import (
    pull_kraken_orders,
)
from tests.mock_database import MockDatabase


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with disable_run_logger():
        with prefect_test_harness():
            yield


@pytest.fixture(autouse=True, scope="session")
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


@pytest.fixture(autouse=True, scope="session")
def mock_database():
    """
    Mock fixture for PostgreSQL database connection.
    This is a placeholder and should be replaced with actual database mocking logic.
    """

    # Mock implementations of the functions that interact with the database
    db = MockDatabase()

    async def mock_get_provider_asset_order_data(
        proider_id: int,
        from_asset_ids: list[int],
        to_asset_ids: list[int],
        start_datetime: datetime = None,
        end_datetime: datetime = None,
    ) -> pd.DataFrame:
        return db.get_provider_asset_order_data(
            provider_ids=[proider_id],
            from_asset_ids=from_asset_ids,
            to_asset_ids=to_asset_ids,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    async def mock_save_provider_asset_order_data(to_set: pd.DataFrame) -> None:
        for _, row in to_set.iterrows():
            db.add_provider_asset_order_data(
                timestamp=row["timestamp"],
                provider_id=row["provider_id"],
                from_asset_id=row["from_asset_id"],
                to_asset_id=row["to_asset_id"],
                price=row["price"],
                volume=row["volume"],
            )

    # Patch the functions with mock implementations
    with (
        patch(
            "src.order.kraken_trade_book_flows.get_provider_asset_order_data",
            mock_get_provider_asset_order_data,
        ),
        patch(
            "src.order.kraken_trade_book_flows.save_provider_asset_order_data",
            mock_save_provider_asset_order_data,
        ),
    ):
        yield db


@pytest.mark.asyncio
async def test_pull_when_both_database_and_kraken_is_empty(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_database.get_provider_asset_order_count() == 0
    assert mock_data_source.get_provider_asset_order_count() == 0


@pytest.mark.asyncio
async def test_pull_when_database_is_empty(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
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
    assert mock_database.get_provider_asset_order_count() == 1
    assert mock_data_source.get_provider_asset_order_count() == 1


@pytest.mark.asyncio
async def test_pull_when_database_has_an_existing_record(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )

    # Add the same data to the mock database.
    mock_database.add_provider_asset_order_data(
        timestamp=use_time,
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
    assert mock_database.get_provider_asset_order_count() == 1
    assert mock_data_source.get_provider_asset_order_count() == 1


@pytest.mark.asyncio
async def test_pull_when_database_has_multiple_records(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=101.0,
        volume=11.0,
    )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_database.get_provider_asset_order_count() == 2
    assert mock_data_source.get_provider_asset_order_count() == 2


@pytest.mark.asyncio
async def test_pull_with_multiple_from_and_to_asset_ids(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=1,
        from_asset_id=3,
        to_asset_id=4,
        price=101.0,
        volume=11.0,
    )

    # Call the pull_kraken_orders function with multiple from and to asset IDs
    await pull_kraken_orders(
        from_asset_ids=[1, 3],
        to_asset_ids=[2, 4],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_database.get_provider_asset_order_count() == 2
    assert mock_data_source.get_provider_asset_order_count() == 2


@pytest.mark.asyncio
async def test_pull_with_multiple_existing_duplicates_with_one_new_duplicate_and_non_duplicates(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=1),
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=101.0,
        volume=11.0,
    )

    # Add the same data to the mock database.
    mock_database.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )

    # Ensure the state of the mock database and data source before pulling.
    assert mock_database.get_provider_asset_order_count() == 1
    assert mock_data_source.get_provider_asset_order_count() == 3

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_database.get_provider_asset_order_count() == 2
    assert mock_data_source.get_provider_asset_order_count() == 3


@pytest.mark.asyncio
async def test_pull_with_multiple_new_duplicates_and_non_duplicates(
    mock_database: MockDatabase, mock_data_source: MockDatabase
):
    # Clear any existing data in the mock database
    mock_database.clear_provider_asset_order_data()
    mock_data_source.clear_provider_asset_order_data()

    # Add some mock data to the mock data source.
    use_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time,
        provider_id=1,
        from_asset_id=1,
        to_asset_id=2,
        price=100.0,
        volume=10.0,
    )
    mock_data_source.add_provider_asset_order_data(
        timestamp=use_time - timedelta(seconds=2),
        provider_id=1,
        from_asset_id=3,
        to_asset_id=4,
        price=102.0,
        volume=12.0,
    )

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[1, 3],
        to_asset_ids=[2, 4],
    )

    # Verify that no data was pulled from the mock database.
    assert mock_database.get_provider_asset_order_count() == 3
    assert mock_data_source.get_provider_asset_order_count() == 3
