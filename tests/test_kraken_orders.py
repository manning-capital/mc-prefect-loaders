import os
import sys

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch
from kraken_trade_book_flows import (
    pull_kraken_orders,
)

from prefect.testing.utilities import prefect_test_harness
from prefect.logging import disable_run_logger


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with disable_run_logger():
        with prefect_test_harness():
            yield


@pytest.fixture(autouse=True)
def mock_postgres():
    """
    Mock fixture for PostgreSQL database connection.
    This is a placeholder and should be replaced with actual database mocking logic.
    """

    # Mock implementations of the functions that interact with the database

    async def mock_get_kraken_provider_asset_order_data(
        kraken_provider_id: int,
        from_asset_ids: list[int],
        to_asset_ids: list[int],
        count: int = 500,
        lookback_seconds: int = 60,
    ):
        # Mock implementation for testing
        return pd.DataFrame(
            columns=[
                "timestamp",
                "provider_id",
                "from_asset_id",
                "to_asset_id",
                "price",
                "volume",
            ]
        )

    async def mock_get_provider_asset_order_data(
        proider_id: int,
        from_asset_ids: list[int],
        to_asset_ids: list[int],
        start_datetime: datetime = None,
        end_datetime: datetime = None,
    ) -> pd.DataFrame:
        # Mock implementation for testing
        return pd.DataFrame(
            columns=[
                "timestamp",
                "provider_id",
                "from_asset_id",
                "to_asset_id",
                "price",
                "volume",
            ]
        )

    async def mock_save_provider_asset_order_data(toset: pd.DataFrame) -> None:
        # Mock implementation for testing
        return None

    # Patch the functions with mock implementations
    with (
        patch(
            "kraken_trade_book_flows.get_kraken_provider_asset_order_data",
            mock_get_kraken_provider_asset_order_data,
        ),
        patch(
            "kraken_trade_book_flows.get_provider_asset_order_data",
            mock_get_provider_asset_order_data,
        ),
        patch(
            "kraken_trade_book_flows.save_provider_asset_order_data",
            mock_save_provider_asset_order_data,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_pull_when_both_database_and_kraken_is_empty():
    await pull_kraken_orders(
        from_asset_ids=[1],
        to_asset_ids=[2],
    )
