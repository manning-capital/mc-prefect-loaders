import os
import sys
import json
import random

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
import pandas as pd
from prefect import task
from typing import Any
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.testing.utilities import clear_database
from mc_postgres_db.models import (
    ProviderAssetOrder,
    ProviderType,
    Provider,
    Asset,
    AssetType,
    ProviderAsset,
)
from mc_postgres_db.prefect.asyncio.tasks import get_engine as get_engine_async
from src.order.kraken_trade_book_flows import pull_kraken_orders, INTERVAL_SECONDS

@pytest.fixture(scope="function")
def kraken_orders_data():
    # Initialize the data dictionary.
    data: dict[str, dict[str, list[tuple[float, float, int]]]] = {}

     # Create a mock function that gets the kraken provider asset order data.
    @task()
    async def mock_get_kraken_order_book(
        pair: str,
        count: int = 500,
    ) -> dict[str, dict[str, list[tuple[float, float, int]]]]:
        
        if pair not in data:
            return {
                pair : {
                    "asks" : [],
                    "bids" : [],
                }
            }

        return {
            pair : data[pair]
        }

    with patch(
        "src.order.kraken_trade_book_flows.get_kraken_order_book",
        mock_get_kraken_order_book,
    ):
        yield data
    


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
        session.commit()

        # Get the ids for the assets.
        btc_asset_id_stmt = select(Asset.id).where(Asset.name == "BTC")
        btc_asset_id = session.execute(btc_asset_id_stmt).scalar_one()
        eth_asset_id_stmt = select(Asset.id).where(Asset.name == "ETH")
        eth_asset_id = session.execute(eth_asset_id_stmt).scalar_one()
        usd_asset_id_stmt = select(Asset.id).where(Asset.name == "USD")
        usd_asset_id = session.execute(usd_asset_id_stmt).scalar_one()

        # Add the provider asset data.
        session.add(
            ProviderAsset(
                date=(datetime.now(timezone.utc) - timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=btc_asset_id,
                asset_code="XXBT",
                is_active=True,
            )
        )
        session.add(
            ProviderAsset(
                date=(datetime.now(timezone.utc) - timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=eth_asset_id,
                asset_code="XETH",
                is_active=True,
            )
        )
        session.add(
            ProviderAsset(
                date=(datetime.now(timezone.utc) - timedelta(days=1)).date(),
                provider_id=kraken_provider_id,
                asset_id=usd_asset_id,
                asset_code="ZUSD",
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
        )


@pytest.mark.asyncio
async def test_pull_when_both_database_and_kraken_is_empty(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the database.
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

   
    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id],
        to_asset_ids=[usd_asset_id],
    )


@pytest.mark.asyncio
async def test_pull_when_database_is_empty(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        kraken_provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC provider asset.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == kraken_provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == kraken_provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()

        # Add a few orders to the kraken orders data.
        use_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
            "asks": [(10000.0, 0.001, int(use_time.timestamp()))],
            "bids": [(9000.0, 0.001, int(use_time.timestamp()))],
        }

        # Call the pull_kraken_orders function with empty data
        await pull_kraken_orders(
            from_asset_ids=[btc_asset_id],
            to_asset_ids=[usd_asset_id],
        )

        # See if the data was saved to the database.
        btc_usd_stmt = select(ProviderAssetOrder).where(
            ProviderAssetOrder.provider_id == kraken_provider_id,
            ProviderAssetOrder.from_asset_id == btc_asset_id,
            ProviderAssetOrder.to_asset_id == usd_asset_id,
        )
        btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
        assert len(btc_usd_df) == 1
        assert btc_usd_df.iloc[0]["provider_id"] == kraken_provider_id
        assert btc_usd_df.iloc[0]["from_asset_id"] == btc_asset_id
        assert btc_usd_df.iloc[0]["to_asset_id"] == usd_asset_id
        assert int(btc_usd_df.iloc[0]["timestamp"].timestamp()) == int(use_time.timestamp())
        assert btc_usd_df.iloc[0]["price"] == 10000.0
        assert btc_usd_df.iloc[0]["volume"] == 0.001
        usd_btc_stmt = select(ProviderAssetOrder).where(
            ProviderAssetOrder.provider_id == kraken_provider_id,
            ProviderAssetOrder.from_asset_id == usd_asset_id,
            ProviderAssetOrder.to_asset_id == btc_asset_id,
        )
        usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
        assert len(usd_btc_df) == 1
        assert usd_btc_df.iloc[0]["provider_id"] == kraken_provider_id
        assert usd_btc_df.iloc[0]["from_asset_id"] == usd_asset_id
        assert usd_btc_df.iloc[0]["to_asset_id"] == btc_asset_id
        assert int(usd_btc_df.iloc[0]["timestamp"].timestamp()) == int(use_time.timestamp())
        assert usd_btc_df.iloc[0]["price"] == 9000.0
        assert usd_btc_df.iloc[0]["volume"] == 0.001



@pytest.mark.asyncio
async def test_pull_when_database_has_an_existing_record(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the database.
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC and USD provider assets.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()

    # Add some mock data to the database. Floor the timestamp to the nearest second.
    use_time = datetime.fromtimestamp(int((datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()), tz=timezone.utc)
    ask_price = 10000.0
    ask_volume = 0.001
    bid_price = 9000.0
    bid_volume = 0.001
    kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [(ask_price, ask_volume, int(use_time.timestamp()))],
        "bids": [(bid_price, bid_volume, int(use_time.timestamp()))],
    }

    # Add the same data to the mock database.
    with Session(engine) as session:
        session.add(
            ProviderAssetOrder(
                provider_id=provider_id,
                from_asset_id=btc_asset_id,
                to_asset_id=usd_asset_id,
                timestamp=use_time,
                price=ask_price,
                volume=ask_volume,
            )
        )
        session.commit()

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id],
        to_asset_ids=[usd_asset_id],
    )

    # Verify that no data was pulled from the mock database.
    btc_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
    assert len(btc_usd_df) == 1
    assert btc_usd_df.iloc[0]["provider_id"] == provider_id
    assert btc_usd_df.iloc[0]["price"] == ask_price
    assert btc_usd_df.iloc[0]["volume"] == ask_volume
    usd_btc_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == btc_asset_id,
    )
    usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
    assert len(usd_btc_df) == 1
    assert usd_btc_df.iloc[0]["provider_id"] == provider_id
    assert usd_btc_df.iloc[0]["price"] == bid_price
    assert usd_btc_df.iloc[0]["volume"] == bid_volume


@pytest.mark.asyncio
async def test_pull_when_database_is_empty_and_source_has_multiple_records(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the database.
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC and USD provider assets.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()

    # Add some mock data to the database. Floor the timestamp to the nearest second.
    use_time = datetime.fromtimestamp(int((datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()), tz=timezone.utc)
    ask_price = 10000.0
    ask_volume = 0.001
    bid_price = 9000.0
    bid_volume = 0.001
    kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [(ask_price, ask_volume, int(use_time.timestamp()))],
        "bids": [(bid_price, bid_volume, int(use_time.timestamp()))],
    }

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id, usd_asset_id],
        to_asset_ids=[usd_asset_id, btc_asset_id],
    )

    # Verify that no data was pulled from the mock database.
    btc_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
    assert len(btc_usd_df) == 1
    assert btc_usd_df.iloc[0]["provider_id"] == provider_id
    assert btc_usd_df.iloc[0]["price"] == ask_price
    assert btc_usd_df.iloc[0]["volume"] == ask_volume
    usd_btc_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == btc_asset_id,
    )
    usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
    assert len(usd_btc_df) == 1
    assert usd_btc_df.iloc[0]["provider_id"] == provider_id
    assert usd_btc_df.iloc[0]["price"] == bid_price
    assert usd_btc_df.iloc[0]["volume"] == bid_volume


@pytest.mark.asyncio
async def test_pull_when_database_is_empty_and_source_has_multiple_from_and_to_asset_ids(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC and USD provider assets.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        eth_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == eth_asset_id,
        )
        eth_provider_asset = session.execute(eth_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()
    
    # Add some mock data to the source.
    use_time = datetime.fromtimestamp(int((datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()), tz=timezone.utc)
    ask_price = 10000.0
    ask_volume = 0.001
    bid_price = 9000.0
    bid_volume = 0.001
    kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [(ask_price, ask_volume, int(use_time.timestamp()))],
        "bids": [(bid_price, bid_volume, int(use_time.timestamp()))],
    }
    kraken_orders_data[eth_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [(ask_price, ask_volume, int(use_time.timestamp()))],
        "bids": [(bid_price, bid_volume, int(use_time.timestamp()))],
    }

    # Call the pull_kraken_orders function with multiple from and to asset IDs
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id, eth_asset_id],
        to_asset_ids=[usd_asset_id, usd_asset_id],
    )

    # Verify that no data was pulled from the mock database.
    btc_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
    assert len(btc_usd_df) == 1
    assert btc_usd_df.iloc[0]["provider_id"] == provider_id
    assert btc_usd_df.iloc[0]["from_asset_id"] == btc_asset_id
    assert btc_usd_df.iloc[0]["to_asset_id"] == usd_asset_id
    assert btc_usd_df.iloc[0]["price"] == ask_price
    assert btc_usd_df.iloc[0]["volume"] == ask_volume
    eth_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == eth_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    eth_usd_df = pd.read_sql(eth_usd_stmt, engine)
    assert len(eth_usd_df) == 1
    assert eth_usd_df.iloc[0]["provider_id"] == provider_id
    assert eth_usd_df.iloc[0]["from_asset_id"] == eth_asset_id
    assert eth_usd_df.iloc[0]["to_asset_id"] == usd_asset_id
    assert eth_usd_df.iloc[0]["price"] == ask_price
    assert eth_usd_df.iloc[0]["volume"] == ask_volume
    usd_btc_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == btc_asset_id,
    )
    usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
    assert len(usd_btc_df) == 1
    assert usd_btc_df.iloc[0]["provider_id"] == provider_id
    assert usd_btc_df.iloc[0]["from_asset_id"] == usd_asset_id
    assert usd_btc_df.iloc[0]["to_asset_id"] == btc_asset_id
    assert usd_btc_df.iloc[0]["price"] == bid_price
    assert usd_btc_df.iloc[0]["volume"] == bid_volume
    usd_eth_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == eth_asset_id,
    )
    usd_eth_df = pd.read_sql(usd_eth_stmt, engine)
    assert len(usd_eth_df) == 1
    assert usd_eth_df.iloc[0]["provider_id"] == provider_id
    assert usd_eth_df.iloc[0]["from_asset_id"] == usd_asset_id
    assert usd_eth_df.iloc[0]["to_asset_id"] == eth_asset_id
    assert usd_eth_df.iloc[0]["price"] == bid_price
    assert usd_eth_df.iloc[0]["volume"] == bid_volume



@pytest.mark.asyncio
async def test_pull_when_database_has_one_duplicate_and_multiple_non_duplicates(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC and USD provider assets.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()

    # Add some mock data to the source.
    use_time = datetime.fromtimestamp(int((datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()), tz=timezone.utc)
    kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [
            (10000.0, 0.001, int(use_time.timestamp())),
            (50000.0, 0.002, int(use_time.timestamp())),
            (100000.0, 0.003, int(use_time.timestamp())),
            ],
        "bids": [
            (9000.0, 0.001, int(use_time.timestamp())),
            (40000.0, 0.002, int(use_time.timestamp())),
            (80000.0, 0.003, int(use_time.timestamp())),
            ],
    }

    # Add the same data to the mock database.
    with Session(engine) as session:
        session.add(
            ProviderAssetOrder(
                provider_id=provider_id,
                from_asset_id=btc_asset_id,
                to_asset_id=usd_asset_id,
                timestamp=use_time,
                price=10000.0,
                volume=0.001,
            )
        )
        session.commit()

    # Ensure the state of the database has 1.
    assert len(pd.read_sql(select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    ), engine)) == 1

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id],
        to_asset_ids=[usd_asset_id],
    )

    # Verify that no data was pulled from the database.
    btc_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
    assert len(btc_usd_df) == 3
    usd_btc_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == btc_asset_id,
    )
    usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
    assert len(usd_btc_df) == 3


@pytest.mark.asyncio
async def test_pull_when_database_has_one_duplicate_and_source_has_from_and_to_asset_ids(kraken_orders_data: dict[str, dict[str, list[tuple[float, float, int]]]]):
    # Clear any existing data in the mock database
    engine = await get_engine_async()
    clear_database(engine)

    # Create the sample data.
    (
        provider_type_id,
        crypto_asset_type_id,
        fiat_asset_type_id,
        provider_id,
        btc_asset_id,
        eth_asset_id,
        usd_asset_id,
    ) = create_sample_data(engine)

    # Get the BTC and USD provider assets.
    with Session(engine) as session:
        btc_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == btc_asset_id,
        )
        btc_provider_asset = session.execute(btc_provider_asset_stmt).scalar_one()
        usd_provider_asset_stmt = select(ProviderAsset).where(
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == usd_asset_id,
        )
        usd_provider_asset = session.execute(usd_provider_asset_stmt).scalar_one()
        eth_provider_asset_stmt = select(ProviderAsset).where(  
            ProviderAsset.provider_id == provider_id,
            ProviderAsset.asset_id == eth_asset_id,
        )
        eth_provider_asset = session.execute(eth_provider_asset_stmt).scalar_one()

    # Add some mock data to the source.
    use_time = datetime.fromtimestamp(int((datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()), tz=timezone.utc)
    kraken_orders_data[btc_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [
            (10000.0, 0.001, int(use_time.timestamp())),
            (50000.0, 0.002, int(use_time.timestamp())),
            (100000.0, 0.003, int(use_time.timestamp())),
            ],
        "bids": [
            (9000.0, 0.001, int(use_time.timestamp())),
            (40000.0, 0.002, int(use_time.timestamp())),
            (80000.0, 0.003, int(use_time.timestamp())),
            ],
    }
    kraken_orders_data[eth_provider_asset.asset_code + usd_provider_asset.asset_code] = {
        "asks": [(10000.0, 0.001, int(use_time.timestamp()))],
        "bids": [(9000.0, 0.001, int(use_time.timestamp()))],
    }

    # Add one duplicate to the database.
    with Session(engine) as session:
        session.add(
            ProviderAssetOrder(
                provider_id=provider_id,
                from_asset_id=btc_asset_id,
                to_asset_id=usd_asset_id,
                timestamp=use_time,
                price=10000.0,
                volume=0.001,
            )
        )
        session.commit()

    # Ensure the state of the database has 1.
    assert len(pd.read_sql(select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    ), engine)) == 1

    # Call the pull_kraken_orders function with empty data
    await pull_kraken_orders(
        from_asset_ids=[btc_asset_id, eth_asset_id],
        to_asset_ids=[usd_asset_id, usd_asset_id],
    )

    # Verify that no data was pulled from the database.
    btc_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == btc_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    btc_usd_df = pd.read_sql(btc_usd_stmt, engine)
    assert len(btc_usd_df) == 3
    usd_btc_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == btc_asset_id,
    )
    usd_btc_df = pd.read_sql(usd_btc_stmt, engine)
    assert len(usd_btc_df) == 3
    eth_usd_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == eth_asset_id,
        ProviderAssetOrder.to_asset_id == usd_asset_id,
    )
    eth_usd_df = pd.read_sql(eth_usd_stmt, engine)
    assert len(eth_usd_df) == 1
    usd_eth_stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id == usd_asset_id,
        ProviderAssetOrder.to_asset_id == eth_asset_id,
    )
    usd_eth_df = pd.read_sql(usd_eth_stmt, engine)
    assert len(usd_eth_df) == 1

