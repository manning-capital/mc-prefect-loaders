import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from prefect import flow, task, serve, get_run_logger
from sqlalchemy import select
from sqlalchemy.orm import Session
from prefect.artifacts import create_table_artifact
from mc_postgres_db.models import Provider, ProviderAssetOrder
from prefect.concurrency.asyncio import rate_limit
from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from src.shared.utils import get_base_url

INTERVAL_SECONDS = 30


@task(log_prints=True)
async def get_kraken_order_book(pair: str, count: int = 500) -> dict[str, object]:
    logger = get_run_logger()

    # Rate limit the task to avoid hitting Kraken's API too frequently.
    logger.info("Applying rate limit to Kraken API requests.")
    await rate_limit("kraken-api", strict=True, timeout_seconds=60)

    # Fetch the trade book from Kraken's public API.
    logger.info(f"Fetching trade book for {pair} with count {count} from Kraken.")
    response = requests.get(
        f"https://api.kraken.com/0/public/Depth?pair={pair}&count={count}"
    )
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Kraken: {response.status_code}")
    data: dict = response.json()

    # Extract the trade book data.
    trade_book = data.get("result")
    if not trade_book:
        raise Exception("No trade book data found in the response.")
    return trade_book


@task()
async def get_kraken_provider_asset_order_data(
    kraken_provider_id: int,
    from_asset_ids: list[int],
    to_asset_ids: list[int],
    count: int = 500,
    lookback_seconds: int = 60,
) -> pd.DataFrame:
    """
    Fetch the latest asset order data from Kraken's public API.
    """

    # Get all dependent tasks data.
    logger = get_run_logger()
    engine = await get_engine()
    base_url = await get_base_url()

    # Fetch the asset pairs from the database.
    with Session(engine) as session:
        stmt = select(Provider).where(
            Provider.id == kraken_provider_id,
        )
        provider = session.execute(stmt).scalar_one_or_none()
        if provider is None:
            raise Exception(f"Provider with id {kraken_provider_id} not found.")
        assets = provider.get_all_assets(
            engine=engine, asset_ids=from_asset_ids + to_asset_ids
        )

    # Check if assets are found.
    if assets is None or len(assets) == 0:
        logger.warning("No provider asset data found for Kraken.")
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "timestamp",
                    "provider_id",
                    "asset_id",
                    "order_type",
                    "price",
                    "volume",
                ]
            )
        )

    # Create a DataFrame for the provider asset data.
    provider_asset_code_df = pd.DataFrame(
        {
            "provider_id": kraken_provider_id,
            "asset_id": asset.asset_id,
            "asset_code": asset.asset_code,
        }
        for asset in assets
    )
    provider_asset_code_df.drop_duplicates(inplace=True)

    # Check for duplicates in the provider asset data.
    duplicates = provider_asset_code_df.groupby("asset_id").filter(lambda x: len(x) > 1)
    if not duplicates.empty:
        artifact_id = await create_table_artifact(
            key="provider-asset-code-duplicates",
            table=duplicates.to_dict(orient="records"),  # type: ignore
            description="Duplicate asset codes in provider asset data",
        )  # type: ignore
        logger.warning(
            f"Found duplicates in provider asset data. Artifact created: {base_url}/artifacts/artifact/{artifact_id}"
        )
        provider_asset_code_df = provider_asset_code_df.drop_duplicates(
            subset=["asset_id"]
        )
    provider_asset_id_to_code_map = {
        row.asset_id: row.asset_code  # type: ignore
        for row in provider_asset_code_df.itertuples()
    }

    # Create a DataFrame for the asset pairs.
    pairs_df = pd.DataFrame(
        {
            "from_asset_id": pd.Series(from_asset_ids),
            "to_asset_id": pd.Series(to_asset_ids),
        }
    )
    pairs_df["pair"] = (
        pairs_df["from_asset_id"].map(provider_asset_id_to_code_map).str.upper()  # type: ignore
        + pairs_df["to_asset_id"].map(provider_asset_id_to_code_map).str.upper()  # type: ignore
    )

    # Create a list of pairs to fetch.
    pairs = pairs_df["pair"].to_list()
    logger.info(f"Fetching order book data for pairs: {pairs}")

    # Fetch the order book data for each pair.
    trade_books = {}
    for pair in pairs:
        trade_books.update(await get_kraken_order_book(pair, count=count))

    # Convert the trade book data to a DataFrame.
    asks_trade_book_df = pd.DataFrame()
    for pair, book in trade_books.items():
        pair_df = pd.DataFrame(
            book["asks"], columns=pd.Index(["price", "volume", "timestamp"])
        )
        pair_df["pair"] = pair
        pair_df["timestamp"] = pd.to_datetime(pair_df["timestamp"], unit="s")
        pair_df["timestamp"] = pair_df["timestamp"].dt.tz_localize(timezone.utc)
        pair_df["price"] = pd.to_numeric(pair_df["price"], errors="coerce")
        pair_df["volume"] = pd.to_numeric(pair_df["volume"], errors="coerce")
        asks_trade_book_df = pd.concat([asks_trade_book_df, pair_df], ignore_index=True)

    # Merge the trade book with the pair id data.
    asks_trade_book_df = asks_trade_book_df.merge(
        pairs_df[["pair", "from_asset_id", "to_asset_id"]],
        on="pair",
        how="inner",
    )
    asks_trade_book_df["provider_id"] = kraken_provider_id

    # Convert bids trade book to a DataFrame.
    bids_trade_book_df = pd.DataFrame()
    for pair, book in trade_books.items():
        pair_df = pd.DataFrame(
            book["bids"], columns=pd.Index(["price", "volume", "timestamp"])
        )
        pair_df["pair"] = pair
        pair_df["timestamp"] = pd.to_datetime(pair_df["timestamp"], unit="s")
        pair_df["timestamp"] = pair_df["timestamp"].dt.tz_localize(timezone.utc)
        pair_df["price"] = pd.to_numeric(pair_df["price"], errors="coerce")
        pair_df["volume"] = pd.to_numeric(pair_df["volume"], errors="coerce")
        bids_trade_book_df = pd.concat([bids_trade_book_df, pair_df], ignore_index=True)

    # Merge the bids trade book with the pair id data.
    bids_trade_book_df = bids_trade_book_df.merge(
        pairs_df[["pair", "from_asset_id", "to_asset_id"]],
        on="pair",
        how="inner",
    )
    bids_trade_book_df["provider_id"] = kraken_provider_id
    temp = bids_trade_book_df["from_asset_id"].copy()
    bids_trade_book_df["from_asset_id"] = bids_trade_book_df["to_asset_id"]
    bids_trade_book_df["to_asset_id"] = temp

    # Combine asks and bids trade book data.
    trade_book_df = pd.concat(
        [asks_trade_book_df, bids_trade_book_df], ignore_index=True
    )

    # Drop order data that is older 1 hour.
    if not trade_book_df.empty:
        trade_book_df = trade_book_df[
            trade_book_df["timestamp"]
            >= (datetime.now(timezone.utc) - pd.Timedelta(seconds=lookback_seconds))
        ]
        logger.info(
            f"Filtered new provider asset orders to {len(trade_book_df)} records from the last day."
        )

    # Ensure the result is always a DataFrame, not a Series
    columns = [
        "timestamp",
        "provider_id",
        "from_asset_id",
        "to_asset_id",
        "price",
        "volume",
    ]
    return trade_book_df.loc[:, columns].copy()


@task()
async def get_provider_asset_order_data(
    provider_id: int,
    from_asset_ids: list[int],
    to_asset_ids: list[int],
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch the current provider asset order data from PostgreSQL.
    """
    from_asset_ids = [
        int(asset_id) for asset_id in from_asset_ids if asset_id is not None
    ]
    to_asset_ids = [int(asset_id) for asset_id in to_asset_ids if asset_id is not None]
    engine = await get_engine()
    logger = get_run_logger()
    stmt = select(ProviderAssetOrder).where(
        ProviderAssetOrder.provider_id == provider_id,
        ProviderAssetOrder.from_asset_id.in_(from_asset_ids),
        ProviderAssetOrder.to_asset_id.in_(to_asset_ids),
    )
    if start_datetime is not None and not pd.isna(start_datetime):
        stmt = stmt.where(ProviderAssetOrder.timestamp >= start_datetime)
    if end_datetime is not None and not pd.isna(end_datetime):
        stmt = stmt.where(ProviderAssetOrder.timestamp <= end_datetime)
    df = pd.read_sql(stmt, engine)

    # Convert the timestamp to UTC.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)

    if df.empty:
        logger.warning("No provider asset data found for the given IDs.")
        return pd.DataFrame()

    return df.drop(columns=["id"])


@task()
async def save_order_data(
    new_provider_asset_order_data: pd.DataFrame,
    current_provider_asset_data: pd.DataFrame,
):
    """
    Save the provider asset order data to PostgreSQL.
    """
    logger = get_run_logger()

    # Check if the DataFrame is empty.
    if new_provider_asset_order_data.empty:
        logger.info("No new order data to save. Exiting.")
        return

    # Compute the difference between the new and current data.
    if not current_provider_asset_data.empty:
        # Merge the new data with the current data to find differences.
        logger.info("Merging new data with current data to find differences.")
        merged_data = new_provider_asset_order_data.merge(
            current_provider_asset_data,
            on=new_provider_asset_order_data.columns.to_list(),
            how="outer",
            indicator=True,
        )
        # Filter for rows that are only in the new data.
        provider_asset_order_data = merged_data[
            merged_data["_merge"] == "left_only"
        ].drop(columns=["_merge"])
    else:
        # If there is no current data, use the new data as is.
        logger.info("No current data found. Using new data as is.")
        provider_asset_order_data = new_provider_asset_order_data

    # Save the data to PostgreSQL.
    await set_data(
        "provider_asset_order",
        provider_asset_order_data,
        operation_type="append",
    )  # type: ignore


@task()
async def get_kraken_provider_id() -> int:
    """
    Get the Kraken provider ID.
    """
    engine = await get_engine()
    with Session(engine) as session:
        stmt = select(Provider.id).where(Provider.name == "Kraken")
        kraken_provider_id = session.execute(stmt).scalar_one()
    return kraken_provider_id


@flow(log_prints=True)
async def pull_kraken_orders(
    from_asset_ids: list[int],
    to_asset_ids: list[int],
):
    logger = get_run_logger()

    # Use a fixed provider ID for Kraken.
    kraken_provider_id = await get_kraken_provider_id()

    # Validate the input pairs.
    if not from_asset_ids or not to_asset_ids:
        raise ValueError("from_assets and to_assets must be non-empty lists.")
    if len(from_asset_ids) != len(to_asset_ids):
        raise ValueError("from_assets and to_assets must have the same length.")

    # Get new kraken provider asset orders.
    logger.info("Fetching new provider asset data...")
    new_provider_asset_order_data: pd.DataFrame = (
        await get_kraken_provider_asset_order_data(
            kraken_provider_id,
            from_asset_ids,
            to_asset_ids,
            count=500,
            lookback_seconds=INTERVAL_SECONDS * 10,  # Look back 10 intervals,
        )
    )
    logger.info(
        f"Fetched {len(new_provider_asset_order_data)} new provider asset orders."
    )

    # Get current provider asset data.
    logger.info("Fetching current provider asset data...")
    current_provider_asset_data: pd.DataFrame = await get_provider_asset_order_data(
        kraken_provider_id,
        from_asset_ids,
        to_asset_ids,
        start_datetime=new_provider_asset_order_data["timestamp"].min(),
    )  # type: ignore
    logger.info(
        f"Fetched {len(current_provider_asset_data)} current provider asset orders."
    )

    # Save the provider asset data to PostgreSQL.
    logger.info("Saving provider asset data to PostgreSQL...")
    await save_order_data(new_provider_asset_order_data, current_provider_asset_data)


@task()
async def delete_old_orders(cutoff_date: datetime):
    """
    Delete old orders from the database that are older than the cutoff date.
    """
    engine = await get_engine()
    logger = get_run_logger()

    with Session(engine) as session:
        stmt = select(ProviderAssetOrder).where(
            ProviderAssetOrder.timestamp < cutoff_date
        )
        old_orders = session.execute(stmt).scalars().all()

        if not old_orders:
            logger.info("No old orders to delete.")
            return

        for order in old_orders:
            session.delete(order)

        session.commit()
        logger.info(f"Deleted {len(old_orders)} old orders older than {cutoff_date}.")


@flow()
async def clear_orders(
    keep_days: int = 30,
):
    logger = get_run_logger()

    # Get the cutoff date for old orders.
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=keep_days)).date()

    # Delete old orders from the database.
    await delete_old_orders(cutoff_date)  # type: ignore

    logger.info(f"Cleared orders older than {keep_days} days.")


if __name__ == "__main__":
    pull_kraken_orders_deployment = pull_kraken_orders.to_deployment(
        name="pull_kraken_orders_debug",
        concurrency_limit=1,
        parameters={
            "from_asset_ids": [1, 3, 3],
            "to_asset_ids": [2, 2, 1],
        },
    )
    clear_orders_deployment = clear_orders.to_deployment(
        name="clear_orders_debug",
        concurrency_limit=1,
        parameters={
            "keep_days": 30,
        },
    )
    serve(pull_kraken_orders_deployment, clear_orders_deployment)  # type: ignore
