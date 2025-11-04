import datetime as dt
from typing import Optional

from prefect_dask import get_dask_client
import pandas as pd
import dask.dataframe as dd
from dask import delayed
import numpy as np
import mc_postgres_db.models as models
from prefect import flow, get_run_logger, task
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import INTERVAL
from prefect_dask.task_runners import DaskTaskRunner
from coiled import Cluster
from dask.distributed import LocalCluster, Client
from prefect.blocks.system import Secret
import subprocess


from mc_postgres_db.prefect.asyncio.tasks import get_engine

DASK_CLUSTER_NAME = "prefect-cluster"
DASK_N_WORKERS = 30
DASK_REGION = "us-east-1"
DASK_CONTAINER = "ghcr.io/manning-capital/mc-prefect-loaders:main"
DASK_WORKER_MEMORY = "16GB"
DASK_WORKER_CPU = 2

MAX_PROVIDER_ASSET_GROUPS = 5000

@task()
async def coiled_login():
    """
    Login to coiled.
    """
    # Run a command and handle errors
    logger = get_run_logger()
    coiled_api_key: str = (await Secret.load("coiled-api-key")).value()
    try:
        result = subprocess.run(
            ['coiled', 'login', '--api-key', coiled_api_key],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(e.stderr)

@task()
async def create_dask_cluster(use_local_cluster: bool = True) -> Cluster | LocalCluster:
    """
    Create the dask cluster.
    """
    logger = get_run_logger()
    if use_local_cluster:
        logger.info("Creating local dask cluster...")
        return LocalCluster(name=DASK_CLUSTER_NAME, n_workers=4, threads_per_worker=2)
    else:
        # Login to coiled.
        await coiled_login()

        # Create the coiled dask cluster.
        logger.info("Creating coiled dask cluster...")
        return Cluster(
            name=DASK_CLUSTER_NAME,
            n_workers=DASK_N_WORKERS,
            region=DASK_REGION,
            container=DASK_CONTAINER,
            worker_memory=DASK_WORKER_MEMORY,
            worker_cpu=DASK_WORKER_CPU,
            spot_policy="spot_with_fallback",
        )


@task()
async def shutdown_dask_cluster(cluster: Cluster | LocalCluster):
    """
    Shutdown/close the dask cluster.
    """
    if isinstance(cluster, LocalCluster):
        cluster.close()
    else:
        cluster.shutdown()

@task()
async def get_active_provider_asset_group_ids() -> set[int]:
    """
    Get the active provider asset groups.
    """
    engine = await get_engine()
    with Session(engine) as session:
        return set(
            session.scalars(
                select(models.ProviderAssetGroup.id).where(
                    models.ProviderAssetGroup.is_active.is_(True)
                )
            )
        )


@task()
async def get_provider_asset_group_member_data(
    provider_asset_group_ids: list[int],
):
    """
    Get the provider asset group member data.
    """
    engine = await get_engine()
    return pd.read_sql(
        select(
            models.ProviderAssetGroupMember.provider_asset_group_id,
            models.ProviderAssetGroupMember.order,
            models.ProviderAssetGroupMember.provider_id,
            models.ProviderAssetGroupMember.from_asset_id,
            models.ProviderAssetGroupMember.to_asset_id,
        )
        .where(
            models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                provider_asset_group_ids
            )
        )
        .distinct(),
        engine,
    )


@task()
async def get_provider_asset_group_market_data(
    start: dt.datetime,
    end: dt.datetime,
    provider_ids: list[int],
    from_asset_ids: list[int],
    to_asset_ids: list[int],
):
    """
    Get the provider asset group market data.
    """
    engine = await get_engine()
    return pd.read_sql(
        select(
            models.ProviderAssetMarket.timestamp,
            models.ProviderAssetMarket.provider_id,
            models.ProviderAssetMarket.from_asset_id,
            models.ProviderAssetMarket.to_asset_id,
            models.ProviderAssetMarket.close,
        )
        .where(
            models.ProviderAssetMarket.timestamp.between(start, end),
            models.ProviderAssetMarket.provider_id.in_(provider_ids),
            models.ProviderAssetMarket.from_asset_id.in_(from_asset_ids),
            models.ProviderAssetMarket.to_asset_ids.in_(to_asset_ids),
        )
        .order_by(models.ProviderAssetMarket.timestamp),
        engine,
    )


@delayed
def load_pairs_trading_frame_chunk(
    start: dt.datetime,
    end: dt.datetime,
    provider_asset_group_member_data: pd.DataFrame,
    provider_asset_market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the pairs trading frame for a chunk of provider asset groups.
    Returns only the essential columns needed for cointegration analysis.

    Args:
        start: Start datetime (timezone-naive)
        end: End datetime (timezone-naive)
        provider_asset_group_member_data: DataFrame of provider asset group member data
        provider_asset_market_data: DataFrame of provider asset market data

    Returns:
        pandas DataFrame indexed by provider_asset_group_id with columns:
            - timestamp
            - close_1
            - close_2
    """
    # Generate the time frame using pd.date_range.
    time_frame = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="1min")})

    # Cross join the provider asset group member data with the time frame.
    full_frame = time_frame.merge(provider_asset_group_member_data, how="cross")
    full_frame = full_frame.sort_values("timestamp")

    # Merge the provider asset market data with the full frame using merge_asof.
    full_market_frame = pd.merge_asof(
        full_frame,
        provider_asset_market_data,
        on="timestamp",
        by=["provider_id", "from_asset_id", "to_asset_id"],
        direction="backward",
    )

    # Split the full market frame by order and create pairs - only keep essential columns.
    close_1 = full_market_frame[full_market_frame["order"] == 1][
        ["timestamp", "provider_asset_group_id", "close"]
    ].rename(columns={"close": "close_1"})
    close_2 = full_market_frame[full_market_frame["order"] == 2][
        ["timestamp", "provider_asset_group_id", "close"]
    ].rename(columns={"close": "close_2"})

    # Merge the close_1 and close_2 frames to create pairs - only timestamp, close_1, close_2
    pairs = pd.merge(
        close_1, close_2, on=["timestamp", "provider_asset_group_id"], how="inner"
    )

    # Keep only essential columns.
    pairs = pairs[["provider_asset_group_id", "timestamp", "close_1", "close_2"]]

    # Set index to provider_asset_group_id.
    pairs = pairs.set_index("provider_asset_group_id")

    return pairs


@task()
async def get_pairs_trading_frame(
    start: dt.datetime,
    end: dt.datetime,
    provider_asset_group_ids: list[int],
    provider_asset_group_member_data: pd.DataFrame,
    provider_asset_market_data: pd.DataFrame,
) -> dd.DataFrame:
    """
    Get the pairs trading frame with only essential columns for cointegration analysis.

    Args:
        start: Start datetime (timezone-naive)
        end: End datetime (timezone-naive)
        provider_asset_group_ids: List of provider asset group IDs to process
        members_data: Pre-loaded DataFrame of provider asset group members
        market_data_future: Broadcasted market data future
        n_workers: Number of parallel workers

    Returns:
        Dask DataFrame indexed by provider_asset_group_id with columns:
            - timestamp
            - close_1
            - close_2
    """
    # Split provider asset groups into chunks
    n_chunks = min(DASK_N_WORKERS, len(provider_asset_group_ids))
    group_chunks = np.array_split(provider_asset_group_ids, n_chunks)

    # Create delayed tasks with filtered member chunks
    delayed_dfs = []
    for chunk in group_chunks:
        # Filter members data for this specific chunk
        provider_asset_group_member_data_chunk = provider_asset_group_member_data[
            provider_asset_group_member_data["provider_asset_group_id"].isin(
                chunk.tolist()
            )
        ].copy()

        delayed_dfs.append(
            load_pairs_trading_frame_chunk(
                start,
                end,
                provider_asset_group_member_data_chunk,
                provider_asset_market_data,
            )
        )

    # Define minimal schema
    meta = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "close_1": pd.Series(dtype="float64"),
            "close_2": pd.Series(dtype="float64"),
        }
    )
    meta.index = pd.Index([], name="provider_asset_group_id", dtype="int64")

    # Convert to Dask DataFrame
    pairs_trading_frame = dd.from_delayed(delayed_dfs, meta=meta)

    # Set index to provider_asset_group_id
    pairs_trading_frame = pairs_trading_frame.set_index(
        "provider_asset_group_id", sorted=True
    )

    return pairs_trading_frame


@flow()
async def refresh_pairs_trading_attribute_data(
    date: Optional[dt.date] = None,
    lookback_window_days: Optional[int] = None,
):
    """
    Refresh the pairs trading attribute data.

    Args:
        date: The date to refresh the pairs trading attribute data for. If not provided, the current date will be used.
        lookback_window_days: The lookback window in days. If not provided, the default lookback window will be used.
    """
    logger = get_run_logger()

    # If the date is not provided, set it to today.
    if date is None:
        date = dt.date.today()

    # If the lookback window is not provided, set it to 30 days.
    if lookback_window_days is None:
        ValueError("Lookback window days must be provided.")

    # Get the start and end dates.
    end = dt.datetime.combine(date, dt.time.min)
    start = end - dt.timedelta(days=lookback_window_days)
    start_naive = start.replace(tzinfo=None).replace(second=0, microsecond=0)
    end_naive = end.replace(tzinfo=None).replace(second=0, microsecond=0)

    # Get the provider asset group ids.
    provider_asset_group_ids = await get_active_provider_asset_group_ids()

    # Get the provider asset group member data.
    provider_asset_group_member_data = await get_provider_asset_group_member_data(
        provider_asset_group_ids=provider_asset_group_ids,
    )
    provider_ids = provider_asset_group_member_data["provider_id"].unique()
    from_asset_ids = provider_asset_group_member_data["from_asset_id"].unique()
    to_asset_ids = provider_asset_group_member_data["to_asset_id"].unique()

    # Get the market data for the window.
    provider_asset_market_data = await get_provider_asset_group_market_data(
        start=start_naive,
        end=end_naive,
        provider_ids=provider_ids,
        from_asset_ids=from_asset_ids,
        to_asset_ids=to_asset_ids,
    )

    # Create the dask cluster.
    cluster: Cluster | LocalCluster = await create_dask_cluster(use_local_cluster=True)
    try:
        # Get the dask client.
        client: Client = cluster.get_client()

        # Scatter the provider asset market data to the dask cluster.
        provider_asset_market_data_future = await client.scatter(
            provider_asset_market_data, broadcast=True
        )

        # Get the pairs trading frame.
        pairs_trading_frame = await get_pairs_trading_frame(
            start=start_naive,
            end=end_naive,
            provider_asset_group_member_data=provider_asset_group_member_data,
            provider_asset_market_data=provider_asset_market_data_future,
        )

        # Compute the pairs trading frame.
    finally:
        # Shutdown the dask cluster.
        logger.info("Shutting down the dask cluster...")
        await shutdown_dask_cluster(cluster=cluster)
