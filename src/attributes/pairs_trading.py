import datetime as dt
from typing import Optional

import dask
import numpy as np
import pandas as pd
import dask.dataframe as dd
import statsmodels.api as sm
import mc_postgres_db.models as models
from dask import delayed
from coiled import Cluster
from prefect import flow, task, get_run_logger
from sqlalchemy import select
from sqlalchemy.orm import Session
from dask.distributed import Client, LocalCluster
from prefect.blocks.system import Secret
from statsmodels.tsa.stattools import coint
from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from src.attributes.stochastic_models import OrnsteinUhlenbeck

DASK_CLUSTER_NAME = "prefect-cluster"
DASK_N_WORKERS = 40
DASK_REGION = "us-east-1"
DASK_CONTAINER = "ghcr.io/manning-capital/mc-prefect-loaders:main"
DASK_WORKER_MEMORY = "16GB"
DASK_WORKER_CPU = 2

MAX_PROVIDER_ASSET_GROUPS = 5000

COINTEGRATION_P_VALUE_THRESHOLD = 0.001


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
        coiled_api_key: str = (await Secret.load("coiled-api-key")).value()

        # Set the coiled token.
        dask.config.set({"coiled.token": coiled_api_key})

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
async def get_active_provider_asset_group_ids() -> list[int]:
    """
    Get the active provider asset groups.
    """
    engine = await get_engine()
    with Session(engine) as session:
        return list(
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
            models.ProviderAssetMarket.to_asset_id.in_(to_asset_ids),
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
    client: Client,
) -> dd.DataFrame:
    """
    Get the pairs trading frame with only essential columns for cointegration analysis.

    Args:
        start: Start datetime (timezone-naive)
        end: End datetime (timezone-naive)
        provider_asset_group_ids: List of provider asset group IDs to process
        members_data: Pre-loaded DataFrame of provider asset group members
        market_data_future: Broadcasted market data future
        client: Dask client

    Returns:
        Dask DataFrame indexed by provider_asset_group_id with columns:
            - timestamp
            - close_1
            - close_2
    """
    # Split provider asset groups into chunks
    n_chunks = min(len(client.cluster.workers), len(provider_asset_group_ids))
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


def get_cointegrated_stats(df: pd.DataFrame) -> pd.Series:
    """
    Get the cointegrated stats for a given dataframe.
    """

    # Compute the linear regression.
    X = df["close_1"].to_numpy()
    y = df["close_2"].to_numpy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Get the residuals.
    linear_fit_alpha = results.params[0]
    linear_fit_beta = results.params[1]
    linear_fit_mse = results.mse_total
    linear_fit_r_squared = results.rsquared
    linear_fit_r_squared_adj = results.rsquared_adj
    residuals = results.resid

    # Get the cointegration stats.
    ou_params = OrnsteinUhlenbeck().fit(residuals)

    return pd.Series(
        [
            linear_fit_alpha,
            linear_fit_beta,
            linear_fit_mse,
            linear_fit_r_squared,
            linear_fit_r_squared_adj,
            ou_params.mu,
            ou_params.theta,
            ou_params.sigma,
        ],
        index=[
            "linear_fit_alpha",
            "linear_fit_beta",
            "linear_fit_mse",
            "linear_fit_r_squared",
            "linear_fit_r_squared_adj",
            "ou_mu",
            "ou_theta",
            "ou_sigma",
        ],
        dtype=float,
    )


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
    provider_ids = provider_asset_group_member_data["provider_id"].tolist()
    from_asset_ids = provider_asset_group_member_data["from_asset_id"].tolist()
    to_asset_ids = provider_asset_group_member_data["to_asset_id"].tolist()

    # Get the market data for the window.
    provider_asset_market_data = await get_provider_asset_group_market_data(
        start=start_naive,
        end=end_naive,
        provider_ids=provider_ids,
        from_asset_ids=from_asset_ids,
        to_asset_ids=to_asset_ids,
    )

    # Create the dask cluster.
    logger.info("Creating the dask cluster...")
    cluster: Cluster | LocalCluster = await create_dask_cluster(use_local_cluster=True)
    try:
        # Get the dask client.
        client: Client = cluster.get_client()

        # Log the address of the dask cluster.
        logger.info(f"Dask cluster address: {client.dashboard_link}")

        # Scatter the provider asset market data to the dask cluster.
        logger.info("Scattering the provider asset market data to the dask cluster...")
        provider_asset_market_data_future = client.scatter(
            provider_asset_market_data, broadcast=True
        )

        # Get the pairs trading frame.
        logger.info("Getting the pairs trading frame...")
        pairs_trading_frame: dd.DataFrame = await get_pairs_trading_frame(
            start=start_naive,
            end=end_naive,
            provider_asset_group_ids=provider_asset_group_ids,
            provider_asset_group_member_data=provider_asset_group_member_data,
            provider_asset_market_data=provider_asset_market_data_future,
            client=client,
        )
        pairs_trading_frame = pairs_trading_frame.persist()

        # Compute the pairs trading frame.
        cointegration_p_values: dd.DataFrame = pairs_trading_frame.groupby(
            "provider_asset_group_id"
        )[["close_1", "close_2"]].apply(
            lambda df: pd.Series(
                coint(df["close_1"], df["close_2"])[1], index=["p_value"]
            ),
            meta={"p_value": pd.Series([], dtype=float)},
        )

        # Compute the cointegration p-values.
        logger.info("Computing the cointegration p-values...")
        cointegration_p_values_computed: pd.DataFrame = cointegration_p_values.compute()

        # Filter the provider asset group member data to only include the provider asset group ids with cointegration p-values less than a threshold.
        logger.info(
            f"Filtering the pairs trading frame to only include the provider asset group ids with cointegration p-values less than {COINTEGRATION_P_VALUE_THRESHOLD}..."
        )
        cointegrated_provider_asset_group_ids = cointegration_p_values_computed.loc[
            cointegration_p_values_computed["p_value"] < COINTEGRATION_P_VALUE_THRESHOLD
        ].index.tolist()
        logger.info(
            f"Found {len(cointegrated_provider_asset_group_ids)} cointegrated provider asset group ids."
        )

        # Get the cointegrated provider asset group member data.
        cointegrated_pairs_trading_frame = pairs_trading_frame.loc[
            pairs_trading_frame.index.isin(cointegrated_provider_asset_group_ids)
        ]

        # Compute the statistical attributes for the cointegrated pairs trading frame.
        cointegrated_pairs_trading_stats = cointegrated_pairs_trading_frame.groupby(
            "provider_asset_group_id"
        )[["close_1", "close_2"]].apply(
            lambda df: get_cointegrated_stats(df),
            meta={
                "linear_fit_alpha": pd.Series([], dtype=float),
                "linear_fit_beta": pd.Series([], dtype=float),
                "linear_fit_mse": pd.Series([], dtype=float),
                "linear_fit_r_squared": pd.Series([], dtype=float),
                "linear_fit_r_squared_adj": pd.Series([], dtype=float),
                "ou_mu": pd.Series([], dtype=float),
                "ou_theta": pd.Series([], dtype=float),
                "ou_sigma": pd.Series([], dtype=float),
            },
        )

        # Compute the cointegrated pairs trading frame attributes.
        logger.info("Computing the cointegrated pairs trading frame attributes...")
        cointegrated_pairs_trading_stats_computed = (
            cointegrated_pairs_trading_stats.compute()
        )

        # Merge the cointegration p-values and the cointegrated pairs trading stats.
        toset = cointegration_p_values_computed.merge(
            cointegrated_pairs_trading_stats_computed, left_index=True, right_index=True
        ).reset_index()
        toset = toset.rename(columns={"p_value": "cointegration_p_value"})
        toset["lookback_window_seconds"] = lookback_window_days * 24 * 60 * 60
        toset["timestamp"] = dt.datetime.combine(date, dt.time.min)
        toset = toset[
            [
                "timestamp",
                "provider_asset_group_id",
                "lookback_window_seconds",
                "cointegration_p_value",
                "linear_fit_alpha",
                "linear_fit_beta",
                "linear_fit_mse",
                "linear_fit_r_squared",
                "linear_fit_r_squared_adj",
                "ou_mu",
                "ou_theta",
                "ou_sigma",
            ]
        ]

        # Set the data to the database.
        await set_data(models.ProviderAssetGroupAttribute.__tablename__, toset)

    finally:
        # Shutdown the dask cluster.
        logger.info("Shutting down the dask cluster...")
        await shutdown_dask_cluster(cluster=cluster)
