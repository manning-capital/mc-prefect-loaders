import datetime as dt
from typing import Optional

import polars as pl
import dask.dataframe as dd
import mc_postgres_db.models as models
from prefect import flow, task, get_run_logger
from sqlalchemy import select
from prefect_dask import DaskTaskRunner, get_dask_client
from prefect.cache_policies import NO_CACHE
from mc_postgres_db.prefect.asyncio.tasks import get_engine

from src.attributes.abstract import AbstractAssetGroupType
from src.attributes.asset_group_attributes import StatisticalPairsTrading


@task(cache_policy=NO_CACHE)
async def refresh_by_asset_group_type(
    asset_group_type: AbstractAssetGroupType, start: dt.datetime, end: dt.datetime
):
    """
    Refresh the provider asset attribute data.
    """
    logger = get_run_logger()

    # Get an engine.
    engine = await get_engine()

    # Refresh the provider asset groups.
    logger.info(
        f"Refreshing the provider asset groups for {asset_group_type.asset_group_type.name}..."
    )
    asset_group_type.refresh_provider_asset_groups(start=start, end=end)

    # Get the current provider asset groups.
    logger.info(
        f"Getting the current provider asset groups for {asset_group_type.asset_group_type.name}..."
    )
    provider_asset_groups = asset_group_type.get_current_provider_asset_groups()

    # Generate a data from for all provider asset groups.
    provider_asset_group_ids = [
        provider_asset_group.id for provider_asset_group in provider_asset_groups
    ]
    provider_asset_group_members_df = pl.read_database(
        select(models.ProviderAssetGroupMember).where(
            models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                provider_asset_group_ids
            )
        ),
        engine,
    )

    # Get the provider asset market data for each provider asset group.
    key_columns: set[str] = {
        models.ProviderAssetMarket.timestamp,
        models.ProviderAssetMarket.provider_id,
        models.ProviderAssetMarket.from_asset_id,
        models.ProviderAssetMarket.to_asset_id,
    }
    query_columns: set[str] = key_columns.union(
        asset_group_type.provider_asset_market_columns
    )
    provider_asset_market_df: pl.DataFrame = pl.read_database(
        select(*query_columns).where(
            models.ProviderAssetMarket.provider_id.in_(
                provider_asset_group_members_df["provider_id"]
            ),
            models.ProviderAssetMarket.from_asset_id.in_(
                provider_asset_group_members_df["from_asset_id"]
            ),
            models.ProviderAssetMarket.to_asset_id.in_(
                provider_asset_group_members_df["to_asset_id"]
            ),
            models.ProviderAssetMarket.timestamp
            >= start - dt.timedelta(days=1),  # Get an extra day so we can forward fill.
            models.ProviderAssetMarket.timestamp <= end,
        ),
        engine,
    )

    # Cross-join the provider asset market data with the provider asset group members.
    provider_asset_market_df: pl.DataFrame = provider_asset_market_df.join(
        provider_asset_group_members_df,
        on=["provider_id", "from_asset_id", "to_asset_id"],
        how="cross",
    )

    # Pivot the provider asset market data based on the order so that each provider asset group member is a column.
    provider_asset_group_market_df: pl.DataFrame = provider_asset_market_df.pivot(
        index="timestamp", columns="order", values=query_columns
    )
    pivotted_id_columns: list[str] = [
        f"{key_column}_{i}"
        for i in range(1, asset_group_type.group_size + 1)
        for key_column in key_columns
    ]

    # Group by and forward fill the provider asset market data.
    provider_asset_group_market_df: pl.DataFrame = (
        provider_asset_group_market_df.group_by(pivotted_id_columns).agg(
            pl.col("*").sort_by("timestamp").forward_fill()
        )
    )
    provider_asset_group_market_df = provider_asset_group_market_df.filter(
        pl.col("timestamp") >= start
    )
    provider_asset_group_market_df = provider_asset_group_market_df.filter(
        pl.col("timestamp") <= end
    )
    provider_asset_group_market_df = provider_asset_group_market_df.dropna()
    provider_asset_group_market_df: dd.DataFrame = dd.from_pandas(
        provider_asset_group_market_df.to_pandas(), npartitions=4
    )

    # Calculate the attributes for the provider asset market data dataframes.
    for window in asset_group_type.windows:
        logger.info(
            f"Calculating attributes for {asset_group_type.asset_group_type.name} with window {window}..."
        )
        with get_dask_client():
            attribute_results = provider_asset_group_market_df.group_by(
                pivotted_id_columns
            ).apply(
                lambda df: asset_group_type.calculate_group_attributes(
                    window=window,
                    step=asset_group_type.step,
                    group_market_df=df,
                )
            )


@flow(
    task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": 4, "threads_per_worker": 2})
)
async def refresh_provider_asset_attribute_data(
    start: Optional[dt.datetime] = None, end: Optional[dt.datetime] = None
):
    """
    Refresh the provider asset attribute data.
    """
    logger = get_run_logger()

    # If the start or end is not provided, set it to today.
    if (start is None) or (end is None):
        end = dt.datetime.now()
        start = end - dt.timedelta(days=1)
        logger.info(
            f"Start or end not provided, setting start to {start} and end to {end}."
        )

    # Get an engine.
    engine = await get_engine()

    # Initialize the asset group type.
    asset_group_types = [StatisticalPairsTrading(engine)]

    # Refresh the provider asset attribute data for each asset group type.
    for asset_group_type in asset_group_types:
        logger.info(
            f"Refreshing the provider asset attribute data for {asset_group_type.asset_group_type.name}..."
        )
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)


if __name__ == "__main__":
    refresh_provider_asset_attribute_data()
