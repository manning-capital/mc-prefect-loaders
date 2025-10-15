import datetime as dt
from typing import Optional

import pandas as pd
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
    provider_asset_group_ids = asset_group_type.get_current_provider_asset_group_ids()

    # Generate a data from for all provider asset groups.
    provider_asset_group_members_ddf: dd.DataFrame = dd.read_sql(
        sql=select(
            models.ProviderAssetGroupMember.provider_asset_group_id,
            models.ProviderAssetGroupMember.order,
            models.ProviderAssetMarket.timestamp,
            models.ProviderAssetMarket.provider_id,
            models.ProviderAssetMarket.from_asset_id,
            models.ProviderAssetMarket.to_asset_id,
            *asset_group_type.provider_asset_market_columns,
        )
        .join(
            models.ProviderAssetMarket,
            models.ProviderAssetGroupMember.provider_id
            == models.ProviderAssetMarket.provider_id
            and models.ProviderAssetGroupMember.from_asset_id
            == models.ProviderAssetMarket.from_asset_id
            and models.ProviderAssetGroupMember.to_asset_id
            == models.ProviderAssetMarket.to_asset_id,
        )
        .where(
            models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                provider_asset_group_ids
            ),
            models.ProviderAssetMarket.timestamp
            >= start - dt.timedelta(days=1),  # Get an extra day so we can forward fill.
            models.ProviderAssetMarket.timestamp <= end,
        ),
        con=engine.url.render_as_string(hide_password=False),
        index_col=models.ProviderAssetGroupMember.provider_asset_group_id.name,
    )

    # Convert to pandas.
    provider_asset_group_members_dff: pd.DataFrame = (
        provider_asset_group_members_ddf.compute()
    )

    # Construct the timeframe.
    tf_all = dd.from_pandas(
        {
            "timestamp": pd.date_range(
                start=provider_asset_group_members_ddf["timestamp"].min().compute(),
                end=provider_asset_group_members_ddf["timestamp"].max().compute(),
                freq="1min",
            )
        }
    )
    tf = dd.from_pandas({"timestamp": pd.date_range(start=start, end=end, freq="1min")})

    # Forward fill the provider asset market data.
    all_columns = set(provider_asset_group_members_ddf.columns)
    id_columns = {"provider_id", "from_asset_id", "to_asset_id"}
    non_id_columns = all_columns - id_columns
    provider_asset_group_members_ddf = provider_asset_group_members_ddf.join(
        tf_all,
        how="cross",
    )
    provider_asset_group_members_ddf = (
        provider_asset_market_df.group_by(id_columns)
        .agg(pl.col("*").sort_by("timestamp").forward_fill())
        .explode(non_id_columns)
    )
    provider_asset_market_df = provider_asset_market_df.filter(
        pl.col("timestamp") >= start
    )
    provider_asset_market_df = provider_asset_market_df.filter(
        pl.col("timestamp") <= end
    )
    provider_asset_market_df = provider_asset_market_df.drop_nulls()

    # Cross-join the provider asset market data with the provider asset group members.
    provider_asset_group_members_df: pl.DataFrame = (
        provider_asset_group_members_df.join(
            tf,
            how="cross",
        )
    )
    provider_asset_group_members_df = provider_asset_group_members_df.join(
        provider_asset_market_df,
        on=["timestamp", "provider_id", "from_asset_id", "to_asset_id"],
        how="left",
    )

    # Pivot the provider asset market data based on the order so that each provider asset group member is a column.
    provider_asset_group_members_df: pl.DataFrame = (
        provider_asset_group_members_df.pivot(
            index="timestamp",
            columns="order",
            values={column.name for column in query_columns} - set(["timestamp"]),
        )
    )
    pivotted_id_columns: list[str] = [
        f"{key_column}_{i}"
        for i in range(1, asset_group_type.group_size + 1)
        for key_column in key_columns
    ]

    # Group by and forward fill the provider asset market data.
    provider_asset_group_members_df = provider_asset_group_members_df.drop_nulls()
    provider_asset_group_members_ddf: dd.DataFrame = dd.from_pandas(
        provider_asset_group_members_df.to_pandas(), npartitions=4
    )

    # Calculate the attributes for the provider asset market data dataframes.
    for window in asset_group_type.windows:
        logger.info(
            f"Calculating attributes for {asset_group_type.asset_group_type.name} with window {window}..."
        )
        with get_dask_client() as client:
            attribute_results = provider_asset_group_members_ddf.group_by(
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
