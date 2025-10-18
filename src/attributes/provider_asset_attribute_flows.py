import datetime as dt
from typing import Optional

import polars as pl
from prefect import flow, task, get_run_logger
from prefect_dask import DaskTaskRunner
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

    # Calculate the attributes for the provider asset market data dataframes.
    for window in asset_group_type.windows:
        # Get the provider asset group market data for the window.
        logger.info(
            f"Getting the provider asset group market data for {asset_group_type.asset_group_type.name} with window {window}..."
        )
        provider_asset_group_members_df: pl.DataFrame = (
            asset_group_type.get_provider_asset_group_market_data(
                provider_asset_group_ids=provider_asset_group_ids,
                start=start - window,
                end=end,
            )
        )

        # Calculate the attributes for the provider asset group market data dataframes.
        logger.info(
            f"Calculating attributes for {asset_group_type.asset_group_type.name} with window {window}..."
        )
        for name, data in provider_asset_group_members_df.group_by(
            [
                "provider_asset_group_id",
            ]
        ):
            # Check if the data is empty.
            if data.is_empty():
                logger.info(f"Data is empty for {name}, skipping...")
                continue

            # Calculate the attributes for the provider asset group market data dataframes.
            logger.info(f"Calculating attributes for {name}...")
            attribute_results = asset_group_type.calculate_group_attributes(
                window=window,
                step=asset_group_type.step,
                group_market_df=data,
            )
            logger.info(f"Attribute results: {attribute_results}")


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
