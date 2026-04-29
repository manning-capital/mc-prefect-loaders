import asyncio
import datetime as dt
from typing import Optional

import polars as pl
import humanize
import mc_postgres_db.models as models
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from src.attributes.abstract import AbstractAssetGroupType
from src.attributes.asset_group_attributes import StatisticalPairsTrading


@task(cache_policy=NO_CACHE)
async def calculate_attributes(
    asset_group_type: AbstractAssetGroupType,
    provider_asset_group_id: int,
    window: dt.timedelta,
    market_data: pl.DataFrame,
):
    """
    Calculate the attributes for the provider asset group market data dataframes.
    """
    logger = get_run_logger()

    # Calculate the attributes for the provider asset group market data dataframes.
    logger.info(
        f"Calculating attributes for provider asset group {provider_asset_group_id}..."
    )
    attribute_results = asset_group_type.calculate_group_attributes(
        window=window,
        step=asset_group_type.step,
        group_market_df=market_data,
    )
    attribute_results = attribute_results.with_columns(
        pl.lit(provider_asset_group_id, dtype=pl.Int64).alias(
            models.ProviderAssetGroupAttribute.provider_asset_group_id.name
        ),
        pl.lit(int(window.total_seconds()), dtype=pl.Int64).alias(
            models.ProviderAssetGroupAttribute.lookback_window_seconds.name
        ),
    )

    return attribute_results


@task(cache_policy=NO_CACHE)
async def refresh_by_asset_group_type(
    asset_group_type: AbstractAssetGroupType, start: dt.datetime, end: dt.datetime
):
    """
    Refresh the provider asset attribute data.
    """
    logger = get_run_logger()

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
    provider_asset_group_id_list = list(provider_asset_group_ids)
    
    # Save the attributes.
    logger.info(
        f"Saving the attributes for {asset_group_type.asset_group_type.name}..."
    )
    asset_group_type.save_attributes(provider_asset_group_ids=provider_asset_group_ids)


@flow()
async def refresh_provider_asset_attribute_data(
    date: Optional[dt.date] = None,
    lookback_window_days: Optional[int] = None,
):
    """
    Refresh the provider asset attribute data.
    """
    logger = get_run_logger()

    # If the start or end is not provided, set it to today.
    if (start is None) or (end is None):
        end = dt.datetime.now()
        start = end - dt.timedelta(hours=default_lookback_hours)
        logger.info(
            f"Start or end not provided, setting start to {start} and end to {end} (default lookback: {default_lookback_hours}h)."
        )

    # Log the processing time range
    total_hours = (end - start).total_seconds() / 3600
    logger.info(
        f"Processing provider asset attribute data from {start} to {end} (total range: {total_hours:.1f}h)"
    )

    # Get an engine.
    engine = await get_engine()

    try:
        # Initialize the asset group type.
        asset_group_types = [StatisticalPairsTrading(engine)]

        # Refresh the provider asset attribute data for each asset group type.
        for asset_group_type in asset_group_types:
            logger.info(
                f"Refreshing the provider asset attribute data for {asset_group_type.asset_group_type.name}..."
            )
            await refresh_by_asset_group_type(asset_group_type, start=start, end=end)
    finally:
        # Dispose the engine to release database connections back to the pool
        engine.dispose()
        logger.info("Disposed database engine connection pool")


if __name__ == "__main__":
    refresh_provider_asset_attribute_data()
