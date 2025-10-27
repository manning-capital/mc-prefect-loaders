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
    provider_asset_group_id_list = list(provider_asset_group_ids)
    batch_size = asset_group_type.batch_size

    # Calculate the attributes for the provider asset market data dataframes.
    for window in asset_group_type.windows:
        window_duration = humanize.naturaldelta(window)
        logger.info(
            f"Processing {len(provider_asset_group_id_list)} groups in batches of {batch_size} for {window_duration} window..."
        )

        # Process in batches
        for i in range(0, len(provider_asset_group_id_list), batch_size):
            batch_ids = set(provider_asset_group_id_list[i : i + batch_size])
            batch_num = (i // batch_size) + 1
            total_batches = (
                len(provider_asset_group_id_list) + batch_size - 1
            ) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} groups) for {window_duration} window..."
            )

            # Get the provider asset group market data for the batch.
            provider_asset_group_members_df: pl.DataFrame = (
                asset_group_type.get_provider_asset_group_market_data(
                    provider_asset_group_ids=batch_ids,
                    start=start - window,
                    end=end,
                )
            )

            # Calculate the attributes for the provider asset group market data dataframes.
            logger.info(
                f"Calculating attributes for batch {batch_num}/{total_batches} with {window_duration} window..."
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
                provider_asset_group_id = name[0]
                attribute_results = asset_group_type.calculate_group_attributes(
                    window=window,
                    step=asset_group_type.step,
                    group_market_df=data,
                )
                attribute_results = attribute_results.with_columns(
                    pl.lit(provider_asset_group_id, dtype=pl.Int64).alias(
                        models.ProviderAssetGroupAttribute.provider_asset_group_id.name
                    ),
                    pl.lit(int(window.total_seconds()), dtype=pl.Int64).alias(
                        models.ProviderAssetGroupAttribute.lookback_window_seconds.name
                    ),
                )

                # Drop nulls before setting the data.
                to_set_data = attribute_results.drop_nulls().to_pandas()

                # Set the data.
                await set_data(
                    models.ProviderAssetGroupAttribute.__tablename__, to_set_data
                )


@flow()
async def refresh_provider_asset_attribute_data(
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    default_lookback_hours: int = 24,
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
