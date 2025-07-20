from prefect import flow, task
from src.market.market_data import KrakenProviderAssetMarketData
from src.market.abstract import AbstractProviderAssetMarketData
from mc_postgres_db.prefect.asyncio.tasks import get_engine, set_data
from prefect.cache_policies import NO_CACHE
from typing import Optional
import datetime as dt
import pandas as pd
from prefect.concurrency.asyncio import rate_limit
from mc_postgres_db.models import ProviderAssetMarket


@task(cache_policy=NO_CACHE)
async def get_data(
    market_data: AbstractProviderAssetMarketData, as_of_date: Optional[dt.date] = None
) -> pd.DataFrame:
    # Set the as_of_date to today if it is not provided.
    if as_of_date is None:
        as_of_date = dt.date.today()

    # Check if all the column(s) are in the database.
    invalid_columns = set(market_data.columns) - set(
        ProviderAssetMarket.__table__.columns.keys()
    )
    if len(invalid_columns) > 0:
        raise ValueError(
            f"The following columns are not in the database: {invalid_columns}"
        )

    # Get all the asset codes to request.
    provider_asset_map = market_data.get_provider_asset_map(as_of_date=as_of_date)
    asset_codes = pd.Series(provider_asset_map.keys())

    # Get the asset pairs.
    await rate_limit(market_data.rate_limit_name, strict=True)
    asset_pairs = await market_data.get_asset_pairs(
        asset_codes=asset_codes, as_of_date=as_of_date
    )

    # Check the type of the asset pairs.
    if not asset_pairs.dtype == pd.Series:
        raise ValueError("The asset pairs must be a pandas series.")

    # Check that the data type of the series is a tuple.
    if not all(isinstance(asset_pair, tuple) for asset_pair in asset_pairs):
        raise ValueError("The asset pairs must be a tuple of two asset codes.")

    # Check that each tuple has two elements.
    if not all(len(asset_pair) == 2 for asset_pair in asset_pairs):
        raise ValueError("The asset pairs must be a tuple of two asset codes.")

    # Get the market data.
    data = pd.DataFrame(
        {
            column_name: pd.Series(
                [],
                dtype=ProviderAssetMarket.__table__.columns[
                    column_name
                ].type.python_type,
            )
            for column_name in market_data.columns
        }
    )
    for asset_pair in asset_pairs:
        # Request the market data for the asset pair.
        await rate_limit(market_data.rate_limit_name, strict=True)
        data_i = await market_data.get_market_data(
            from_asset_code=asset_pair[0],
            to_asset_code=asset_pair[1],
            as_of_date=as_of_date,
        )

        # Check that the data has the required columns.
        invalid_columns = set(data_i.columns).symmetric_difference(
            set(market_data.columns + market_data.__base_required_columns)
        )
        if invalid_columns:
            raise ValueError(f"The data has invalid columns: {invalid_columns}")

        # Add the data to the existing data.
        data = pd.concat([data, data_i])

    # Check for any asset codes that are not in the provider asset map.
    missing_asset_codes = asset_codes.loc[
        ~asset_codes.isin(provider_asset_map.keys())
    ].tolist()
    if len(missing_asset_codes) > 0:
        raise ValueError(
            f"The following asset codes are not in the provider asset map: {missing_asset_codes}"
        )

    # Match assets based on the asset_code column and the provider_asset table.
    data["from_asset_id"] = data["from_asset_code"].apply(
        lambda asset_code: provider_asset_map[asset_code]
    )
    data["to_asset_id"] = data["to_asset_code"].apply(
        lambda asset_code: provider_asset_map[asset_code]
    )

    # Add the provider id column.
    data["provider_id"] = market_data.get_provider().id

    # Drop the asset_code column.
    data = data.drop(columns=["from_asset_code", "to_asset_code"])

    return data


@flow()
async def pull_provider_asset_market_data(as_of_date: Optional[dt.date] = None):
    # If the as_of_date is not provided, set it to today.
    if as_of_date is None:
        as_of_date = dt.date.today()

    # Get an engine.
    engine = await get_engine()

    # Collect all types of provider asset market data.
    data_classes = [KrakenProviderAssetMarketData]

    # Create an instance of each provider asset market data class.
    data_instances = [data_class(engine) for data_class in data_classes]

    # Get the data for each provider asset market data instance.
    for instance in data_instances:
        # Get the data for the instance.
        data = await get_data(market_data=instance, as_of_date=as_of_date)

        # Save the data to the database.
        await set_data(ProviderAssetMarket.__tablename__, data, operation_type="upsert")
