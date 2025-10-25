import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
from typing import Optional

import pandas as pd
from prefect import flow, task, serve, get_run_logger
from mc_postgres_db.models import ProviderAssetMarket
from prefect.cache_policies import NO_CACHE
from prefect.concurrency.asyncio import rate_limit
from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from src.market.abstract import AbstractProviderAssetMarketData
from src.market.market_data import KrakenProviderAssetMarketData


@task(cache_policy=NO_CACHE)
async def get_data(
    market_data: AbstractProviderAssetMarketData, as_of_date: Optional[dt.date] = None
) -> pd.DataFrame:
    logger = get_run_logger()

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

    # Check for any asset codes that are not in the provider asset map.
    missing_asset_codes = asset_codes.loc[
        ~asset_codes.isin(provider_asset_map.keys())
    ].tolist()
    if len(missing_asset_codes) > 0:
        raise ValueError(
            f"The following asset codes are not in the provider asset map: {missing_asset_codes}"
        )

    # Get the asset pairs.
    await rate_limit(market_data.rate_limit_name, strict=True)
    asset_pairs = await market_data.get_asset_pairs(
        asset_codes=asset_codes, as_of_date=as_of_date
    )

    # Check the type of the asset pairs.
    if not isinstance(asset_pairs, pd.DataFrame):
        raise ValueError("The asset pairs must be a pandas dataframe.")

    # Check that the asset pairs have the required columns.
    missing_asset_pair_columns = set(
        market_data.required_asset_pair_data_columns
    ) - set(asset_pairs.columns)
    if len(missing_asset_pair_columns) > 0:
        raise ValueError(
            f"The asset pairs must have the following columns: {missing_asset_pair_columns}"
        )

    # Get the market data.
    data = pd.DataFrame(
        {
            column_name: pd.Series(
                [],
                dtype=(
                    "datetime64[ns]"
                    if ProviderAssetMarket.__table__.columns[
                        column_name
                    ].type.python_type
                    is dt.datetime
                    else pd.api.types.pandas_dtype(
                        ProviderAssetMarket.__table__.columns[
                            column_name
                        ].type.python_type
                    )
                ),
            )
            for column_name in set(market_data.key_columns + market_data.columns)
        }
    )
    for asset_pair in asset_pairs.to_dict(orient="records"):
        try:
            # Log the asset pair.
            logger.info(f"Getting data for {asset_pair['pair_asset_code']}...")

            # Request the market data for the asset pair.
            await rate_limit(market_data.rate_limit_name, strict=True)
            data_i = await market_data.get_market_data(
                pair_asset_code=asset_pair["pair_asset_code"],
                from_asset_code=asset_pair["from_asset_code"],
                to_asset_code=asset_pair["to_asset_code"],
                as_of_date=as_of_date,
            )

            # Check that the data has the required columns.
            invalid_columns = set(data_i.columns).symmetric_difference(
                set(market_data.columns + market_data.required_market_data_columns)
            )
            if invalid_columns:
                raise ValueError(f"The data has invalid columns: {invalid_columns}")

            # Check for from asset codes that are not in the provider asset map.
            missing_from_asset_codes = (
                data_i["from_asset_code"]
                .loc[
                    ~data_i["from_asset_code"].isin(provider_asset_map.keys())  # type: ignore
                ]
                .tolist()
            )
            if len(missing_from_asset_codes) > 0:
                raise ValueError(
                    f"The following from asset codes are not in the provider asset map: {missing_from_asset_codes}"
                )

            # Check for to asset codes that are not in the provider asset map.
            missing_to_asset_codes = (
                data_i["to_asset_code"]
                .loc[
                    ~data_i["to_asset_code"].isin(provider_asset_map.keys())  # type: ignore
                ]
                .tolist()
            )
            if len(missing_to_asset_codes) > 0:
                raise ValueError(
                    f"The following to asset codes are not in the provider asset map: {missing_to_asset_codes}"
                )

            # Match assets based on the asset_code column and the provider_asset table.
            data_i["from_asset_id"] = data_i["from_asset_code"].apply(
                lambda asset_code: provider_asset_map[asset_code]
            )
            data_i["to_asset_id"] = data_i["to_asset_code"].apply(
                lambda asset_code: provider_asset_map[asset_code]
            )

            # Add the provider id column.
            data_i["provider_id"] = market_data.get_provider().id

            # Drop the asset_code column.
            data_i = data_i.drop(columns=["from_asset_code", "to_asset_code"])

            # Add the data to the existing data.
            data = pd.concat([data, data_i])
        except Exception as e:
            logger.error(f"Error getting data for {asset_pair['pair_asset_code']}: {e}")
            continue

    return data


@flow()
async def pull_provider_asset_market_data(as_of_date: Optional[dt.date] = None, batch_size: int = 10000):
    logger = get_run_logger()

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
        logger.info(
            f"Getting data for the provider type {instance.__class__.__name__}..."
        )

        # Get the data for the instance.
        data = await get_data(market_data=instance, as_of_date=as_of_date)

        # Save the data to the database. Batch by 10,000 rows at a time.
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            await set_data(ProviderAssetMarket.__tablename__, batch, operation_type="upsert")


if __name__ == "__main__":
    pull_provider_asset_market_data_deployment = (
        pull_provider_asset_market_data.to_deployment(
            name="pull_provider_asset_market_data_debug",
        )
    )
    serve(pull_provider_asset_market_data_deployment)  # type: ignore
