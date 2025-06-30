import requests
import pandas as pd
import datetime as dt
from sqlalchemy.orm import Session
from prefect import flow, task, get_run_logger
from src.utils.tasks import get_postgres_url
from sqlalchemy import create_engine, select

from mcpdb.tables import ProviderContent, Provider, ProviderAsset, Asset, AssetType


@task()
async def get_coin_desk_categories() -> set[str]:
    url = get_postgres_url()
    engine = create_engine(url)
    categories: set[str] = set()
    with Session(engine) as session:
        # Get the coin desk provider.
        stmt = select(Provider).where(Provider.name == "CoinDesk")
        provider = session.execute(stmt).scalar_one_or_none()

        # Get all categories from provider assets.
        if provider:
            # Get sub query that connects Asset and AssetType to get all assets that have a type of "DIGITAL_CURRENCY".
            sub_query = (
                select(Asset)
                .join(AssetType, AssetType.id == Asset.asset_type_id)
                .where(
                    AssetType.name == "DIGITAL_CURRENCY",
                    Asset.is_active.is_(True),
                )
            )

            # Get sub query with alias to get all provider assets that are active.
            stmt = (
                select(ProviderAsset)
                .join(sub_query, Asset.id == ProviderAsset.asset_id)
                .join(Provider, Provider.id == ProviderAsset.provider_id)
                .where(
                    ProviderAsset.provider_id == provider.id,
                    Asset.is_active.is_(True),
                    ProviderAsset.is_active.is_(True),
                )
            )

            # Use the sub query to select asset codes
            provider_assets = session.execute(stmt).scalars().all()
            for provider_asset in provider_assets:
                if provider_asset.asset_code:
                    categories.add(provider_asset.asset_code)

    return categories


@task(retries=3, retry_delay_seconds=5)
async def get_recent_coin_desk_content(coin_desk_news_provider_ids: list[str]):
    """
    Task to get content from CoinDesk for a specific provider.
    """

    # Get categories to filter content.
    categories = await get_coin_desk_categories()

    # Get get the provider.

    # Retrieve recent content from CoinDesk.
    host = "https://data-api.coindesk.com"
    uri = "/news/v1/article/list"
    params = {
        "lang": "EN",
        "limit": 100,
        "to_ts": (dt.datetime.now() - dt.timedelta(hours=2)).timestamp(),
        "categories": ",".join(categories),
        "source_ids": ",".join([provider.name.lower()]),
    }
    response = requests.get(f"{host}{uri}", params=params)

    # Check if the response is successful.
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data from Coindesk API: {response.status_code}"
        )

    # Parse the JSON response.
    data = response.json()["Data"]
    df = pd.DataFrame(
        [
            {
                "timestamp": dt.datetime.fromtimestamp(item["PUBLISHED_ON"]),
            }
            for item in data
        ],
        columns=[],
    )

    # Convert types.

    return df


@task()
async def get_current_coin_desk_content(
    provider_ids: list[str],
    from_datetime: dt.datetime = None,
    to_datetime: dt.datetime = None,
) -> pd.DataFrame:
    """
    Task to get current content from the Postgres database.
    """
    url = await get_postgres_url()
    engine = create_engine(url)
    logger = get_run_logger()
    with Session(engine) as session:
        # Query the database for content from the specified providers.
        stmt = select(ProviderContent).where(
            ProviderContent.provider_id.in_(provider_ids)
        )
        if from_datetime:
            stmt = stmt.where(ProviderContent.created_at >= from_datetime)
        if to_datetime:
            stmt = stmt.where(ProviderContent.created_at <= to_datetime)

        # Execute the query and fetch results.
        df = pd.read_sql(stmt, session.bind)

        # Check if the DataFrame is empty and log a warning if it is.
        if df.empty:
            logger.warning("No provider asset data found for the given IDs.")

        return df


@task()
def save_coin_desk_content(new_content: pd.DataFrame, old_content: pd.DataFrame):
    """
    Task to save CoinDesk content to the database.
    """
    pass


@flow()
async def pull_coin_desk_content(provider_ids: list[str]):
    """
    Flow to pull content from CoinDesk.
    """
    logger = get_run_logger()

    # Get the current content from the database.
    logger.info("Fetching current CoinDesk content from the database.")
    current = await get_current_coin_desk_content(provider_ids)

    # Get the recent content from CoinDesk.
    logger.info("Fetching recent CoinDesk content.")
    recent = await get_recent_coin_desk_content(provider_ids)

    # Save down new content to the database.
    logger.info("Saving new CoinDesk content to the database.")
    await save_coin_desk_content(recent, current)
