import pandas as pd
import datetime as dt
from prefect import flow, task, get_run_logger
from src.utils.tasks import get_postgres_url
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mcpdb.tables import ProviderContent


@task()
def get_recent_coin_desk_content(provider_ids: list[str]):
    """
    Task to get content from CoinDesk for a specific provider.
    """
    pass


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
    columns = [
        "published_at",
        "provider_id",
        "content_unique_identifier",
        "authors",
        "title",
        "description",
        "url",
        "image_url",
        "content",
        "created_at",
        "updated_at",
    ]
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
            return pd.DataFrame(columns=columns)

        return df[columns]


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
    recent = get_recent_coin_desk_content(provider_ids)

    # Save down new content to the database.
    logger.info("Saving new CoinDesk content to the database.")
    await save_coin_desk_content(recent, current)
