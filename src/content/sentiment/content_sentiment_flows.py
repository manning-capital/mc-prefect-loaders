import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

import datetime as dt
from typing import Optional
from prefect import flow, task, serve
import pandas as pd
from mc_postgres_db.models import ProviderContent, ProviderContentSentiment
from sqlalchemy import select, or_
from mc_postgres_db.prefect.asyncio.tasks import get_engine, set_data
from src.content.sentiment.abstract import AbstractContentSentimentType
from src.content.sentiment.sentiment_types import NLTKVaderContentSentimentType
from prefect.cache_policies import NO_CACHE

@task(cache_policy=NO_CACHE)
async def get_unprocessed_content_sentiment_data(
    from_date: dt.date, to_date: dt.date, sentiment_type: AbstractContentSentimentType
) -> pd.DataFrame:
    """
    Get the content sentiment data that hasn't been processed yet.

    Args:
        from_date: The date to get the content sentiment data for.
        to_date: The date to get the content sentiment data for.
        check_columns: The columns to check for null values.

    Returns:
        A pandas DataFrame of the content sentiment data.
    """

    # Check if the column(s) are valid.
    for column in sentiment_type.columns:
        if column not in ProviderContentSentiment.__table__.columns:
            raise ValueError(
                f"Column {column} is not a valid column in ProviderContentSentiment."
            )

    # Get the engine.
    engine = await get_engine()

    # Get the content sentiment data.
    stmt = (
        select(
            ProviderContent.id,
            ProviderContent.timestamp,
            ProviderContent.content,
            ProviderContentSentiment.sentiment_type_id,
            *[
                getattr(ProviderContentSentiment, column)
                for column in sentiment_type.columns
            ],
        )
        .where(
            ProviderContent.content_type_id.in_(sentiment_type.get_content_type_ids()),
            or_(
                *[
                    getattr(ProviderContentSentiment, column).is_(None)
                    for column in sentiment_type.columns
                ]
            ),
            ProviderContent.timestamp
            > dt.datetime.combine(
                from_date, dt.time.min, tzinfo=dt.timezone.utc
            ).replace(tzinfo=None),
            ProviderContent.timestamp
            < dt.datetime.combine(to_date, dt.time.max, tzinfo=dt.timezone.utc).replace(
                tzinfo=None
            ),
        )
        .join(
            ProviderContentSentiment,
            ProviderContent.id == ProviderContentSentiment.provider_content_id,
            isouter=True,
        )
    )

    return pd.read_sql(stmt, engine)


@flow()
async def refresh_content_sentiment(
    from_date: Optional[dt.date] = None, to_date: Optional[dt.date] = None
):
    # Set the from_date and to_date if they are not provided.
    if from_date is None:
        from_date = dt.date.today()
    if to_date is None:
        to_date = dt.date.today()

    # Check if the from_date is before the to_date.
    if from_date > to_date:
        raise ValueError("The from_date must be before the to_date.")

    # Get the engine.
    engine = await get_engine()

    # Get the sentiment types.
    sentiment_types = [
        NLTKVaderContentSentimentType(engine),
    ]

    # Get the sentiment data for the content.
    for sentiment_type in sentiment_types:
        # Get the content sentiment data that hasn't been saved yet.
        content_sentiment_data = await get_unprocessed_content_sentiment_data(
            from_date,
            to_date,
            sentiment_type,
        )

        # Get the sentiment data for the content.
        content_series = pd.Series(content_sentiment_data["content"].astype(str))
        sentiment_data = sentiment_type.get_sentiment_data(content_series)
        sentiment_data["provider_content_id"] = content_sentiment_data["id"]
        sentiment_data["sentiment_type_id"] = sentiment_type.get_sentiment_type_id()

        # Save the content sentiment data to the database.
        await set_data(
            ProviderContentSentiment.__tablename__,
            sentiment_data,
            operation_type="upsert",
        )


if __name__ == "__main__":
    refresh_content_sentiment_deployment = refresh_content_sentiment.to_deployment(
        name="refresh_content_sentiment_debug",
        concurrency_limit=1,
    )
    serve(refresh_content_sentiment_deployment)  # type: ignore
