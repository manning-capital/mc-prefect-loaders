import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import datetime as dt

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import (
    Provider,
    ContentType,
    ProviderType,
    SentimentType,
    ProviderContent,
    ProviderContentSentiment,
)
from mc_postgres_db.prefect.asyncio.tasks import get_engine

from src.content.sentiment.content_sentiment_flows import refresh_content_sentiment


async def create_base_data(
    engine: Engine,
) -> tuple[ProviderType, Provider, ContentType, SentimentType]:
    # Create the provider type.
    with Session(engine) as session:
        provider_type = ProviderType(
            name="NEWS_PROVIDER", description="News provider", is_active=True
        )
        session.add(provider_type)
        session.commit()

        # Create the provider.
        provider = Provider(
            name="Coindesk", provider_type_id=provider_type.id, is_active=True
        )
        session.add(provider)
        session.commit()

        # Create the content type.
        content_type = ContentType(name="NEWS", description="News", is_active=True)
        session.add(content_type)
        session.commit()

        # Create the sentiment type.
        sentiment_type = SentimentType(
            name="NLTKVader", description="NLTKVader", is_active=True
        )
        session.add(sentiment_type)
        session.commit()

        # Get the ORM objects for all of the above.
        provider_type = session.execute(
            select(ProviderType).where(ProviderType.name == "NEWS_PROVIDER")
        ).scalar_one()
        provider = session.execute(
            select(Provider).where(Provider.name == "Coindesk")
        ).scalar_one()
        content_type = session.execute(
            select(ContentType).where(ContentType.name == "NEWS")
        ).scalar_one()
        sentiment_type = session.execute(
            select(SentimentType).where(SentimentType.name == "NLTKVader")
        ).scalar_one()

        return provider_type, provider, content_type, sentiment_type


@pytest.mark.asyncio
async def test_sentiment_analysis_added_for_existing_content_without_sentiment():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add some fake provider content data.
    with Session(engine) as session:
        provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(provider_content)
        session.commit()
        session.refresh(provider_content)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id == provider_content.id
            )
        ).scalar_one()
        assert content_sentiment_data is not None
        assert content_sentiment_data.sentiment_type_id == sentiment_type.id
        assert content_sentiment_data.sentiment_score is not None
        assert content_sentiment_data.positive_sentiment_score is not None
        assert content_sentiment_data.negative_sentiment_score is not None
        assert content_sentiment_data.neutral_sentiment_score is not None


@pytest.mark.asyncio
async def test_sentiment_analysis_added_for_multiple_existing_content_without_sentiment():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add positive content.
    with Session(engine) as session:
        positive_provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Very positive content",
            authors="Preppy Pete",
            content="This is content that is meant to be very positive. It is a test content and is really great!",
        )
        session.add(positive_provider_content)
        session.commit()
        session.refresh(positive_provider_content)

    # Add negative content.
    with Session(engine) as session:
        negative_provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567891",
            content_type_id=content_type.id,
            title="Very negative content",
            authors="Grumpy George",
            content="This is content that is meant to be very negative. It is a test content and is really bad!",
        )
        session.add(negative_provider_content)
        session.commit()
        session.refresh(negative_provider_content)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        # Check the positive content sentiment data.
        positive_content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == positive_provider_content.id
            )
        ).scalar_one()
        assert positive_content_sentiment_data is not None
        assert positive_content_sentiment_data.sentiment_type_id == sentiment_type.id
        assert positive_content_sentiment_data.sentiment_score is not None
        assert positive_content_sentiment_data.positive_sentiment_score is not None
        assert positive_content_sentiment_data.negative_sentiment_score is not None
        assert positive_content_sentiment_data.neutral_sentiment_score is not None

        # Check the negative content sentiment data.
        negative_content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == negative_provider_content.id
            )
        ).scalar_one()
        assert negative_content_sentiment_data is not None
        assert negative_content_sentiment_data.sentiment_type_id == sentiment_type.id
        assert negative_content_sentiment_data.sentiment_score is not None
        assert negative_content_sentiment_data.positive_sentiment_score is not None
        assert negative_content_sentiment_data.negative_sentiment_score is not None
        assert negative_content_sentiment_data.neutral_sentiment_score is not None


@pytest.mark.asyncio
async def test_sentiment_analysis_is_positive_for_positive_content():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add positive content.
    with Session(engine) as session:
        positive_provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Very positive content",
            authors="Preppy Pete",
            content="This is content that is meant to be very positive. It is a test content and is really great!",
        )
        session.add(positive_provider_content)
        session.commit()
        session.refresh(positive_provider_content)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == positive_provider_content.id
            )
        ).scalar_one()
        assert content_sentiment_data is not None
        assert content_sentiment_data.sentiment_score is not None
        assert content_sentiment_data.positive_sentiment_score is not None
        assert content_sentiment_data.sentiment_score > 0.0
        assert content_sentiment_data.positive_sentiment_score > 0.0


@pytest.mark.asyncio
async def test_sentiment_analysis_is_negative_for_negative_content():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add negative content.
    with Session(engine) as session:
        negative_provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567891",
            content_type_id=content_type.id,
            title="Very negative content",
            authors="Grumpy George",
            content="This is content that is meant to be very negative. It is a test content and is really bad!",
        )
        session.add(negative_provider_content)
        session.commit()
        session.refresh(negative_provider_content)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == negative_provider_content.id
            )
        ).scalar_one()
        assert content_sentiment_data is not None
        assert content_sentiment_data.sentiment_score is not None
        assert content_sentiment_data.negative_sentiment_score is not None
        assert content_sentiment_data.sentiment_score < 0.0
        assert content_sentiment_data.negative_sentiment_score > 0.0


@pytest.mark.asyncio
async def test_sentiment_analysis_not_added_for_existing_content_with_sentiment():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add some provider content data.
    with Session(engine) as session:
        provider_content = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(provider_content)
        session.commit()
        session.refresh(provider_content)

    # Add sentiment data for the content.
    with Session(engine) as session:
        provider_content_sentiment = ProviderContentSentiment(
            provider_content_id=provider_content.id,
            sentiment_type_id=sentiment_type.id,
            sentiment_score=0.5,
            positive_sentiment_score=0.5,
            negative_sentiment_score=0.5,
            neutral_sentiment_score=0.5,
        )
        session.add(provider_content_sentiment)
        session.commit()
        session.refresh(provider_content_sentiment)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data is the same as the original.
    with Session(engine) as session:
        content_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id == provider_content.id
            )
        ).scalar_one()
        assert (
            content_sentiment_data.provider_content_id
            == provider_content_sentiment.provider_content_id
        )
        assert (
            content_sentiment_data.sentiment_type_id
            == provider_content_sentiment.sentiment_type_id
        )
        assert (
            content_sentiment_data.sentiment_score
            == provider_content_sentiment.sentiment_score
        )
        assert (
            content_sentiment_data.positive_sentiment_score
            == provider_content_sentiment.positive_sentiment_score
        )
        assert (
            content_sentiment_data.negative_sentiment_score
            == provider_content_sentiment.negative_sentiment_score
        )
        assert (
            content_sentiment_data.neutral_sentiment_score
            == provider_content_sentiment.neutral_sentiment_score
        )


@pytest.mark.asyncio
async def test_sentiment_analysis_handles_existing_and_new_sentiment_entries_correctly():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Add some provider content data.
    with Session(engine) as session:
        content_with_sentiment = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(content_with_sentiment)
        session.commit()
        session.refresh(content_with_sentiment)

    # Add some provider content data without sentiment.
    with Session(engine) as session:
        content_without_sentiment_1 = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567891",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(content_without_sentiment_1)
        session.commit()
        session.refresh(content_without_sentiment_1)

    # Add some provider content data without sentiment.
    with Session(engine) as session:
        content_without_sentiment_2 = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567892",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(content_without_sentiment_2)
        session.commit()
        session.refresh(content_without_sentiment_2)

    # Add sentiment data for the content.
    with Session(engine) as session:
        content_with_sentiment_sentiment = ProviderContentSentiment(
            provider_content_id=content_with_sentiment.id,
            sentiment_type_id=sentiment_type.id,
            sentiment_score=0.5,
            positive_sentiment_score=0.5,
            negative_sentiment_score=0.5,
            neutral_sentiment_score=0.5,
        )
        session.add(content_with_sentiment_sentiment)
        session.commit()
        session.refresh(content_with_sentiment_sentiment)

    # Record the current time to use in the comparison after the refresh.
    current_time = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        # Check the content with sentiment data.
        content_with_sentiment_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == content_with_sentiment.id
            )
        ).scalar_one()
        assert content_with_sentiment_sentiment_data is not None
        assert (
            content_with_sentiment_sentiment_data.sentiment_type_id
            == content_with_sentiment_sentiment.sentiment_type_id
        )
        assert (
            content_with_sentiment_sentiment_data.sentiment_score
            == content_with_sentiment_sentiment.sentiment_score
        )
        assert (
            content_with_sentiment_sentiment_data.positive_sentiment_score
            == content_with_sentiment_sentiment.positive_sentiment_score
        )
        assert (
            content_with_sentiment_sentiment_data.negative_sentiment_score
            == content_with_sentiment_sentiment.negative_sentiment_score
        )
        assert (
            content_with_sentiment_sentiment_data.neutral_sentiment_score
            == content_with_sentiment_sentiment.neutral_sentiment_score
        )
        assert content_with_sentiment_sentiment_data.created_at < current_time

        # Check the content without sentiment data.
        content_without_sentiment_1_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == content_without_sentiment_1.id
            )
        ).scalar_one()
        assert content_without_sentiment_1_sentiment_data is not None
        assert (
            content_without_sentiment_1_sentiment_data.sentiment_type_id
            == content_with_sentiment_sentiment.sentiment_type_id
        )
        assert content_without_sentiment_1_sentiment_data.sentiment_score is not None
        assert (
            content_without_sentiment_1_sentiment_data.positive_sentiment_score
            is not None
        )
        assert (
            content_without_sentiment_1_sentiment_data.negative_sentiment_score
            is not None
        )
        assert (
            content_without_sentiment_1_sentiment_data.neutral_sentiment_score
            is not None
        )
        assert content_without_sentiment_1_sentiment_data.created_at > current_time

        # Check the content without sentiment data.
        content_without_sentiment_2_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == content_without_sentiment_2.id
            )
        ).scalar_one()
        assert content_without_sentiment_2_sentiment_data is not None
        assert (
            content_without_sentiment_2_sentiment_data.sentiment_type_id
            == content_with_sentiment_sentiment.sentiment_type_id
        )
        assert content_without_sentiment_2_sentiment_data.sentiment_score is not None
        assert (
            content_without_sentiment_2_sentiment_data.positive_sentiment_score
            is not None
        )
        assert (
            content_without_sentiment_2_sentiment_data.negative_sentiment_score
            is not None
        )
        assert (
            content_without_sentiment_2_sentiment_data.neutral_sentiment_score
            is not None
        )
        assert content_without_sentiment_2_sentiment_data.created_at > current_time


@pytest.mark.asyncio
async def test_sentiment_analysis_handles_existing_partially_filled_sentiment_entries_correctly():
    # Get the engine.
    engine = await get_engine()

    # Create the base data.
    _, provider, content_type, sentiment_type = await create_base_data(engine)

    # Create provider content.
    with Session(engine) as session:
        content_with_sentiment = ProviderContent(
            timestamp=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            provider_id=provider.id,
            content_external_code="1234567890",
            content_type_id=content_type.id,
            title="Test title",
            authors="Test author",
            content="This is a test content",
        )
        session.add(content_with_sentiment)
        session.commit()
        session.refresh(content_with_sentiment)

    # Create provider content with sentiment data, but only some of the columns are filled.
    with Session(engine) as session:
        content_with_sentiment_partial = ProviderContentSentiment(
            provider_content_id=content_with_sentiment.id,
            sentiment_type_id=sentiment_type.id,
            sentiment_score=0.5,
            positive_sentiment_score=None,
            negative_sentiment_score=None,
            neutral_sentiment_score=None,
        )
        session.add(content_with_sentiment_partial)
        session.commit()
        session.refresh(content_with_sentiment_partial)

    # Run the refresh content sentiment flow.
    await refresh_content_sentiment(from_date=dt.date.today(), to_date=dt.date.today())

    # Check the content sentiment data.
    with Session(engine) as session:
        content_with_sentiment_partial_sentiment_data = session.execute(
            select(ProviderContentSentiment).where(
                ProviderContentSentiment.provider_content_id
                == content_with_sentiment.id
            )
        ).scalar_one()
        assert content_with_sentiment_partial_sentiment_data is not None
        assert content_with_sentiment_partial_sentiment_data.sentiment_score is not None
        assert (
            content_with_sentiment_partial_sentiment_data.positive_sentiment_score
            is not None
        )
        assert (
            content_with_sentiment_partial_sentiment_data.negative_sentiment_score
            is not None
        )
        assert (
            content_with_sentiment_partial_sentiment_data.neutral_sentiment_score
            is not None
        )
