import os
import sys
import datetime as dt
import time

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import pytest
import pandas as pd
from unittest.mock import patch
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from prefect import task
from mc_postgres_db.models import ProviderType, ContentType, Provider, ProviderContent
from mc_postgres_db.prefect.asyncio.tasks import get_engine
from mc_postgres_db.testing.utilities import clear_database
from src.content.coin_desk_content_flows import pull_coindesk_news_content


def setup_base_data(engine: Engine):
    with Session(engine) as session:
        # Add the provider type.
        provider_type = ProviderType(name="NEWS_PROVIDER", is_active=True)
        session.add(provider_type)
        session.commit()
        provider_type_id_stmt = select(ProviderType.id).where(
            ProviderType.name == "NEWS_PROVIDER"
        )
        provider_type_id = session.execute(provider_type_id_stmt).scalar_one()

        # Add the content type.
        content_type = ContentType(name="NEWS", is_active=True)
        session.add(content_type)
        session.commit()
        content_type_id_stmt = select(ContentType.id).where(ContentType.name == "NEWS")
        content_type_id = session.execute(content_type_id_stmt).scalar_one()

        # Add the base coindesk provider.
        provider = Provider(
            name="CoinDesk",
            is_active=True,
            provider_type_id=provider_type.id,
            provider_external_code="COINDESK",
        )
        session.add(provider)
        session.commit()
        provider_id_stmt = select(Provider.id).where(
            Provider.provider_external_code == "COINDESK"
        )
        provider_id = session.execute(provider_id_stmt).scalar_one()

        return provider_type_id, content_type_id, provider_id


@pytest.mark.asyncio
async def test_empty_database_and_one_new_provider_and_content():
    # Clear the database.
    engine = await get_engine()
    clear_database(engine)

    # Setup the base data.
    _, _, provider_id = setup_base_data(engine)

    # Setup the mock news providers function.
    @task()
    async def mock_get_coindesk_news_providers():
        with open("tests/data/coindesk/all_sources.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([82])]
            return df

    # Setup the mock news content function.
    @task()
    async def mock_get_coindesk_news_content():
        with open("tests/data/coindesk/all_articles.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([48431841])]
            return df

    with (
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_content",
            mock_get_coindesk_news_content,
        ),
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_providers",
            mock_get_coindesk_news_providers,
        ),
    ):
        # Run the content flow.
        await pull_coindesk_news_content()

        # Check the provider data.
        provider_stmt = select(Provider).where(
            Provider.underlying_provider_id == provider_id
        )
        provider_df = pd.read_sql(provider_stmt, engine)
        assert len(provider_df) == 1
        assert provider_df.iloc[0]["name"] == "Bitcoin World"
        assert provider_df.iloc[0]["provider_external_code"] == "82"
        bitcoin_world_provider_id = int(provider_df.iloc[0]["id"])

        # Check the content data.
        content_stmt = select(ProviderContent).where(
            ProviderContent.provider_id == bitcoin_world_provider_id
        )
        content_df = pd.read_sql(content_stmt, engine)
        assert len(content_df) == 1
        assert content_df.iloc[0]["content_external_code"] == "48431841"
        assert content_df.iloc[0]["provider_id"] == bitcoin_world_provider_id


@pytest.mark.asyncio
async def test_empty_database_with_multiple_new_providers_and_content():
    # Clear the database.
    engine = await get_engine()
    clear_database(engine)

    # Setup the base data.
    _, _, provider_id = setup_base_data(engine)

    # Setup the mock news providers function.
    @task()
    async def mock_get_coindesk_news_providers():
        with open("tests/data/coindesk/all_sources.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([82, 93])]
            return df

    # Setup the mock news content function.
    @task()
    async def mock_get_coindesk_news_content():
        with open("tests/data/coindesk/all_articles.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([48431693, 48430510, 48431841])]
            return df

    with (
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_content",
            mock_get_coindesk_news_content,
        ),
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_providers",
            mock_get_coindesk_news_providers,
        ),
    ):
        # Run the content flow.
        await pull_coindesk_news_content()

        # Check the provider data.
        provider_stmt = select(Provider).where(
            Provider.underlying_provider_id == provider_id
        )
        provider_df = pd.read_sql(provider_stmt, engine)
        assert len(provider_df) == 2
        assert provider_df.iloc[0]["name"] == "Bitcoin World"
        assert provider_df.iloc[0]["provider_external_code"] == "82"
        bitcoin_world_provider_id = int(provider_df.iloc[0]["id"])
        assert provider_df.iloc[1]["name"] == "Coinpaper"
        assert provider_df.iloc[1]["provider_external_code"] == "93"
        coinpaper_provider_id = int(provider_df.iloc[1]["id"])

        # Check the content data.
        content_stmt_1 = select(ProviderContent).where(
            ProviderContent.provider_id == bitcoin_world_provider_id
        )
        content_stmt_2 = select(ProviderContent).where(
            ProviderContent.provider_id == coinpaper_provider_id
        )
        content_df_1 = pd.read_sql(content_stmt_1, engine)
        content_df_2 = pd.read_sql(content_stmt_2, engine)
        assert len(content_df_1) == 1
        assert len(content_df_2) == 2
        assert set(content_df_1["content_external_code"]) == {"48431841"}
        assert set(content_df_2["content_external_code"]) == {
            "48431693",
            "48430510",
        }


@pytest.mark.asyncio
async def test_one_existing_provider_and_new_content():
    # Clear the database.
    engine = await get_engine()
    clear_database(engine)

    # Setup the base data.
    provider_type_id, _, provider_id = setup_base_data(engine)

    # Setup the mock news providers function.
    @task()
    async def mock_get_coindesk_news_providers():
        with open("tests/data/coindesk/all_sources.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([82])]
            return df

    # Setup the mock news content function.
    @task()
    async def mock_get_coindesk_news_content():
        with open("tests/data/coindesk/all_articles.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([48431841, 48429722, 48429207])]
            return df

    with (
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_content",
            mock_get_coindesk_news_content,
        ),
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_providers",
            mock_get_coindesk_news_providers,
        ),
    ):
        # Create the provider first.
        with Session(engine) as session:
            session.add(
                Provider(
                    name="Bitcoin World",
                    is_active=True,
                    provider_type_id=provider_type_id,
                    provider_external_code="82",
                    url="https://bitcoinworld.co.in/feed/",
                    image_url="https://resources.cryptocompare.com/news/82/default.png",
                    underlying_provider_id=provider_id,
                )
            )
            session.commit()

        # Record the timestamp before the content flow.
        timestamp = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        time.sleep(1.0)

        # Run the content flow.
        await pull_coindesk_news_content()

        # Check the provider data.
        provider_stmt = select(Provider).where(
            Provider.underlying_provider_id == provider_id
        )
        provider_df = pd.read_sql(provider_stmt, engine)
        assert len(provider_df) == 1
        assert provider_df.iloc[0]["name"] == "Bitcoin World"
        assert provider_df.iloc[0]["provider_external_code"] == "82"
        assert provider_df.iloc[0]["updated_at"] <= timestamp
        assert provider_df.iloc[0]["created_at"] <= timestamp
        bitcoin_world_provider_id = int(provider_df.iloc[0]["id"])

        # Check the content data.
        content_stmt = select(ProviderContent).where(
            ProviderContent.provider_id == bitcoin_world_provider_id
        )
        content_df = pd.read_sql(content_stmt, engine)
        assert len(content_df) == 3
        assert set(content_df["content_external_code"]) == {
            "48431841",
            "48429722",
            "48429207",
        }


@pytest.mark.asyncio
async def test_multiple_existing_and_new_providers():
    # Clear the database.
    engine = await get_engine()
    clear_database(engine)

    # Setup the base data.
    provider_type_id, _, provider_id = setup_base_data(engine)

    # Setup the mock news providers function.
    @task()
    async def mock_get_coindesk_news_providers():
        with open("tests/data/coindesk/all_sources.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([82, 93, 29, 79])]
            return df

    # Setup the mock news content function.
    @task()
    async def mock_get_coindesk_news_content():
        with open("tests/data/coindesk/all_articles.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([48431841, 48429722, 48429207])]
            return df

    with (
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_content",
            mock_get_coindesk_news_content,
        ),
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_providers",
            mock_get_coindesk_news_providers,
        ),
    ):
        # Create the providers first.
        with Session(engine) as session:
            session.add(
                Provider(
                    name="Bitcoin World",
                    is_active=True,
                    provider_type_id=provider_type_id,
                    provider_external_code="82",
                    url="https://bitcoinworld.co.in/feed/",
                    image_url="https://resources.cryptocompare.com/news/82/default.png",
                    underlying_provider_id=provider_id,
                )
            )
            session.add(
                Provider(
                    name="Coinpaper",
                    is_active=True,
                    provider_type_id=provider_type_id,
                    provider_external_code="93",
                    url="https://coinpaper.com/newsfeed",
                    image_url="https://resources.cryptocompare.com/news/93/default.png",
                    underlying_provider_id=provider_id,
                )
            )
            session.commit()

        # Record the timestamp before the content flow.
        timestamp = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        time.sleep(1.0)

        # Run the content flow.
        await pull_coindesk_news_content()

        # Check the provider data.
        provider_stmt = select(Provider).where(
            Provider.underlying_provider_id == provider_id
        )
        provider_df = pd.read_sql(provider_stmt, engine)
        existing_providers = provider_df.loc[provider_df["created_at"] <= timestamp]
        new_providers = provider_df.loc[provider_df["created_at"] > timestamp]
        assert len(existing_providers) == 2
        assert set(existing_providers["provider_external_code"]) == {"82", "93"}
        assert len(new_providers) == 2
        assert set(new_providers["provider_external_code"]) == {"29", "79"}

@pytest.mark.asyncio
async def test_empty_content_in_database_and_new_content():
    # Clear the database.
    engine = await get_engine()
    clear_database(engine)

    # Setup the base data.
    provider_type_id, _, provider_id = setup_base_data(engine)

    # Setup the mock news providers function.
    @task()
    async def mock_get_coindesk_news_providers():
        with open("tests/data/coindesk/all_sources.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([82])]
            return df
            
    # Setup the mock news content function.
    @task()
    async def mock_get_coindesk_news_content():
        with open("tests/data/coindesk/all_articles.json", "r") as f:
            df = pd.read_json(f)
            df = df.loc[df["ID"].isin([48431841, 48429722, 48429207])]
            return df

    with (
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_content",
            mock_get_coindesk_news_content,
        ),
        patch(
            "src.content.coin_desk_content_flows.get_coindesk_news_providers",
            mock_get_coindesk_news_providers,
        ),
    ):
        # Ensure that the provider is created.
        with Session(engine) as session:
            session.add(
                Provider(
                    name="Bitcoin World",
                    is_active=True,
                    provider_type_id=provider_type_id,
                    provider_external_code="82",
                    url="https://bitcoinworld.co.in/feed/",
                    image_url="https://resources.cryptocompare.com/news/82/default.png",
                    underlying_provider_id=provider_id,
                )
            )
            session.commit()
            bitcoin_world_provider_id = session.execute(select(Provider.id).where(Provider.provider_external_code == "82")).scalar_one()

        # Run the content flow.
        await pull_coindesk_news_content()

        # Check the content data.
        content_stmt = select(ProviderContent).where(
            ProviderContent.provider_id == bitcoin_world_provider_id
        )
        content_df = pd.read_sql(content_stmt, engine)
        assert len(content_df) == 3
        assert set(content_df["content_external_code"]) == {
            "48431841",
            "48429722",
            "48429207",
        }
