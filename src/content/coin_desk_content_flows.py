import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import requests
import datetime as dt
import pandas as pd
from prefect import flow, get_run_logger, task, serve
from prefect.concurrency.asyncio import rate_limit
from sqlalchemy import select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Provider, ProviderType, ProviderContent, ContentType
from mc_postgres_db.prefect.asyncio.tasks import get_engine, set_data
from src.shared.utils import compare_dataframes


@task()
async def get_coindesk_news_providers() -> pd.DataFrame:
    host = "https://data-api.coindesk.com"
    uri = "/news/v1/source/list"
    params = {
        "lang": "EN",
        "status": "ACTIVE",
    }
    await rate_limit(names="coindesk-api", occupy=1, strict=True)
    response = requests.get(f"{host}{uri}", params=params)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data from Coindesk API: {response.status_code}"
        )
    data = response.json()["Data"]
    return pd.DataFrame(data)


@task()
async def get_coindesk_news_content() -> pd.DataFrame:
    host = "https://data-api.coindesk.com"
    uri = "/news/v1/article/list"
    params = {
        "lang": "EN",
        "limit": 100,
        "to_ts": (dt.datetime.now() - dt.timedelta(hours=2)).timestamp(),
    }
    await rate_limit(names="coindesk-api", occupy=1, strict=True)
    response = requests.get(f"{host}{uri}", params=params)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch data from Coindesk API: {response.status_code}"
        )
    data = response.json()["Data"]
    return pd.DataFrame(data)


@task()
async def get_coindesk_provider_id() -> int:
    engine = await get_engine()
    with Session(engine) as session:
        stmt = select(Provider.id).where(Provider.provider_external_code == "COINDESK")
        return session.execute(stmt).scalar_one()


@task()
async def get_news_provider_type_id() -> int:
    engine = await get_engine()
    with Session(engine) as session:
        stmt = select(ProviderType.id).where(ProviderType.name == "NEWS_PROVIDER")
        return session.execute(stmt).scalar_one()


@task()
async def get_provider_type_data() -> pd.DataFrame:
    engine = await get_engine()
    stmt = select(ProviderType)
    provider_type_data = pd.read_sql(stmt, engine)
    return provider_type_data


@task()
async def get_news_providers_from_database(
    news_provider_type_id: int,
    underlying_provider_id: int,
) -> pd.DataFrame:
    engine = await get_engine()
    stmt = select(Provider).where(
        Provider.underlying_provider_id == underlying_provider_id,
        Provider.provider_type_id == news_provider_type_id,
    )
    provider_data = pd.read_sql(stmt, engine)

    # Drop columns.
    provider_data = provider_data.drop(
        columns=["created_at", "updated_at", "description"]
    )

    # Format the provider data.
    provider_data["id"] = provider_data["id"].astype("Int64")
    provider_data["provider_external_code"] = provider_data[
        "provider_external_code"
    ].astype("str")
    provider_data["is_active"] = provider_data["is_active"].astype("bool")
    provider_data["underlying_provider_id"] = provider_data[
        "underlying_provider_id"
    ].astype("Int64")
    provider_data["provider_type_id"] = provider_data["provider_type_id"].astype(
        "Int64"
    )

    return provider_data


@task()
async def save_coindesk_news_providers(
    news_provider_type_id: int,
    coindesk_provider_id: int,
    old_provider_data: pd.DataFrame,
    new_raw_provider_data: pd.DataFrame,
) -> pd.DataFrame:
    # Format the raw provider data.
    new_provider_data: pd.DataFrame = new_raw_provider_data.copy(deep=True)  # type: ignore
    new_provider_data["provider_type_id"] = news_provider_type_id
    new_provider_data["underlying_provider_id"] = coindesk_provider_id
    new_provider_data.rename(
        columns={
            "ID": "provider_external_code",
            "NAME": "name",
            "URL": "url",
            "IMAGE_URL": "image_url",
        },
        inplace=True,
    )
    new_provider_data["is_active"] = new_provider_data["STATUS"].map(
        lambda x: True if x == "ACTIVE" else False
    )
    new_provider_data: pd.DataFrame = new_provider_data[
        pd.Index(
            [
                "provider_type_id",
                "provider_external_code",
                "name",
                "url",
                "image_url",
                "is_active",
                "underlying_provider_id",
            ]
        )
    ]  # type: ignore
    new_provider_data["provider_external_code"] = new_provider_data[
        "provider_external_code"
    ].astype("str")
    new_provider_data = new_provider_data.merge(
        old_provider_data[["id", "provider_external_code"]].drop_duplicates(),
        on="provider_external_code",
        how="left",
    )
    new_provider_data["id"] = new_provider_data["id"].astype("Int64")
    new_provider_data["is_active"] = new_provider_data["is_active"].astype("bool")
    new_provider_data["underlying_provider_id"] = new_provider_data[
        "underlying_provider_id"
    ].astype("Int64")
    new_provider_data["provider_type_id"] = new_provider_data[
        "provider_type_id"
    ].astype("Int64")

    # Compare the provider data to the current provider data.
    _, added, _, different_records = compare_dataframes(
        old_provider_data, new_provider_data, ["provider_external_code"]
    )

    # Add the new providers to the database.
    await set_data(
        Provider.__tablename__,
        added.drop(columns=["id"]),
        operation_type="upsert",
    )

    # Update the existing providers in the database.
    await set_data(
        Provider.__tablename__,
        different_records,
        operation_type="upsert",
    )

    # Fetch the updated provider data from the database.
    updated_provider_data = await get_news_providers_from_database(
        news_provider_type_id, coindesk_provider_id
    )

    return updated_provider_data


@task()
async def get_coindesk_news_content_from_database(
    raw_content_data: pd.DataFrame,
) -> pd.DataFrame:
    engine = await get_engine()

    # Extract the id(s) from the raw content data as this will be used to find existing data with the content external code column.
    ids = raw_content_data["ID"].apply(str).drop_duplicates().tolist()

    # Get the existing content data from the database.
    stmt = select(
        ProviderContent.id,
        ProviderContent.timestamp,
        ProviderContent.provider_id,
        ProviderContent.content_external_code,
        ProviderContent.content_type_id,
        ProviderContent.authors,
        ProviderContent.title,
        ProviderContent.content,
    ).where(ProviderContent.content_external_code.in_(ids))
    content_data = pd.read_sql(stmt, engine)

    # Format the content data.
    content_data["id"] = content_data["id"].astype("Int64")
    content_data["timestamp"] = content_data["timestamp"].astype("datetime64[ns]")
    content_data["provider_id"] = content_data["provider_id"].astype("Int64")
    content_data["content_external_code"] = content_data[
        "content_external_code"
    ].astype("str")
    content_data["content_type_id"] = content_data["content_type_id"].astype("Int64")
    content_data["authors"] = content_data["authors"].astype("str")
    content_data["title"] = content_data["title"].astype("str")
    content_data["content"] = content_data["content"].astype("str")

    return content_data


@task()
async def get_news_content_type_id() -> int:
    engine = await get_engine()
    with Session(engine) as session:
        stmt = select(ContentType.id).where(ContentType.name == "NEWS")
        return session.execute(stmt).scalar_one()


@task()
async def save_coindesk_news_content(
    news_content_type_id: int,
    provider_data: pd.DataFrame,
    current_content_data: pd.DataFrame,
    raw_content_data: pd.DataFrame,
) -> pd.DataFrame:
    # Get map of provider external code to provider id.
    provider_map = dict(
        zip(provider_data["provider_external_code"], provider_data["id"])
    )

    # Format the raw content data.
    new_content_data: pd.DataFrame = raw_content_data.copy(deep=True)  # type: ignore
    new_content_data["timestamp"] = new_content_data["PUBLISHED_ON"].apply(
        lambda x: dt.datetime.fromtimestamp(x)
    )
    new_content_data["provider_id"] = (
        new_content_data["SOURCE_ID"]
        .astype(str)
        .apply(lambda x: provider_map.get(x, None))
    )
    new_content_data["content_external_code"] = new_content_data["ID"].astype(str)
    new_content_data["content_type_id"] = news_content_type_id
    new_content_data["authors"] = new_content_data["AUTHORS"]
    new_content_data["title"] = new_content_data["TITLE"]
    new_content_data["content"] = new_content_data["BODY"]
    new_content_data = new_content_data[
        pd.Index(
            [
                "timestamp",
                "provider_id",
                "content_external_code",
                "content_type_id",
                "authors",
                "title",
                "content",
            ]
        )
    ]  # type: ignore
    new_content_data = new_content_data.merge(
        current_content_data[["id", "content_external_code"]].drop_duplicates(),
        on="content_external_code",
        how="left",
    )
    new_content_data["id"] = new_content_data["id"].astype("Int64")
    new_content_data["timestamp"] = new_content_data["timestamp"].astype(
        "datetime64[ns]"
    )
    new_content_data["provider_id"] = new_content_data["provider_id"].astype("Int64")
    new_content_data["content_external_code"] = new_content_data[
        "content_external_code"
    ].astype("str")
    new_content_data["content_type_id"] = new_content_data["content_type_id"].astype(
        "Int64"
    )
    new_content_data["authors"] = new_content_data["authors"].astype("str")

    # Compare the content data to the current content data.
    _, added, _, different_records = compare_dataframes(
        current_content_data, new_content_data, ["content_external_code"]
    )

    # Add the new content to the database.
    await set_data(
        ProviderContent.__tablename__,
        added.drop(columns=["id"]),
        operation_type="upsert",
    )

    # Update the existing content in the database.
    await set_data(
        ProviderContent.__tablename__, different_records, operation_type="upsert"
    )

    # Fetch the updated content data from the database.
    updated_content_data = await get_coindesk_news_content_from_database(
        raw_content_data
    )

    return updated_content_data


@flow()
async def pull_coindesk_news_content():
    logger = get_run_logger()

    # Get the coindesk provider id.
    coindesk_provider_id = await get_coindesk_provider_id()

    # Get the new provider type id.
    news_provider_type_id = await get_news_provider_type_id()

    # Get current news providers from the database.
    current_provider_data = await get_news_providers_from_database(
        news_provider_type_id, coindesk_provider_id
    )

    # Get provider data first, we have to do this first because the content data depends on the provider data.
    raw_provider_data = await get_coindesk_news_providers()
    logger.info(f"Fetched {len(raw_provider_data)} providers from Coindesk API.")

    # Save provider data to database.
    updated_provider_data = await save_coindesk_news_providers(
        news_provider_type_id,
        coindesk_provider_id,
        current_provider_data,
        raw_provider_data,
    )

    # Get content data
    raw_content_data = await get_coindesk_news_content()
    logger.info(f"Fetched {len(raw_content_data)} content items from Coindesk API.")

    # Get the existing content data from the database.
    current_content_data = await get_coindesk_news_content_from_database(
        raw_content_data
    )

    # Get the content type id for news content.
    news_content_type_id = await get_news_content_type_id()

    # Save content data to database
    await save_coindesk_news_content(
        news_content_type_id,
        updated_provider_data,
        current_content_data,
        raw_content_data,
    )


if __name__ == "__main__":
    pull_coindesk_news_content_deployment = pull_coindesk_news_content.to_deployment(
        name="pull_coindesk_news_content_debug",
        concurrency_limit=1,
    )
    serve(pull_coindesk_news_content_deployment)  # type: ignore
