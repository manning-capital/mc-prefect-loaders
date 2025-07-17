import requests
import datetime as dt
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.concurrency.asyncio import rate_limit
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from mc_postgres_db.models import Provider, ProviderType, ProviderContent    
from mc_postgres_db.prefect.asyncio.tasks import get_engine
from src.shared.utils import get_postgres_url, compare_dataframes, set_data


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
    new_provider_data["is_active"] = True
    new_provider_data["underlying_provider_id"] = coindesk_provider_id
    new_provider_data.rename(
        columns={
            "ID": "provider_external_code",
            "NAME": "name",
            "URL": "url",
            "IMAGE_URL": "image_url",
            "DESCRIPTION": "description",
            "STATUS": "is_active",
        },
        inplace=True,
    )
    new_provider_data["is_active"] = new_provider_data["is_active"].map(
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
                "description",
                "is_active",
                "underlying_provider_id",
            ]
        )
    ]  # type: ignore

    # Compare the provider data to the current provider data.
    dropped, added, _, different_records = compare_dataframes(
        old_provider_data, new_provider_data, ["provider_external_code"]
    )

    # Combine the dropped, added, and different records.
    combined_data = pd.concat([dropped, added, different_records])
    combined_data = combined_data.reset_index(drop=True)

    # Upsert the combined data to the database.
    await set_data(Provider.__tablename__, combined_data, operation_type="upsert")

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
    ids = raw_content_data["ID"].drop_duplicates().tolist()

    # Get the existing content data from the database.
    stmt = select(ProviderContent).where(ProviderContent.content_external_code.in_(ids))
    content_data = pd.read_sql(stmt, engine)

    return content_data

@task()
async def save_coindesk_news_content(
    content_type_id: int,
    news_provider_type_id: int,
    provider_data: pd.DataFrame,
    current_content_data: pd.DataFrame,
    raw_content_data: pd.DataFrame,
) -> pd.DataFrame:

    # Format the raw content data.
    new_content_data: pd.DataFrame = raw_content_data.copy(deep=True)  # type: ignore
    new_content_data["provider_external_code"] = new_content_data["SOURCE_ID"].astype(str)
    new_content_data["content_external_code"] = new_content_data["ID"].astype(str)
    new_content_data["content_type_id"] = content_type_id
    new_content_data["is_active"] = new_content_data["STATUS"].map(lambda x: True if x == "ACTIVE" else False)
    

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
        news_provider_type_id, coindesk_provider_id, current_provider_data, raw_provider_data
    )

    # Get content data
    raw_content_data = await get_coindesk_news_content()
    logger.info(f"Fetched {len(raw_content_data)} content items from Coindesk API.")

    # Get the existing content data from the database.
    current_content_data = await get_coindesk_news_content_from_database(raw_content_data)

    # Save content data to database
    provider_content_data = await save_coindesk_news_content(
        updated_provider_data, current_content_data, raw_content_data
    )

    # Log the number of content items saved.
    logger.info(f"Saved {len(provider_content_data)} content items to database.")
