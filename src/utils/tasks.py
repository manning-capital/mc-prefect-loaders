import os
from prefect import task
from prefect.blocks.system import Secret


@task()
async def get_api_url() -> str:
    """
    Get the base URL for the Kraken API.
    """
    return os.environ["PREFECT_API_URL"]


@task()
async def get_base_url() -> str:
    """
    Get the base URL for the Kraken API.
    """
    api_url: str = await get_api_url()
    if not api_url:
        raise ValueError("PREFECT_API_URL environment variable is not set.")
    if api_url.endswith("/api"):
        api_url = api_url[:-4]  # Remove the "/api" suffix
    return api_url


@task()
async def get_postgres_url() -> str:
    """
    Get the PostgreSQL connection string from a secret block.
    """
    postgresql_password: str = (await Secret.load("postgresql-password")).get()
    host = "db-postgresql-lon1-65351-do-user-18535103-0.m.db.ondigitalocean.com"
    port = 25060
    database = "defaultdb"
    user = "doadmin"
    url = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=user,
        password=postgresql_password,
        host=host,
        port=port,
        database=database,
    )
    return url
