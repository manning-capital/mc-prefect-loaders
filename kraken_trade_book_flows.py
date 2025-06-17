import requests
import pandas as pd
from prefect.blocks.system import Secret
from prefect import flow, serve, task, get_run_logger
from prefect.concurrency.sync import rate_limit
from sqlalchemy import create_engine, text
from datetime import datetime


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


@task(log_prints=True)
async def get_kraken_order_book(pair: str, count: int = 500) -> dict[str, object]:
    logger = get_run_logger()

    # Rate limit the task to avoid hitting Kraken's API too frequently.
    logger.info("Applying rate limit to Kraken API requests.")
    rate_limit("kraken-api", strict=True, timeout_seconds=60)

    # Fetch the trade book from Kraken's public API.
    logger.info(f"Fetching trade book for {pair} with count {count} from Kraken.")
    response = requests.get(
        f"https://api.kraken.com/0/public/Depth?pair={pair}&count={count}"
    )
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Kraken: {response.status_code}")
    data: dict = response.json()

    # Extract the trade book data.
    trade_book: dict[str, object] = data.get("result")
    if not trade_book:
        raise Exception("No trade book data found in the response.")
    return trade_book


@task()
async def get_max_datetime_from_order_book_postgres() -> datetime:
    """
    Get the maximum timestamp from the PostgreSQL trade book table.
    """
    query = "SELECT MAX(timestamp) FROM kraken_trade_book"
    url = await get_postgres_url()
    engine = create_engine(url)
    with engine.connect() as conn:
        max_datetime_str: str = conn.execute(text(query)).scalar()
        if max_datetime_str is None:
            return datetime.min
        return pd.to_datetime(max_datetime_str).to_pydatetime()


@task()
async def save_trade_book_data_to_postgres(trade_books: dict[str, object]):
    logger = get_run_logger()

    # Define the PostgreSQL connection parameters.
    url = await get_postgres_url()
    engine = create_engine(url)

    # Convert the trade book data to a pandas DataFrame.
    trade_book_df = pd.DataFrame()
    for pair, book in trade_books.items():
        logger.info(f"Processing trade book for {pair}.")
        pair_df = pd.DataFrame(book["asks"], columns=["price", "volume", "timestamp"])
        pair_df["pair"] = pair
        pair_df["timestamp"] = pd.to_datetime(pair_df["timestamp"], unit="s")
        trade_book_df = pd.concat([trade_book_df, pair_df], ignore_index=True)

    # Sort the DataFrame by pair and timestamp.
    trade_book_df.sort_values(by=["pair", "timestamp"], inplace=True, ascending=False)

    # Get max timestamp from postgres to avoid duplicates.
    max_timestamp = await get_max_datetime_from_order_book_postgres()
    logger.info(f"Maximum timestamp in PostgreSQL: {max_timestamp}")

    # Filter out rows with timestamps less than the maximum timestamp.
    trade_book_df = trade_book_df[trade_book_df["timestamp"] > max_timestamp]
    if trade_book_df.empty:
        logger.info("No new trade book data to save. Exiting.")
        return
    logger.info(f"Number of new rows to save: {len(trade_book_df)}")

    # Add new rows to the PostgreSQL database.
    logger.info("Connecting to PostgreSQL database to save trade book data...")
    with engine.connect() as conn:
        logger.info("Connection established successfully.")
        trade_book_df.to_sql(
            "kraken_trade_book",
            conn,
            if_exists="append",
            index=False,
        )


@flow(log_prints=True)
async def pull_kraken_trade_book(pairs: list[str], count: int = 500):
    logger = get_run_logger()

    # Fetch the trade book data from Kraken.
    logger.info(
        f"Starting to pull trade book data for pairs: {pairs} with count {count}."
    )
    trade_books: dict[str, object] = {}
    for pair in pairs:
        logger.info(f"Fetching trade book for {pair}.")
        trade_books.update(await get_kraken_order_book(pair, count))
    logger.info("All trade books fetched successfully.")

    # Get maximum timestamp from the trade book table.

    # Save the trade book data to PostgreSQL.
    logger.info("Saving trade book data to PostgreSQL.")
    await save_trade_book_data_to_postgres(trade_books)


if __name__ == "__main__":
    pull_kraken_trade_book_deployment = pull_kraken_trade_book.to_deployment(
        name="pull_kraken_trade_book_debug",
        parameters={"pairs": ["XBTUSD", "XBTEUR"], "count": 500},  # Default parameters
    )
    serve(pull_kraken_trade_book_deployment)
