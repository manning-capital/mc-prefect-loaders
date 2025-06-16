import requests
import pandas as pd
from prefect import flow, serve
from prefect.artifacts import create_table_artifact


@flow(log_prints=True)
async def pull_kraken_trade_book(pair: str, count: int = 500):
    
    # Get the trade book from Kraken.
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

    # Convert the trade book data to a pandas DataFrame.
    for pair, book in trade_book.items():
        trade_book_df = pd.DataFrame(
            book["asks"], columns=["price", "volume", "timestamp"]
        )
        trade_book_df["pair"] = pair
        trade_book_df["timestamp"] = pd.to_datetime(
            trade_book_df["timestamp"], unit="s"
        )
        trade_book_df["timestamp"] = trade_book_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        trade_book_df.sort_values(by="timestamp", inplace=True, ascending=False)
        await create_table_artifact(
            key=f"kraken-trade-book-{pair}".lower(),
            table=trade_book_df.to_dict(orient="records"),
            description=f"Trade book data for {pair} from Kraken.",
        )


if __name__ == "__main__":
    pull_kraken_trade_book_deployment = pull_kraken_trade_book.to_deployment(
        name="pull_kraken_trade_book",
        parameters={"pair": "XBTUSD", "count": 500},  # Default parameters
    )
    serve(pull_kraken_trade_book_deployment)
