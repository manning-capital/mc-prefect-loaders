import datetime
import os
import sys
from typing import List, Optional
import pandas as pd

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Extend MockDatabase with CRUD and query methods
class MockDatabase:
    """
    A mock database that simulates database operations for testing purposes.
    """

    __df: pd.DataFrame

    def __reset_df(self):
        """
        Reset the DataFrame to its initial state.
        """
        self.__df = pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "provider_id",
                "from_asset_id",
                "to_asset_id",
                "price",
                "volume",
            ]
        )
        self.__df["id"] = pd.Series([], dtype=int)
        self.__df["timestamp"] = pd.Series([], dtype="datetime64[ns, UTC]")
        self.__df["provider_id"] = pd.Series([], dtype=int)
        self.__df["from_asset_id"] = pd.Series([], dtype=int)
        self.__df["to_asset_id"] = pd.Series([], dtype=int)
        self.__df["price"] = pd.Series([], dtype=float)
        self.__df["volume"] = pd.Series([], dtype=float)

    def __init__(self):
        """
        Initialize the mock database with an empty DataFrame.
        """
        self.__reset_df()

    def add_provider_asset_order_data(
        self,
        timestamp: datetime.datetime,
        provider_id: int,
        from_asset_id: int,
        to_asset_id: int,
        price: Optional[float] = None,
        volume: Optional[float] = None,
    ):
        self.__df = pd.concat(
            [
                self.__df,
                pd.DataFrame(
                    {
                        "id": [self.__df["id"].max() + 1]
                        if not self.__df.empty
                        else [1],
                        "timestamp": [timestamp],
                        "provider_id": [provider_id],
                        "from_asset_id": [from_asset_id],
                        "to_asset_id": [to_asset_id],
                        "price": [price],
                        "volume": [volume],
                    }
                ),
            ],
            ignore_index=True,
        )

    def get_provider_asset_order_data(
        self,
        provider_ids: List[int] = [],
        from_asset_ids: List[int] = [],
        to_asset_ids: List[int] = [],
        start_datetime: Optional[datetime.datetime] = None,
        end_datetime: Optional[datetime.datetime] = None,
    ):
        output = self.__df

        # Filter by provider IDs
        if provider_ids:
            output = output[output["provider_id"].isin(provider_ids)]

        # Filter by from asset IDs
        if from_asset_ids:
            output = output[output["from_asset_id"].isin(from_asset_ids)]

        # Filter by to asset IDs
        if to_asset_ids:
            output = output[output["to_asset_id"].isin(to_asset_ids)]

        # Filter by start datetime
        if start_datetime:
            output = output[output["timestamp"] >= start_datetime]

        # Filter by end datetime
        if end_datetime:
            output = output[output["timestamp"] <= end_datetime]

        return output.reset_index(drop=True).drop(columns=["id"])

    def get_provider_asset_order_count(self) -> int:
        """
        Get the count of records in the mock database.
        """
        return len(self.__df)

    def clear_provider_asset_order_data(self):
        """
        Clear all data in the mock database.
        """
        self.__reset_df()
