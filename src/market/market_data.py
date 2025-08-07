import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src.market.abstract import AbstractProviderAssetMarketData
from sqlalchemy.orm import Session
from sqlalchemy import select
from mc_postgres_db.models import Provider
import pandas as pd
from typing import Optional, Any
import datetime as dt
import requests


class KrakenProviderAssetMarketData(AbstractProviderAssetMarketData):
    def get_provider(self) -> Provider:
        with Session(self.engine) as session:
            stmt = select(Provider).where(Provider.name == "Kraken")
            return session.execute(stmt).scalar_one()

    @property
    def columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def rate_limit_name(self) -> str:
        return "kraken-api"

    async def request_kraken(
        self,
        method: str,
        url: str,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make a request to Kraken.
        """
        # Make the request.
        response = requests.request(method=method, url=url, params=params, json=json)

        # Check the response for an error.
        if not response.ok:
            raise Exception(f"Failed to request {url}: {response.json()['error']}")

        # Check the response for an error.
        error: list[str] = response.json().get("error", [])
        if len(error) > 0:
            raise Exception(f"The request to {url} the following errors: {error}")

        # Return the response.
        return response.json()

    async def request_asset_pairs(self) -> dict[str, dict[str, Any]]:
        url = "https://api.kraken.com/0/public/AssetPairs"
        return await self.request_kraken(method="GET", url=url)

    async def get_asset_pairs(
        self, asset_codes: pd.Series, as_of_date: Optional[dt.date] = None
    ) -> pd.DataFrame:
        # Get the asset pairs from Kraken.
        response_data = await self.request_asset_pairs()
        result: dict[str, dict[str, Any]] = response_data["result"]

        # Format the raw asset pairs result into a pandas dataframe.
        asset_pairs = pd.DataFrame(
            [
                {
                    "pair_asset_code": asset_pair,
                    "from_asset_code": asset_data["quote"],
                    "to_asset_code": asset_data["base"],
                }
                for asset_pair, asset_data in result.items()
            ]
        )

        # Filter the asset pairs to only include the asset codes that are in the asset_codes series.
        asset_pairs = asset_pairs.loc[
            (asset_pairs["from_asset_code"].isin(asset_codes))
            & (asset_pairs["to_asset_code"].isin(asset_codes))
        ]

        return asset_pairs.drop_duplicates().reset_index(drop=True)

    async def request_market_data(self, pair_asset_code: str) -> dict[str, Any]:
        url = "https://api.kraken.com/0/public/OHLC"
        return await self.request_kraken(
            method="GET", url=url, params={"pair": pair_asset_code}
        )

    async def get_market_data(
        self,
        pair_asset_code: str,
        from_asset_code: str,
        to_asset_code: str,
        as_of_date: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        # Request the OHLCV data for the asset pairs.
        response_data = await self.request_market_data(pair_asset_code=pair_asset_code)
        result: dict[str, list[list[Any]]] = response_data["result"]

        # Get the OHLCV data. Will be of the form: [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
        raw_data: list[list[Any]] = result[pair_asset_code]

        # Convert the result to a dataframe.
        data = pd.DataFrame(
            raw_data,
            columns=pd.Index(
                ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
            ),
        )

        # Ensure the data has the correct types.
        data["from_asset_code"] = from_asset_code
        data["to_asset_code"] = to_asset_code
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
        data["open"] = data["open"].astype(float)
        data["high"] = data["high"].astype(float)
        data["low"] = data["low"].astype(float)
        data["close"] = data["close"].astype(float)
        data["volume"] = data["volume"].astype(float)

        # Drop the vwap and count columns.
        data = data.drop(columns=["vwap", "count"])

        # Return the data.
        return data
