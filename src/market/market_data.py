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
        return ["open", "high", "low", "close", "vwap", "volume", "count"]

    @property
    def rate_limit_name(self) -> str:
        return "kraken-api"

    async def get_asset_pairs(
        self, asset_codes: pd.Series, as_of_date: Optional[dt.date] = None
    ) -> pd.Series:
        # Get the asset pairs from Kraken.
        url = "https://api.kraken.com/0/public/AssetPairs"
        response = requests.get(url)
        response.raise_for_status()
        asset_pairs_result: dict[str, dict[str, Any]] = response.json()["result"]

        # Format the raw asset pairs result into a pandas dataframe.
        asset_pairs = pd.DataFrame(
            [
                {
                    "asset_pair": asset_pair,
                    "base_asset_code": asset_data["base"],
                    "quote_asset_code": asset_data["quote"],
                }
                for asset_pair, asset_data in asset_pairs_result.items()
            ]
        )

        # Filter the asset pairs to only include the asset codes that are in the asset_codes series.
        asset_pairs = asset_pairs.loc[
            asset_pairs["base_asset_code"].isin(asset_codes)
            | asset_pairs["quote_asset_code"].isin(asset_codes)
        ]

        return asset_pairs.apply(
            lambda row: (row["base_asset_code"], row["quote_asset_code"]), axis=1
        )

    async def get_market_data(
        self,
        from_asset_code: str,
        to_asset_code: str,
        as_of_date: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        # Request the OHLCV data for the asset pairs.
        url = "https://api.kraken.com/0/public/OHLC"
        response = requests.get(
            url, params={"pair": f"{from_asset_code}{to_asset_code}"}
        )
        response.raise_for_status()
        raw_data: list[list[Any]] = response.json()["result"][
            f"{from_asset_code}{to_asset_code}"
        ]

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
        data["timestamp"] = (
            data["timestamp"].dt.tz_localize(dt.timezone.utc).dt.tz_convert(None)
        )
        data["open"] = data["open"].astype(float)
        data["high"] = data["high"].astype(float)
        data["low"] = data["low"].astype(float)
        data["close"] = data["close"].astype(float)
        data["vwap"] = data["vwap"].astype(float)
        data["volume"] = data["volume"].astype(float)
        data["count"] = data["count"].astype(int)

        # Return the data.
        return data
