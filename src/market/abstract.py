from abc import ABC, abstractmethod
from sqlalchemy.engine import Engine
import pandas as pd
import datetime as dt
from mc_postgres_db.models import Provider, ProviderAsset, Asset
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Optional


class AbstractProviderAssetMarketData(ABC):
    key_columns = ["timestamp", "from_asset_id", "to_asset_id"]
    required_market_data_columns = ["timestamp", "from_asset_code", "to_asset_code"]
    required_asset_pair_data_columns = [
        "pair_asset_code",
        "from_asset_code",
        "to_asset_code",
    ]

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_provider_asset_map(
        self, as_of_date: Optional[dt.date] = None
    ) -> dict[str, int]:
        # Get the provider id.
        provider_id = self.get_provider().id

        # Get the provider asset data. Group by the asset and provider id. Then take the row where the date is the max.
        with Session(self.engine) as session:
            # Get the max date for each asset and provider id.
            subquery = (
                session.query(
                    ProviderAsset.asset_code,
                    ProviderAsset.provider_id,
                    func.max(ProviderAsset.date).label("max_date"),
                )
                .filter(
                    ProviderAsset.date <= as_of_date, ProviderAsset.is_active.is_(True)
                )
                .group_by(ProviderAsset.asset_code, ProviderAsset.provider_id)
                .subquery()
            )

            # Join the provider asset data with the max date for each asset and provider id.
            query = (
                session.query(ProviderAsset.asset_code, ProviderAsset.asset_id)
                .join(
                    subquery,
                    (ProviderAsset.asset_code == subquery.c.asset_code)
                    & (ProviderAsset.provider_id == subquery.c.provider_id)
                    & (ProviderAsset.date == subquery.c.max_date),
                )
                .join(Asset, Asset.id == ProviderAsset.asset_id)
                .where(
                    ProviderAsset.provider_id == provider_id, Asset.is_active.is_(True)
                )
            )

            # Read the query into a pandas dataframe.
            provider_asset_data = pd.read_sql(query.statement, self.engine)

            # Check for any duplicate asset codes.
            duplicated_asset_codes = provider_asset_data.loc[
                provider_asset_data.duplicated(subset=["asset_code"])
            ]
            if len(duplicated_asset_codes) > 0:
                raise ValueError(
                    f"The following asset id(s) had duplicated asset codes in the provider_asset table as-of {as_of_date}: {duplicated_asset_codes['asset_id'].tolist()}"
                )

            # Get the provider asset map.
            provider_asset_map = dict(
                zip(provider_asset_data["asset_code"], provider_asset_data["asset_id"])
            )

            return provider_asset_map

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def rate_limit_name(self) -> str:
        pass

    @abstractmethod
    def get_provider(self) -> Provider:
        pass

    @abstractmethod
    async def get_asset_pairs(
        self, asset_codes: pd.Series, as_of_date: Optional[dt.date] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_market_data(
        self,
        pair_asset_code: str,
        from_asset_code: str,
        to_asset_code: str,
        as_of_date: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        pass
