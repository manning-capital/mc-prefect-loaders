import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
import itertools

import mc_postgres_db.models as models
from sqlalchemy import select, distinct
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from src.attributes.abstract import AbstractAssetGroupType


class StatisticalPairsTrading(AbstractAssetGroupType):
    def __init__(self, engine: Engine):
        super().__init__(engine)

    @property
    def asset_group_type(self) -> models.AssetGroupType:
        with Session(self.engine) as session:
            return session.execute(
                select(models.AssetGroupType).where(
                    models.AssetGroupType.symbol == "STATISTICAL_PAIRS_TRADING"
                )
            ).scalar_one()

    @property
    def providers(self) -> list[models.Provider]:
        provider_name = ["Kraken"]
        with Session(self.engine) as session:
            return list(
                session.execute(
                    select(models.Provider).where(
                        models.Provider.name.in_(provider_name)
                    )
                ).scalars()
            )

    def get_desired_provider_asset_groups(
        self, start_date: dt.date, end_date: dt.date
    ) -> set[models.ProviderAssetGroup]:
        """
        Get the new provider asset groups based on the provider asset market data in the database.
        """
        with Session(self.engine) as session:
            # Get the distinct provider asset market pairs in the given date range.
            provider_asset_market_group_members = (
                session.execute(
                    select(
                        distinct(
                            models.ProviderAssetMarket.provider_id,
                            models.ProviderAssetMarket.from_asset_id,
                            models.ProviderAssetMarket.to_asset_id,
                        )
                    ).where(
                        models.ProviderAssetMarket.provider_id.in_(self.provider_ids),
                        models.ProviderAssetMarket.date >= start_date,
                        models.ProviderAssetMarket.date <= end_date,
                    ),
                )
                .scalars()
                .all()
            )

            # Convert the provider asset market pairs to a list of tuples.
            provider_asset_market_group_members = set(
                (pair.provider_id, pair.from_asset_id, pair.to_asset_id)
                for pair in provider_asset_market_group_members
            )

            # Limit the number of provider asset market pairs to the maximum number of provider asset market pairs.
            if (
                len(provider_asset_market_group_members)
                > self.maximum_provider_asset_market_pairs
            ):
                provider_asset_market_group_members = (
                    provider_asset_market_group_members[
                        : self.maximum_provider_asset_market_pairs
                    ]
                )

            # Check the number of combinations.
            n_combinations = math.comb(len(provider_asset_market_group_members), 2)

            # Check if the number of combinations is greater than the maximum number of provider asset groups.
            if n_combinations > self.maximum_provider_asset_groups:
                raise ValueError(
                    f"The number of combinations is greater than the maximum number of provider asset groups: {n_combinations} > {self.maximum_provider_asset_groups}"
                )

            # Get the combinations.
            combinations = itertools.combinations(
                provider_asset_market_group_members, 2
            )

            # g = ProviderAssetGroup(
            #     asset_group_type_id=self.asset_group_type.id,
            #     is_active=True,
            #     members=[
            #         models.ProviderAssetGroupMember(
            #             provider_id=provider_asset_pair_1.provider_id,
            #             from_asset_id=provider_asset_pair_1.from_asset_id,
            #             to_asset_id=provider_asset_pair_1.to_asset_id,
            #         )
            #     ]
            # )

            # return set(
            #     models.ProviderAssetGroupMember(
            #         provider_id=provider_asset_pair_1.provider_id,
            #         from_asset_id=provider_asset_pair_1.from_asset_id,
            #         to_asset_id=to_asset_id,
            #     )

            #     for combination in combinations
            # )
