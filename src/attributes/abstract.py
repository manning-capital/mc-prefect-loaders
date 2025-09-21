import datetime as dt
from abc import ABC, abstractmethod

import pandas as pd
import mc_postgres_db.models as models
from sqlalchemy import func, select, distinct
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.engine import Engine


class AbstractAssetGroupType(ABC):
    """
    Abstract class for asset group type.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    @property
    @abstractmethod
    def minimum_members(self) -> int:
        """
        Get the minimum number of members required for the asset group type.
        """
        pass

    @property
    @abstractmethod
    def maximum_members(self) -> int:
        """
        Get the maximum number of members for the asset group type.
        """
        pass

    @property
    @abstractmethod
    def providers(self) -> list[models.Provider]:
        """
        Get the providers for the asset group type.
        """
        pass

    @property
    @abstractmethod
    def asset_group_type(self) -> models.AssetGroupType:
        """
        Get the asset group type.
        """
        pass

    @abstractmethod
    def get_provider_asset_attribute_data(
        self, start_date: dt.date, end_date: dt.date
    ) -> pd.DataFrame:
        """
        Get the provider asset attribute data.
        """
        pass

    @property
    def provider_ids(self) -> list[int]:
        """
        Get the provider ids.
        """
        return [provider.id for provider in self.providers]

    def sync_provider_asset_groups(
        self, start: dt.datetime, end: dt.datetime
    ) -> list[models.ProviderAssetGroup]:
        """
        Sync the provider asset groups based on data from ProviderAssetMarket. Will get all combinations of provider and asset pairs in the given date range.
        Then, will compare these with the provider asset groups in the database. Will create new provider asset groups for any combinations that are not in the database.

        Args:
            start: The start date to get the provider asset market data from.
            end: The end date to get the provider asset market data from.

        Returns:
            A list of provider asset groups that were created or updated.
        """

        # Get all combinations of provider and asset pairs in the given date range.
        market_df = pd.read_sql(
            select(
                distinct(
                    models.ProviderAssetMarket.provider_id,
                    models.ProviderAssetMarket.from_asset_id,
                    models.ProviderAssetMarket.to_asset_id,
                )
            ).where(
                models.ProviderAssetMarket.provider_id.in_(self.provider_ids),
                models.ProviderAssetMarket.date >= start,
                models.ProviderAssetMarket.date <= end,
            ),
            self.engine,
        )

        # Convert the market_df to a list of tuples.
        desired_provider_asset_pairs = set(
            market_df.to_records(index=False).apply(tuple, axis=1).to_list()
        )

        # Get all active provider asset groups of the given type with at least the minimum number of members, with members loaded and sorted by ProviderAssetGroupMember.order.
        current_provider_asset_groups = self.get_provider_asset_groups()
        current_provider_asset_pairs = set(
            [
                (group.provider_id, member.from_asset_id, member.to_asset_id)
                for group in current_provider_asset_groups
                for member in group.members
            ]
        )

        # Get the provider asset groups that are not in the current provider asset pairs.
        new_provider_asset_groups = set(
            [
                group
                for group in desired_provider_asset_pairs
                if group not in current_provider_asset_pairs
            ]
        )

        # Get the provider asset groups that are in the current provider asset pairs.
        updated_provider_asset_groups = set(
            [
                group
                for group in desired_provider_asset_pairs
                if group in current_provider_asset_pairs
            ]
        )

        # Create the new provider asset groups.
        with Session(self.engine) as session:
            new_provider_asset_groups = [
                models.ProviderAssetGroup(
                    asset_group_type_id=self.asset_group_type.id, is_active=True
                )
                for group in new_provider_asset_groups
            ]
            session.add_all(new_provider_asset_groups)
            session.commit()

        return new_provider_asset_groups + updated_provider_asset_groups

    def get_provider_asset_groups(
        self, is_active: bool = None
    ) -> list[models.ProviderAssetGroup]:
        """
        Get all active provider asset groups of the given type with at least the minimum number of members and at most the maximum number of members, with members loaded and sorted by ProviderAssetGroupMember.order.
        """
        with Session(self.engine) as session:
            # Subquery to count members per group
            subq = (
                session.query(
                    models.ProviderAssetGroup.id.label("group_id"),
                    func.count(models.ProviderAssetGroupMember.id).label(
                        "member_count"
                    ),
                )
                .join(models.ProviderAssetGroup.members)
                .group_by(models.ProviderAssetGroup.id)
                .subquery()
            )

            # Main query: join with subquery to filter by member count
            provider_asset_groups = (
                session.query(models.ProviderAssetGroup)
                .join(subq, models.ProviderAssetGroup.id == subq.c.group_id)
                .filter(
                    models.ProviderAssetGroup.asset_group_type_id
                    == self.asset_group_type.id,
                    models.ProviderAssetGroup.is_active == is_active
                    if is_active is not None
                    else True,
                    subq.c.member_count >= self.minimum_members,
                    subq.c.member_count <= self.maximum_members,
                )
                .options(
                    joinedload(models.ProviderAssetGroup.members).joinedload(
                        models.ProviderAssetGroupMember.asset
                    )
                )
                .all()
            )

            # Sort members by order for each group
            for group in provider_asset_groups:
                group.members.sort(key=lambda m: m.order)

            return provider_asset_groups
