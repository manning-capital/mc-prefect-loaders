import datetime as dt
from abc import ABC, abstractmethod

import polars as pl
import mc_postgres_db.models as models
from sqlalchemy import func
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
    def windows(self) -> list[str]:
        """
        Get the window sizes for the rolling window.
        For example, ["1d", "2d", "3d"] for a 3 day rolling window.
        """
        pass

    @property
    @abstractmethod
    def step(self) -> str:
        """
        Get the step size for the rolling window.
        For example, "1d" for a 1 day step size.
        """
        pass

    @property
    @abstractmethod
    def maximum_provider_asset_market_pairs(self) -> int:
        """
        Get the maximum number of provider asset market pairs for the asset group type.
        """
        pass

    @property
    @abstractmethod
    def group_size(self) -> int:
        """
        Get the exact number of members required for the asset group type.
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

    @property
    @abstractmethod
    def provider_asset_market_columns(self) -> set[str]:
        """
        Get the columns for the provider asset market data.
        """
        pass

    @property
    def provider_ids(self) -> list[int]:
        """
        Get the provider ids.
        """
        return [provider.id for provider in self.providers]

    @abstractmethod
    def calculate_group_attributes(
        self, window: int, step: int, group_market_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate the attributes for the provider asset group data dataframes.
        Args:
            window: The window size for the rolling window.
            step: The step size for the rolling window.
            group_market_df: The dataframe with the provider asset market data for the group.
        Returns:
            The dataframe with the attributes calculated for each provider asset group data dataframe.
        """
        pass

    def __convert_provider_asset_groups_to_tuples(
        self, provider_asset_groups: set[models.ProviderAssetGroup]
    ) -> set[tuple[tuple[models.Provider, models.Asset, models.Asset], ...]]:
        """
        Convert the provider asset groups to a set of sets of tuples.
        """
        return {
            tuple(
                (member.provider, member.from_asset, member.to_asset)
                for member in provider_asset_group.members
            )
            for provider_asset_group in provider_asset_groups
        }

    def get_current_provider_asset_groups(
        self,
    ) -> set[models.ProviderAssetGroup]:
        """
        Get the current provider asset groups based on the provider asset groups in the database,
        filtered by the exact number of members, at the query level.
        """
        with Session(self.engine) as session:
            # Subquery to count members per group.
            subq = (
                session.query(
                    models.ProviderAssetGroup.id.label("provider_asset_group_id"),
                    func.count(models.ProviderAssetGroupMember.order).label(
                        "member_count"
                    ),
                )
                .join(
                    models.ProviderAssetGroupMember,
                    models.ProviderAssetGroup.id
                    == models.ProviderAssetGroupMember.provider_asset_group_id,
                )
                .group_by(models.ProviderAssetGroup.id)
                .subquery()
            )

            # Main query: join with subquery and filter by member count
            provider_asset_groups = (
                session.query(models.ProviderAssetGroup)
                .join(
                    subq,
                    models.ProviderAssetGroup.id == subq.c.provider_asset_group_id,
                )
                .filter(
                    models.ProviderAssetGroup.asset_group_type_id
                    == self.asset_group_type.id,
                    subq.c.member_count == self.group_size,
                )
                .options(joinedload(models.ProviderAssetGroup.members))
                .all()
            )

            return set(provider_asset_groups)

    def refresh_provider_asset_groups(
        self, start: dt.datetime, end: dt.datetime
    ) -> None:
        """
        Sync the provider asset groups based on data from ProviderAssetMarket. Will get all combinations of provider and asset pairs in the given date range.
        Then, will compare these with the provider asset groups in the database. Will create new provider asset groups for any combinations that are not in the database.

        Args:
            start: The start date to get the provider asset market data from.
            end: The end date to get the provider asset market data from.
        """

        # Get all combinations of provider and asset pairs in the given date range.
        desired_provider_asset_groups = self.get_desired_provider_asset_groups(
            start_date=start, end_date=end
        )

        # Check if the number of desired provider asset group members is greater than the maximum number of members.
        if (
            len(desired_provider_asset_groups)
            > self.maximum_provider_asset_market_pairs
        ):
            raise ValueError(
                f"The number of desired provider asset group members is greater than the maximum number of members: {len(desired_provider_asset_groups)} > {self.maximum_provider_asset_market_pairs}"
            )

        # Get all active provider asset groups of the given type with at least the minimum number of members, with members loaded and sorted by ProviderAssetGroupMember.order.
        current_provider_asset_groups = self.get_current_provider_asset_groups()

        # Convert the desired provider asset groups to a set of sets of tuples.
        desired_provider_asset_tuples: set[
            tuple[tuple[models.Provider, models.Asset, models.Asset], ...]
        ] = self.__convert_provider_asset_groups_to_tuples(
            desired_provider_asset_groups
        )

        # Convert the current provider asset groups to a set of sets of tuples.
        current_provider_asset_tuples: set[
            tuple[tuple[models.Provider, models.Asset, models.Asset], ...]
        ] = self.__convert_provider_asset_groups_to_tuples(
            current_provider_asset_groups
        )

        # Get the new provider asset groups.
        new_provider_asset_tuples = (
            desired_provider_asset_tuples - current_provider_asset_tuples
        )

        # Create the new provider asset groups.
        with Session(self.engine) as session:
            for provider_asset_tuple in new_provider_asset_tuples:
                provider_asset_group_members = [
                    models.ProviderAssetGroupMember(
                        provider_id=provider.id,
                        from_asset_id=from_asset.id,
                        to_asset_id=to_asset.id,
                        order=i + 1,
                    )
                    for i, provider, from_asset, to_asset in enumerate(
                        provider_asset_tuple
                    )
                ]
                session.add(
                    models.ProviderAssetGroup(
                        asset_group_type_id=self.asset_group_type.id,
                        name=self.asset_group_type.name,
                        description=self.asset_group_type.description,
                        is_active=True,
                        members=provider_asset_group_members,
                    )
                )
            session.commit()
