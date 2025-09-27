import datetime as dt
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl
import mc_postgres_db.models as models
from sqlalchemy import func, select
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

    def calculate_attributes(
        self, start_date: dt.date, end_date: dt.date
    ) -> pl.DataFrame:
        """
        Calculate the attributes for the provider asset market data dataframes.
        Args:
            start_date: The start date to get the provider asset market data from.
            end_date: The end date to get the provider asset market data from.
        Returns:
            The dataframe with the attributes calculated for each provider asset market data dataframe.
        """

        # Get the current provider asset groups.
        provider_asset_groups = self.get_current_provider_asset_groups()

        # Generate a data from for all provider asset groups.
        provider_asset_group_ids = [
            provider_asset_group.id for provider_asset_group in provider_asset_groups
        ]
        provider_asset_group_members_df = pl.read_database(
            select(models.ProviderAssetGroupMember).where(
                models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                    provider_asset_group_ids
                )
            )
        )

        # Get the provider asset market data for each provider asset group.
        query_columns: set[str] = {
            models.ProviderAssetMarket.provider_id,
            models.ProviderAssetMarket.from_asset_id,
            models.ProviderAssetMarket.to_asset_id,
        }
        query_columns.update(self.provider_asset_market_columns)
        provider_asset_market_df = pl.read_database(
            select(*query_columns).where(
                models.ProviderAssetMarket.provider_id.in_(
                    provider_asset_group_members_df["provider_id"]
                ),
                models.ProviderAssetMarket.from_asset_id.in_(
                    provider_asset_group_members_df["from_asset_id"]
                ),
                models.ProviderAssetMarket.to_asset_id.in_(
                    provider_asset_group_members_df["to_asset_id"]
                ),
                models.ProviderAssetMarket.timestamp >= start_date,
                models.ProviderAssetMarket.timestamp <= end_date,
            )
        )

        # Cross-join the provider asset market data with the provider asset group members.
        provider_asset_market_df = provider_asset_market_df.join(
            provider_asset_group_members_df,
            on=["provider_id", "from_asset_id", "to_asset_id"],
        )

        # Pivot the provider asset market data based on the order so that each provider asset group member is a column.
        provider_asset_group_market_df = provider_asset_market_df.pivot(
            index="timestamp", columns="order", values=query_columns
        )

        # Calculate the attributes for the provider asset market data dataframes.
        attribute_df = pl.DataFrame()
        for window in self.windows:
            # Group by the provider, from asset id, and to asset id pivotted of the number of items. We will then apply the calculation across each group.
            attribute_df = attribute_df.vstack(
                self.__calculate_attributes(
                    window=window,
                    step=self.step,
                    *provider_asset_group_market_df.group_by(
                        ["provider_id", "from_asset_id", "to_asset_id"]
                    ),
                )
            )
        return attribute_df

    @abstractmethod
    def __calculated_group_attributes(
        self, window: int, step: int, *provider_asset_group_data_dfs: list[pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Calculate the attributes for the provider asset group data dataframes.
        Args:
            window: The window size for the rolling window.
            step: The step size for the rolling window.
            *provider_asset_group_data_dfs: The dataframes to calculate the attributes for.
        Returns:
            The dataframe with the attributes calculated for each provider asset group data dataframe.
        """
        pass

    @abstractmethod
    def get_symbol(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        """
        Get the desired provider asset group symbol.
        """
        pass

    @abstractmethod
    def get_name(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        """
        Get the desired provider asset group name.
        """
        pass

    @abstractmethod
    def get_description(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        """
        Get the desired provider asset group description.
        """
        pass

    def __convert_provider_asset_groups_to_tuples(
        self, provider_asset_groups: set[models.ProviderAssetGroup]
    ) -> set[set[tuple[int, int, int]]]:
        """
        Convert the provider asset groups to a set of sets of tuples.
        """
        return {
            set(
                (member.provider_id, member.from_asset_id, member.to_asset_id)
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
                    models.ProviderAssetGroup.id.label("group_id"),
                    func.count(models.ProviderAssetGroupMember.id).label(
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
                    models.ProviderAssetGroup.id == subq.c.group_id,
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

    def sync_provider_asset_groups(self, start: dt.datetime, end: dt.datetime) -> None:
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
        desired_provider_asset_tuples: set[set[tuple[int, int, int]]] = (
            self.__convert_provider_asset_groups_to_tuples(
                desired_provider_asset_groups
            )
        )

        # Convert the current provider asset groups to a set of sets of tuples.
        current_provider_asset_tuples: set[set[tuple[int, int, int]]] = (
            self.__convert_provider_asset_groups_to_tuples(
                current_provider_asset_groups
            )
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
                        provider_id=provider_id,
                        from_asset_id=from_asset_id,
                        to_asset_id=to_asset_id,
                        order=i + 1,
                    )
                    for i, provider_id, from_asset_id, to_asset_id in enumerate(
                        provider_asset_tuple
                    )
                ]
                session.add(
                    models.ProviderAssetGroup(
                        asset_group_type_id=self.asset_group_type.id,
                        symbol=self.get_symbol(*provider_asset_group_members),
                        name=self.get_name(*provider_asset_group_members),
                        description=self.get_description(*provider_asset_group_members),
                        is_active=True,
                        members=provider_asset_group_members,
                    )
                )
            session.commit()
