import datetime as dt
from abc import ABC, abstractmethod

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
    def windows(self) -> list[dt.timedelta]:
        """
        Get the window sizes for the rolling window.
        For example, [dt.timedelta(days=1), dt.timedelta(days=2), dt.timedelta(days=3)] for a 3 day rolling window.
        """
        pass

    @property
    @abstractmethod
    def step(self) -> dt.timedelta:
        """
        Get the step size for the rolling window.
        For example, dt.timedelta(days=1) for a 1 day step size.
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> dt.timedelta:
        """
        Get the time resolution for the calculation frame.
        For example, dt.timedelta(minutes=1) for a 1 minute resolution.
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
    def provider_asset_market_columns(self) -> set:
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
    ) -> set[tuple[tuple[int, int, int], ...]]:
        """
        Convert the provider asset groups to a set of tuples of (provider_id, from_asset_id, to_asset_id).
        """
        return {
            tuple(
                sorted(
                    [
                        (member.provider.id, member.from_asset.id, member.to_asset.id)
                        for member in provider_asset_group.members
                    ],
                    key=lambda x: x,
                )
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
                .options(
                    joinedload(models.ProviderAssetGroup.members).joinedload(
                        models.ProviderAssetGroupMember.provider
                    )
                )
                .options(
                    joinedload(models.ProviderAssetGroup.members).joinedload(
                        models.ProviderAssetGroupMember.from_asset
                    )
                )
                .options(
                    joinedload(models.ProviderAssetGroup.members).joinedload(
                        models.ProviderAssetGroupMember.to_asset
                    )
                )
                .all()
            )

            return set(provider_asset_groups)

    def get_provider_asset_group_market_data(
        self, provider_asset_group_ids: set[int], start: dt.datetime, end: dt.datetime
    ) -> pl.DataFrame:
        """
        Get the provider asset group market data.
        """

        # Generate the datetime grid. Should be a polars dataframe with a timestamp column.
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)
        floor_start = start_naive.replace(second=0, microsecond=0)
        ceil_end = end_naive.replace(second=0, microsecond=0)
        datetime_grid = pl.DataFrame().with_columns(
            pl.datetime_range(
                floor_start, ceil_end, interval=self.resolution, eager=True
            ).alias("timestamp")
        )

        # Get unique provider asset group member combinations
        unique_combinations: pl.DataFrame = pl.read_database(
            query=select(
                models.ProviderAssetGroupMember.provider_asset_group_id,
                models.ProviderAssetGroupMember.order,
                models.ProviderAssetGroupMember.provider_id,
                models.ProviderAssetGroupMember.from_asset_id,
                models.ProviderAssetGroupMember.to_asset_id,
            )
            .distinct()
            .where(
                models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                    provider_asset_group_ids
                )
            ),
            connection=self.engine,
        )

        # Filter out duplicate order values within each group, keeping only the first occurrence
        filtered_combinations = []
        for group_id in unique_combinations["provider_asset_group_id"].unique():
            group_data = unique_combinations.filter(
                pl.col("provider_asset_group_id") == group_id
            )
            # Keep only the first occurrence of each order value within this group
            deduplicated_group = group_data.group_by("order").first()
            filtered_combinations.append(deduplicated_group)

        # Combine all filtered groups back into a single DataFrame
        if filtered_combinations:
            unique_combinations = pl.concat(filtered_combinations)
        else:
            unique_combinations = pl.DataFrame()

        # Get the market data.
        market_columns: list[str] = [
            col.name for col in self.provider_asset_market_columns
        ]
        market_data: pl.DataFrame = pl.read_database(
            query=select(
                models.ProviderAssetMarket.timestamp,
                models.ProviderAssetMarket.provider_id,
                models.ProviderAssetMarket.from_asset_id,
                models.ProviderAssetMarket.to_asset_id,
                *[getattr(models.ProviderAssetMarket, col) for col in market_columns],
            ).where(
                models.ProviderAssetMarket.timestamp.in_(datetime_grid["timestamp"]),
                models.ProviderAssetMarket.provider_id.in_(
                    unique_combinations["provider_id"]
                ),
                models.ProviderAssetMarket.from_asset_id.in_(
                    unique_combinations["from_asset_id"]
                ),
                models.ProviderAssetMarket.to_asset_id.in_(
                    unique_combinations["to_asset_id"]
                ),
            ),
            connection=self.engine,
        )

        # Join the datetime grid with the unique combinations and then the market data.
        output: pl.DataFrame = datetime_grid.join(unique_combinations, how="cross")
        output = output.join(
            market_data,
            on=["timestamp", "provider_id", "from_asset_id", "to_asset_id"],
            how="left",
        )

        # Forward fill by group using lambda.
        key_columns = [
            "provider_asset_group_id",
            "order",
            "provider_id",
            "from_asset_id",
            "to_asset_id",
        ]
        fill_columns = [col for col in output.columns if col not in key_columns]
        output = (
            output.group_by(key_columns)
            .agg(
                *[
                    pl.col(col).sort_by("timestamp").forward_fill()
                    for col in fill_columns
                ]
            )
            .explode(columns=fill_columns)
        )

        # Filter to requested time range and drop nulls
        output = output.filter(
            (pl.col("timestamp") >= start_naive) & (pl.col("timestamp") <= end_naive)
        ).drop_nulls()

        # Select the columns, keeping the timestamp and key columns first.
        output = output.select(
            "timestamp",
            *key_columns,
            *[
                col
                for col in output.columns
                if col not in key_columns and col not in ["timestamp"]
            ],
        )

        # Transform the data on order keeping the provider asset group id column and pivoting the other columns.
        pivotted_output = output.pivot(
            on="order",
            index=["timestamp", "provider_asset_group_id"],
            values=[
                col
                for col in output.columns
                if col not in ["timestamp", "provider_asset_group_id", "order"]
            ],
        )

        return pivotted_output

    def get_desired_provider_asset_group_ids(
        self, start_date: dt.date, end_date: dt.date
    ) -> set[int]:
        """
        Get the desired provider asset group ids.
        """
        return {
            provider_asset_group.id
            for provider_asset_group in self.get_desired_provider_asset_groups(
                start_date=start_date, end_date=end_date
            )
        }

    def get_current_provider_asset_group_ids(
        self,
    ) -> set[int]:
        """
        Get the current provider asset group ids.
        """
        return {
            provider_asset_group.id
            for provider_asset_group in self.get_current_provider_asset_groups()
        }

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

        # Convert the desired provider asset groups to a set of tuples of (provider_id, from_asset_id, to_asset_id).
        desired_provider_asset_tuples: set[tuple[tuple[int, int, int], ...]] = (
            self.__convert_provider_asset_groups_to_tuples(
                desired_provider_asset_groups
            )
        )

        # Convert the current provider asset groups to a set of tuples of (provider_id, from_asset_id, to_asset_id).
        current_provider_asset_tuples: set[tuple[tuple[int, int, int], ...]] = (
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
                        provider_id=provider_asset_group_member_tuple[0],
                        from_asset_id=provider_asset_group_member_tuple[1],
                        to_asset_id=provider_asset_group_member_tuple[2],
                        order=i + 1,
                    )
                    for i, provider_asset_group_member_tuple in enumerate(
                        provider_asset_tuple
                    )
                ]
                session.add(
                    models.ProviderAssetGroup(
                        asset_group_type_id=self.asset_group_type.id,
                        is_active=True,
                        members=provider_asset_group_members,
                    )
                )
            session.commit()
