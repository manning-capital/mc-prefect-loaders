import datetime as dt
from abc import ABC, abstractmethod

import dask
import pandas as pd
import dask.dataframe as dd
import mc_postgres_db.models as models
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, joinedload
from dask.distributed import Client
from sqlalchemy.engine import Engine


def convert_timedelta_to_dask_frequency(timedelta: dt.timedelta) -> str:
    """
    Convert a timedelta to a dask frequency string.
    """
    total_seconds = timedelta.total_seconds()

    if total_seconds >= 86400:
        days = int(total_seconds / 86400)
        return f"{days}D"
    elif total_seconds >= 3600:
        hours = int(total_seconds / 3600)
        return f"{hours}H"
    elif total_seconds >= 60:
        minutes = int(total_seconds / 60)
        return f"{minutes}T"
    else:
        seconds = int(total_seconds)
        return f"{seconds}S"


def align_timestamp_to_resolution(
    timestamp: dt.datetime, resolution: dt.timedelta
) -> dt.datetime:
    """
    Align a timestamp to the natural boundary of the given resolution.

    Examples:
    - 1 minute resolution: 12:34:56.789 → 12:34:00.000
    - 15 minute resolution: 12:34:56.789 → 12:30:00.000
    - 1 hour resolution: 12:34:56.789 → 12:00:00.000
    - 6 hour resolution: 12:34:56.789 → 12:00:00.000
    - 1 day resolution: 12:34:56.789 → 00:00:00.000

    Args:
        timestamp: The timestamp to align
        resolution: The time resolution (e.g., dt.timedelta(minutes=15))

    Returns:
        The aligned timestamp
    """
    # Remove timezone info for alignment
    naive_timestamp = timestamp.replace(tzinfo=None)

    # Calculate total seconds for easier comparison
    total_seconds = resolution.total_seconds()

    # Handle different resolution types with intermediate values
    if total_seconds >= 86400:  # 1 day or more
        # Daily or longer: align to midnight
        return naive_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

    elif total_seconds >= 3600:  # 1 hour or more
        # Hourly resolution: align to hour boundaries
        # Check if resolution evenly divides into hours
        hours = total_seconds / 3600
        if hours == int(hours) and 24 % int(hours) == 0:
            # Resolution evenly divides into 24 hours (e.g., 6 hours, 8 hours, 12 hours)
            hour_boundary = int(naive_timestamp.hour // int(hours)) * int(hours)
            return naive_timestamp.replace(
                hour=hour_boundary, minute=0, second=0, microsecond=0
            )
        else:
            # Standard hourly alignment
            return naive_timestamp.replace(minute=0, second=0, microsecond=0)

    elif total_seconds >= 60:  # 1 minute or more
        # Minute resolution: align to minute boundaries
        # Check if resolution evenly divides into minutes
        minutes = total_seconds / 60
        if minutes == int(minutes) and 60 % int(minutes) == 0:
            # Resolution evenly divides into 60 minutes (e.g., 5 min, 10 min, 15 min, 30 min)
            minute_boundary = int(naive_timestamp.minute // int(minutes)) * int(minutes)
            return naive_timestamp.replace(
                minute=minute_boundary, second=0, microsecond=0
            )
        else:
            # Standard minute alignment
            return naive_timestamp.replace(second=0, microsecond=0)

    else:
        # Second or shorter: align to the second
        return naive_timestamp.replace(microsecond=0)


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
    def batch_size(self) -> int:
        """
        Get the batch size for processing provider asset groups to manage memory usage.
        Default implementation returns 100.
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
    async def calculate_group_attributes(
        self, window: int, step: int, group_market_df: dd.DataFrame, client: Client
    ) -> dd.DataFrame:
        """
        Calculate the attributes for the provider asset group market data dataframes.
        Args:
            window: The window size for the rolling window.
            step: The step size for the rolling window.
            group_market_df: The dataframe with the provider asset market data for the group.
            client: The Dask client (kept for consistency, .compute() will use it automatically).
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

    async def get_provider_asset_group_market_data(
        self,
        client: Client,
        provider_asset_group_ids: set[int],
        start: dt.datetime,
        end: dt.datetime,
    ) -> dd.DataFrame:
        """
        Get the provider asset group market data.
        """

        # Align start and end times to the natural boundaries of the resolution
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)
        floor_start = align_timestamp_to_resolution(start_naive, self.resolution)
        ceil_end = align_timestamp_to_resolution(end_naive, self.resolution)
        freq = convert_timedelta_to_dask_frequency(self.resolution)
        provider_asset_group_ids_list = list(provider_asset_group_ids)
        
        # Generate the datetime grid using SQLAlchemy select
        datetime_grid_subquery = select(
            func.generate_series(floor_start, ceil_end, freq).label("timestamp")
        ).subquery(name="datetime_grid")
        
        # Build unique combinations query using SQLAlchemy ORM
        unique_combinations_subquery = (
            select(
                models.ProviderAssetGroupMember.provider_asset_group_id,
                models.ProviderAssetGroupMember.order,
                models.ProviderAssetGroupMember.provider_id,
                models.ProviderAssetGroupMember.from_asset_id,
                models.ProviderAssetGroupMember.to_asset_id,
            )
            .distinct()
            .where(
                models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                    provider_asset_group_ids_list
                )
            )
            .subquery(name="unique_combinations")
        )
        
        # Create the cross join query combining datetime grid with unique combinations
        datetime_grid_subquery_final = select(
            datetime_grid_subquery.c.timestamp,
            unique_combinations_subquery.c.provider_asset_group_id,
            unique_combinations_subquery.c.order,
            unique_combinations_subquery.c.provider_id,
            unique_combinations_subquery.c.from_asset_id,
            unique_combinations_subquery.c.to_asset_id,
        ).select_from(
            datetime_grid_subquery,
            unique_combinations_subquery
        ).subquery(name="grid_with_combinations")
        
        # Get unique combinations for market data filtering
        unique_combinations_df: pd.DataFrame = pd.read_sql(
            select(
                models.ProviderAssetGroupMember.provider_asset_group_id,
                models.ProviderAssetGroupMember.order,
                models.ProviderAssetGroupMember.provider_id,
                models.ProviderAssetGroupMember.from_asset_id,
                models.ProviderAssetGroupMember.to_asset_id,
            )
            .distinct()
            .where(
                models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                    provider_asset_group_ids_list
                )
            ),
            self.engine,
            index_col="provider_asset_group_id",
        )

        # Get unique IDs for market data filtering
        unique_provider_ids = unique_combinations_df["provider_id"].unique().tolist()
        unique_from_asset_ids = (
            unique_combinations_df["from_asset_id"].unique().tolist()
        )
        unique_to_asset_ids = unique_combinations_df["to_asset_id"].unique().tolist()

        # Get market columns
        market_columns: list[str] = [
            col.name for col in self.provider_asset_market_columns
        ]
        
        # Build the AS-OF join query using LATERAL join in PostgreSQL
        market_data_subquery = select(
            models.ProviderAssetMarket.timestamp,
            models.ProviderAssetMarket.provider_id,
            models.ProviderAssetMarket.from_asset_id,
            models.ProviderAssetMarket.to_asset_id,
            *[getattr(models.ProviderAssetMarket, col) for col in market_columns],
        ).where(
            models.ProviderAssetMarket.timestamp >= floor_start,
            models.ProviderAssetMarket.timestamp <= ceil_end,
            models.ProviderAssetMarket.provider_id.in_(unique_provider_ids),
            models.ProviderAssetMarket.from_asset_id.in_(unique_from_asset_ids),
            models.ProviderAssetMarket.to_asset_id.in_(unique_to_asset_ids),
        ).subquery(name="market_data")
        
        # Use LATERAL join for AS-OF join (backward fill)
        # This gets the most recent market data point <= the grid timestamp
        from sqlalchemy.sql import text
        
        # Convert queries to strings
        grid_query_str = str(datetime_grid_subquery_final.compile(compile_kwargs={"literal_binds": True}))
        market_query_str = str(market_data_subquery.compile(compile_kwargs={"literal_binds": True}))
        
        # Build the final query with string formatting
        as_of_join_sql = f"""
            SELECT 
                g.timestamp,
                g.provider_asset_group_id,
                g."order",
                g.provider_id,
                g.from_asset_id,
                g.to_asset_id,
                m.*
            FROM (
                SELECT timestamp, provider_asset_group_id, "order", provider_id, from_asset_id, to_asset_id
                FROM ({grid_query_str}) AS subgrid
            ) g
            LEFT JOIN LATERAL (
                SELECT *
                FROM ({market_query_str}) AS m
                WHERE m.provider_id = g.provider_id
                  AND m.from_asset_id = g.from_asset_id
                  AND m.to_asset_id = g.to_asset_id
                  AND m.timestamp <= g.timestamp
                ORDER BY m.timestamp DESC
                LIMIT 1
            ) m ON TRUE
            WHERE g.timestamp >= '{start_naive}'
              AND g.timestamp <= '{end_naive}'
        """
        
        as_of_join_query = text(as_of_join_sql)
        
        # Execute the AS-OF join query
        result_df = pd.read_sql(
            as_of_join_query,
            self.engine,
            index_col="timestamp"
        )
        
        # Drop rows where market data is null (AS-OF join didn't find a match)
        output = result_df.dropna()

        # Define key columns for ordering
        key_columns = [
            "provider_asset_group_id",
            "order",
            "provider_id",
            "from_asset_id",
            "to_asset_id",
        ]

        # Select the columns, keeping the timestamp and key columns first.
        output = output[
            [
                "timestamp",
                *key_columns,
                *[
                    col
                    for col in output.columns
                    if col not in key_columns and col not in ["timestamp"]
                ],
            ]
        ]

        # Transform the data on order keeping the provider asset group id column and pivoting the other columns.
        # Since output is now a pandas DataFrame, use pandas pivot_table
        pivotted_output = output.pivot_table(
            index=["timestamp", "provider_asset_group_id"],
            columns="order",
            values=[
                col
                for col in output.columns
                if col not in ["timestamp", "provider_asset_group_id", "order"]
            ],
        )

        # Convert back to Dask DataFrame for the return type
        return dd.from_pandas(pivotted_output.reset_index(), npartitions=1)

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
