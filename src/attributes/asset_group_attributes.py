import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
import itertools

import numpy as np
import pandas as pd
from dask import delayed
import dask.dataframe as dd
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import coint
from mc_postgres_db.prefect.asyncio.tasks import set_data
from statsmodels.regression.linear_model import OLS
from src.attributes.stochastic_models import OrnsteinUhlenbeck
import statsmodels.api as sm

from src.attributes.abstract import (
    AbstractAssetGroupType,
)

COINTEGRATION_P_VALUE_THRESHOLD = 0.001
MAX_PROVIDER_ASSET_MARKET_PAIRS = 15000


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

    @property
    def group_size(self) -> int:
        return 2

    @property
    def windows(self) -> list[dt.timedelta]:
        return [dt.timedelta(days=30), dt.timedelta(days=60), dt.timedelta(days=90)]

    @property
    def step(self) -> dt.timedelta:
        return dt.timedelta(days=1)

    @property
    def resolution(self) -> dt.timedelta:
        return dt.timedelta(minutes=1)

    @property
    def maximum_provider_asset_market_pairs(self) -> int:
        return MAX_PROVIDER_ASSET_MARKET_PAIRS

    @property
    def batch_size(self) -> int:
        return 10

    @property
    def provider_asset_market_columns(self) -> set:
        return {models.ProviderAssetMarket.close}

    def get_desired_provider_asset_groups(
        self, start_date: dt.date, end_date: dt.date
    ) -> set[models.ProviderAssetGroup]:
        """
        Get the new provider asset groups based on the provider asset market data in the database.
        """
        with Session(self.engine) as session:
            # Get the distinct provider asset market pairs in the given date range.
            provider = aliased(models.Provider)
            from_asset = aliased(models.Asset)
            to_asset = aliased(models.Asset)
            provider_asset_market_group_members = set(
                session.execute(
                    select(
                        provider,
                        from_asset,
                        to_asset,
                    )
                    .where(
                        models.ProviderAssetMarket.provider_id.in_(self.provider_ids),
                        models.ProviderAssetMarket.timestamp >= start_date,
                        models.ProviderAssetMarket.timestamp <= end_date,
                    )
                    .select_from(models.ProviderAssetMarket)
                    .join(
                        provider,
                        models.ProviderAssetMarket.provider_id == provider.id,
                    )
                    .join(
                        from_asset,
                        models.ProviderAssetMarket.from_asset_id == from_asset.id,
                    )
                    .join(
                        to_asset,
                        models.ProviderAssetMarket.to_asset_id == to_asset.id,
                    )
                    .distinct()
                ).tuples()
            )

            # Check the number of combinations. This number will overflow if it is greater than 135805301026.
            n_combinations: np.int64 = (
                np.int64(math.comb(len(provider_asset_market_group_members), 2))
                if len(provider_asset_market_group_members) < 135805301026
                else np.inf
            )

            # Group members by from_asset to ensure we only pair assets with the same from_asset
            # This allows pairs to form across different providers as long as they have the same from_asset
            grouped_members = {}
            for provider, from_asset, to_asset in provider_asset_market_group_members:
                if from_asset not in grouped_members:
                    grouped_members[from_asset] = []
                grouped_members[from_asset].append((provider, from_asset, to_asset))

            # Generate combinations only within each group (same from_asset)
            all_combinations = []
            for group_members in grouped_members.values():
                if len(group_members) >= 2:  # Need at least 2 members to form a pair
                    group_combinations = itertools.combinations(group_members, 2)
                    all_combinations.extend(group_combinations)

            # Limit the combinations to the maximum allowed if needed
            if len(all_combinations) > self.maximum_provider_asset_market_pairs:
                all_combinations = all_combinations[
                    : self.maximum_provider_asset_market_pairs
                ]

            return set(
                models.ProviderAssetGroup(
                    asset_group_type_id=self.asset_group_type.id,
                    is_active=True,
                    members=[
                        models.ProviderAssetGroupMember(
                            provider_id=member[0].id,
                            provider=member[0],
                            from_asset_id=member[1].id,
                            from_asset=member[1],
                            to_asset_id=member[2].id,
                            to_asset=member[2],
                            order=i + 1,
                        )
                        for i, member in enumerate(combination)
                    ],
                )
                for combination in all_combinations
            )

    @delayed
    def load_pairs_trading_frame_chunk(
        self,
        start: dt.datetime,
        end: dt.datetime,
        provider_asset_group_member_data: pd.DataFrame,
        provider_asset_market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Load the pairs trading frame for a chunk of provider asset groups.
        Returns only the essential columns needed for cointegration analysis.

        Args:
            start: Start datetime (timezone-naive)
            end: End datetime (timezone-naive)
            provider_asset_group_member_data: DataFrame of provider asset group member data
            provider_asset_market_data: DataFrame of provider asset market data

        Returns:
            pandas DataFrame indexed by provider_asset_group_id with columns:
                - timestamp
                - close_1
                - close_2
        """
        # Generate the time frame using pd.date_range.
        time_frame = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="1min")})

        # Cross join the provider asset group member data with the time frame.
        full_frame = time_frame.merge(provider_asset_group_member_data, how="cross")
        full_frame = full_frame.sort_values("timestamp")

        # Merge the provider asset market data with the full frame using merge_asof.
        full_market_frame = pd.merge_asof(
            full_frame,
            provider_asset_market_data,
            on="timestamp",
            by=["provider_id", "from_asset_id", "to_asset_id"],
            direction="backward",
        )

        # Split the full market frame by order and create pairs - only keep essential columns.
        close_1 = full_market_frame[full_market_frame["order"] == 1][
            ["timestamp", "provider_asset_group_id", "close"]
        ].rename(columns={"close": "close_1"})
        close_2 = full_market_frame[full_market_frame["order"] == 2][
            ["timestamp", "provider_asset_group_id", "close"]
        ].rename(columns={"close": "close_2"})

        # Merge the close_1 and close_2 frames to create pairs - only timestamp, close_1, close_2
        pairs = pd.merge(
            close_1, close_2, on=["timestamp", "provider_asset_group_id"], how="inner"
        )

        # Keep only essential columns.
        pairs = pairs[["provider_asset_group_id", "timestamp", "close_1", "close_2"]]

        # Set index to provider_asset_group_id.
        pairs = pairs.set_index("provider_asset_group_id")

        return pairs


    async def get_pairs_trading_frame(
        self,
        start: dt.datetime,
        end: dt.datetime,
        provider_asset_group_ids: list[int],
        provider_asset_group_member_data: pd.DataFrame,
        provider_asset_market_data: pd.DataFrame,
    ) -> dd.DataFrame:
        """
        Get the pairs trading frame with only essential columns for cointegration analysis.

        Args:
            start: Start datetime (timezone-naive)
            end: End datetime (timezone-naive)
            provider_asset_group_ids: List of provider asset group IDs to process
            members_data: Pre-loaded DataFrame of provider asset group members
            market_data_future: Broadcasted market data future
            client: Dask client

        Returns:
            Dask DataFrame indexed by provider_asset_group_id with columns:
                - timestamp
                - close_1
                - close_2
        """
        # Split provider asset groups into chunks
        n_chunks = min(len(self.client.cluster.workers), len(provider_asset_group_ids))
        group_chunks = np.array_split(provider_asset_group_ids, n_chunks)

        # Create delayed tasks with filtered member chunks
        delayed_dfs = []
        for chunk in group_chunks:
            # Filter members data for this specific chunk
            provider_asset_group_member_data_chunk = provider_asset_group_member_data[
                provider_asset_group_member_data["provider_asset_group_id"].isin(
                    chunk.tolist()
                )
            ].copy()

            delayed_dfs.append(
                self.load_pairs_trading_frame_chunk(
                    start,
                    end,
                    provider_asset_group_member_data_chunk,
                    provider_asset_market_data,
                )
            )

        # Define minimal schema
        meta = pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "close_1": pd.Series(dtype="float64"),
                "close_2": pd.Series(dtype="float64"),
            }
        )
        meta.index = pd.Index([], name="provider_asset_group_id", dtype="int64")

        # Convert to Dask DataFrame
        pairs_trading_frame = dd.from_delayed(delayed_dfs, meta=meta)

        # Reset the index and sort it.
        pairs_trading_frame = pairs_trading_frame.reset_index()
        pairs_trading_frame = pairs_trading_frame.set_index(
            "provider_asset_group_id", sorted=True
        )

        return pairs_trading_frame


    def get_cointegrated_stats(df: pd.DataFrame) -> pd.Series:
        """
        Get the cointegrated stats for a given dataframe.
        """

        # Compute the linear regression.
        X = df["close_1"].to_numpy()
        y = df["close_2"].to_numpy()
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()

        # Get the linear regression stats.
        linear_fit_alpha = results.params[0]
        linear_fit_beta = results.params[1]
        linear_fit_mse = results.mse_total
        linear_fit_r_squared = results.rsquared
        linear_fit_r_squared_adj = results.rsquared_adj
        residuals = results.resid

        # Fit the residuals to the Ornstein-Uhlenbeck process.
        ou_params = OrnsteinUhlenbeck().fit(residuals)

        return pd.Series(
            [
                linear_fit_alpha,
                linear_fit_beta,
                linear_fit_mse,
                linear_fit_r_squared,
                linear_fit_r_squared_adj,
                ou_params.mu,
                ou_params.theta,
                ou_params.sigma,
            ],
            index=[
                "linear_fit_alpha",
                "linear_fit_beta",
                "linear_fit_mse",
                "linear_fit_r_squared",
                "linear_fit_r_squared_adj",
                "ou_mu",
                "ou_theta",
                "ou_sigma",
            ],
            dtype=float,
        )


    async def save_attributes(
        self, provider_asset_group_ids: set[int], start: dt.datetime, end: dt.datetime
    ) -> None:
        """
        Save the attributes for the provider asset groups with the Dask client to the ProviderAssetGroupAttribute table.
        """

        # Get the provider asset group ids.
        provider_asset_group_ids = self.get_current_provider_asset_group_ids(
            is_active=True
        )

        # Get the provider asset group member data.
        provider_asset_group_member_data = self.get_provider_asset_group_member_data(
            provider_asset_group_ids=provider_asset_group_ids,
        )
        provider_ids = provider_asset_group_member_data["provider_id"].tolist()
        from_asset_ids = provider_asset_group_member_data["from_asset_id"].tolist()
        to_asset_ids = provider_asset_group_member_data["to_asset_id"].tolist()

        # Get the market data for the window.
        provider_asset_market_data = self.get_provider_asset_market_data(
            start=start,
            end=end,
            provider_ids=provider_ids,
            from_asset_ids=from_asset_ids,
            to_asset_ids=to_asset_ids,
        )

        # Scatter the provider asset market data to the dask cluster.
        self.logger.info(
            "Scattering the provider asset market data to the dask cluster..."
        )
        provider_asset_market_data_future = self.client.scatter(
            provider_asset_market_data, broadcast=True
        )

        # Get the pairs trading frame.
        self.logger.info("Getting the pairs trading frame...")
        pairs_trading_frame: dd.DataFrame = await self.get_pairs_trading_frame(
            start=start,
            end=end,
            provider_asset_group_ids=provider_asset_group_ids,
            provider_asset_group_member_data=provider_asset_group_member_data,
            provider_asset_market_data=provider_asset_market_data_future,
            client=self.client,
        )

        # Compute the pairs trading frame.
        cointegration_p_values: dd.DataFrame = pairs_trading_frame.groupby(
            "provider_asset_group_id"
        )[["close_1", "close_2"]].apply(
            lambda df: pd.Series(
                coint(df["close_1"], df["close_2"])[1], index=["p_value"]
            ),
            meta={"p_value": pd.Series([], dtype=float)},
        )

        # Compute the cointegration p-values.
        self.logger.info("Computing the cointegration p-values...")
        cointegration_p_values_computed: pd.DataFrame = cointegration_p_values.compute()

        # Filter the provider asset group member data to only include the provider asset group ids with cointegration p-values less than a threshold.
        self.logger.info(
            f"Filtering the pairs trading frame to only include the provider asset group ids with cointegration p-values less than {COINTEGRATION_P_VALUE_THRESHOLD}..."
        )
        cointegrated_provider_asset_group_ids = cointegration_p_values_computed.loc[
            cointegration_p_values_computed["p_value"] < COINTEGRATION_P_VALUE_THRESHOLD
        ].index.tolist()
        self.logger.info(
            f"Found {len(cointegrated_provider_asset_group_ids)} cointegrated provider asset group ids."
        )

        # Scatter the market data to the dask cluster.
        self.logger.info(
            "Scattering the provider asset market data to the dask cluster..."
        )
        provider_asset_market_data_future = self.client.scatter(
            provider_asset_market_data, broadcast=True
        )

        # Get the cointegrated provider asset group member data.
        cointegrated_pairs_trading_frame = await self.get_pairs_trading_frame(
            start,
            end,
            cointegrated_provider_asset_group_ids,
            provider_asset_group_member_data,
            provider_asset_market_data_future,
        )

        # Compute the statistical attributes for the cointegrated pairs trading frame.
        cointegrated_pairs_trading_stats = cointegrated_pairs_trading_frame.groupby(
            "provider_asset_group_id"
        )[["close_1", "close_2"]].apply(
            lambda df: self.get_cointegrated_stats(df),
            meta={
                "linear_fit_alpha": pd.Series([], dtype=float),
                "linear_fit_beta": pd.Series([], dtype=float),
                "linear_fit_mse": pd.Series([], dtype=float),
                "linear_fit_r_squared": pd.Series([], dtype=float),
                "linear_fit_r_squared_adj": pd.Series([], dtype=float),
                "ou_mu": pd.Series([], dtype=float),
                "ou_theta": pd.Series([], dtype=float),
                "ou_sigma": pd.Series([], dtype=float),
            },
        )

        # Compute the cointegrated pairs trading frame attributes.
        self.logger.info("Computing the cointegrated pairs trading frame attributes...")
        cointegrated_pairs_trading_stats_computed = (
            cointegrated_pairs_trading_stats.compute()
        )

        # Merge the cointegration p-values and the cointegrated pairs trading stats.
        toset = cointegration_p_values_computed.merge(
            cointegrated_pairs_trading_stats_computed, left_index=True, right_index=True
        ).reset_index()
        toset = toset.rename(columns={"p_value": "cointegration_p_value"})
        toset["lookback_window_seconds"] = (end - start).total_seconds()
        toset["timestamp"] = end
        toset = toset[
            [
                "timestamp",
                "provider_asset_group_id",
                "lookback_window_seconds",
                "cointegration_p_value",
                "linear_fit_alpha",
                "linear_fit_beta",
                "linear_fit_mse",
                "linear_fit_r_squared",
                "linear_fit_r_squared_adj",
                "ou_mu",
                "ou_theta",
                "ou_sigma",
            ]
        ]

        # Set the data to the database.
        await set_data(models.ProviderAssetGroupAttribute.__tablename__, toset)
