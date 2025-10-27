import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import dask.dataframe as dd
import statsmodels.api as sm
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased
from dask.distributed import Client
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

from src.attributes.abstract import AbstractAssetGroupType
from src.attributes.stochastic_models import OrnsteinUhlenbeck


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
        return dt.timedelta(hours=1)

    @property
    def resolution(self) -> dt.timedelta:
        return dt.timedelta(minutes=1)

    @property
    def maximum_provider_asset_market_pairs(self) -> int:
        return 15000

    @property
    def batch_size(self) -> int:
        return 100  # Process 100 pairs at a time

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

    async def calculate_group_attributes(
        self,
        window: dt.timedelta,
        step: dt.timedelta,
        group_market_df: dd.DataFrame,
        client: Client,
    ) -> dd.DataFrame:
        """
        Calculate the attributes for the provider asset group data dataframe.
        """

        # Determine the timestamps we will use as window anchors based on the step size (timedelta)
        timestamps = group_market_df["timestamp"].compute().sort_values().to_numpy()
        start_time = timestamps[0]
        end_time = timestamps[-1]
        anchor_timestamps: list[dt.datetime] = []
        current_time = start_time + window
        while current_time <= end_time:
            anchor_timestamps.append(current_time)
            current_time = current_time + step

        # Initialize the arrays.
        timestamp_anchor_array: np.ndarray[dt.datetime] = np.array(
            anchor_timestamps, dtype=dt.datetime
        )
        beta_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        alpha_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        mse_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        r_squared_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        r_squared_adj_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        theta_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        mu_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        sigma_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )
        p_value_array: np.ndarray[np.float64 | np.nan] = np.full(
            len(anchor_timestamps), np.nan, dtype=np.float64
        )

        # Perform a linear regression over the window and step size.
        for i, anchor_timestamp in enumerate(anchor_timestamps):
            try:
                # Get the window of data.
                group_market_df_window = group_market_df[
                    (group_market_df["timestamp"] >= anchor_timestamp - window)
                    & (group_market_df["timestamp"] <= anchor_timestamp)
                ]

                # Get the close columns.
                close_1 = group_market_df_window["close_1"].compute().to_numpy()
                close_2 = group_market_df_window["close_2"].compute().to_numpy()

                # Run the cointegration test.
                cointegration_result = coint(close_1, close_2)
                p_value_array[i] = cointegration_result[1]

                # Perform a linear regression over the window of data.
                X = sm.add_constant(close_1)
                y = close_2
                linear_regression = OLS(y, X).fit()

                # Get the slope and intercept of the linear regression.
                beta_array[i] = linear_regression.params[1]
                alpha_array[i] = linear_regression.params[0]
                residuals = linear_regression.resid

                # Calculate the mean squared error of the residuals.
                mse_array[i] = linear_regression.mse_total

                # Calculate the R-squared of the linear regression.
                r_squared_array[i] = linear_regression.rsquared

                # Calculate the adjusted R-squared of the linear regression.
                r_squared_adj_array[i] = linear_regression.rsquared_adj

                # Fit the residuals to the Ornstein-Uhlenbeck process.
                ou_params = OrnsteinUhlenbeck().fit(residuals)
                theta_array[i] = ou_params.theta
                mu_array[i] = ou_params.mu
                sigma_array[i] = ou_params.sigma

            except Exception as e:
                print(
                    f"Error fitting the residuals to the Ornstein-Uhlenbeck process for anchor timestamp {anchor_timestamp}: {e}"
                )
                beta_array[i] = np.nan
                alpha_array[i] = np.nan
                mse_array[i] = np.nan
                r_squared_array[i] = np.nan
                r_squared_adj_array[i] = np.nan
                theta_array[i] = np.nan
                mu_array[i] = np.nan
                sigma_array[i] = np.nan
                p_value_array[i] = np.nan
                continue

        # Return the provider asset group data dataframe as a dask dataframe.
        return dd.from_pandas(
            pd.DataFrame(
                {
                    models.ProviderAssetGroupAttribute.timestamp.name: timestamp_anchor_array,
                    models.ProviderAssetGroupAttribute.linear_fit_beta.name: beta_array,
                    models.ProviderAssetGroupAttribute.linear_fit_alpha.name: alpha_array,
                    models.ProviderAssetGroupAttribute.linear_fit_mse.name: mse_array,
                    models.ProviderAssetGroupAttribute.linear_fit_r_squared.name: r_squared_array,
                    models.ProviderAssetGroupAttribute.linear_fit_r_squared_adj.name: r_squared_adj_array,
                    models.ProviderAssetGroupAttribute.ou_theta.name: theta_array,
                    models.ProviderAssetGroupAttribute.ou_mu.name: mu_array,
                    models.ProviderAssetGroupAttribute.ou_sigma.name: sigma_array,
                    models.ProviderAssetGroupAttribute.cointegration_p_value.name: p_value_array,
                }
            ),
            npartitions=1,
        )
