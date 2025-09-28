import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
import itertools

import numpy as np
import polars as pl
import mc_postgres_db.models as models
from sqlalchemy import select, distinct
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS

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

    def get_symbol(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        return "-".join(
            [
                f"{member.to_asset_id}{member.from_asset_id}"
                for member in provider_asset_group_members
            ]
        )

    def get_name(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        return "-".join(
            [
                f"{member.to_asset_id}{member.from_asset_id}"
                for member in provider_asset_group_members
            ]
        )

    def get_description(
        self, *provider_asset_group_members: list[models.ProviderAssetGroupMember]
    ) -> str:
        return "-".join(
            [
                f"{member.to_asset_id}{member.from_asset_id}"
                for member in provider_asset_group_members
            ]
        )

    @property
    def group_size(self) -> int:
        return 2

    @property
    def windows(self) -> list[str]:
        return ["1d", "2d", "3d"]

    @property
    def step(self) -> str:
        return "1d"

    @property
    def maximum_provider_asset_market_pairs(self) -> int:
        return 5000

    @property
    def provider_asset_market_columns(self) -> set[str]:
        return {"close"}

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

            # Check the number of combinations. This number will overflow if it is greater than 135805301026.
            n_combinations: np.int64 = (
                np.int64(math.comb(len(provider_asset_market_group_members), 2))
                if len(provider_asset_market_group_members) >= 135805301026
                else np.inf
            )

            # Check if the number of combinations is greater than the maximum number of provider asset groups.
            if n_combinations > self.maximum_provider_asset_market_pairs:
                raise ValueError(
                    f"The number of combinations is greater than the maximum number of provider asset groups: {n_combinations} > {self.maximum_provider_asset_market_pairs}"
                )

            # Get the combinations.
            combinations = itertools.combinations(
                provider_asset_market_group_members, 2
            )

            return set(
                models.ProviderAssetGroup(
                    asset_group_type_id=self.asset_group_type.id,
                    symbol=self.get_symbol(*combination),
                    name=self.get_name(*combination),
                    description=self.get_description(*combination),
                    is_active=True,
                    members=[
                        models.ProviderAssetGroupMember(
                            provider_id=pair.provider_id,
                            from_asset_id=pair.from_asset_id,
                            to_asset_id=pair.to_asset_id,
                            order=i + 1,
                        )
                        for i, pair in enumerate(combination)
                    ],
                )
                for combination in combinations
            )

    def calculate_group_attributes(
        self, window: int, step: int, group_market_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate the attributes for the provider asset group data dataframe.
        """

        # Initialize the arrays.
        beta = np.empty(len(group_market_df) // step)
        alpha = np.empty(len(group_market_df) // step)
        timestamp = np.empty(len(group_market_df) // step)
        mse = np.empty(len(group_market_df) // step)
        r_squared = np.empty(len(group_market_df) // step)
        r_squared_adj = np.empty(len(group_market_df) // step)
        theta = np.empty(len(group_market_df) // step)
        mu = np.empty(len(group_market_df) // step)
        sigma = np.empty(len(group_market_df) // step)
        p_value = np.empty(len(group_market_df) // step)

        # Perform a linear regression over the window and step size.
        for i in range(0, len(group_market_df), step):
            # Get the window of data.
            group_market_df_window = group_market_df.slice(i, window)

            # Perform a linear regression over the window of data.
            linear_regression = OLS(
                group_market_df_window["close"], group_market_df_window["timestamp"]
            ).fit()

            # Get the slope and intercept of the linear regression.
            timestamp[i] = group_market_df_window["timestamp"].max()
            beta[i] = linear_regression.params[1]
            alpha[i] = linear_regression.params[0]
            residuals = linear_regression.resid

            # Calculate the mean squared error of the residuals.
            mse[i] = linear_regression.mse

            # Calculate the R-squared of the linear regression.
            r_squared[i] = linear_regression.rsquared

            # Calculate the adjusted R-squared of the linear regression.
            r_squared_adj[i] = linear_regression.rsquared_adj

            # Run the adfuller test on the residuals.
            adfuller_result = adfuller(residuals)

            # Get the p-value of the adfuller test.
            p_value[i] = adfuller_result[1]

            # Fit the residuals to the Ornstein-Uhlenbeck process.
            ornstein_uhlenbeck = ARIMA(residuals, order=(1, 1, 0)).fit()

            # Get the parameters of the Ornstein-Uhlenbeck process.
            theta[i] = ornstein_uhlenbeck.params[0]
            mu[i] = ornstein_uhlenbeck.params[1]
            sigma[i] = ornstein_uhlenbeck.params[2]

        # Return the provider asset group data dataframe.
        return pl.DataFrame(
            {
                models.ProviderAssetGroupAttribute.timestamp.name: timestamp,
                models.ProviderAssetGroupAttribute.linear_fit_beta.name: beta,
                models.ProviderAssetGroupAttribute.linear_fit_alpha.name: alpha,
                models.ProviderAssetGroupAttribute.linear_fit_mse.name: mse,
                models.ProviderAssetGroupAttribute.linear_fit_r_squared.name: r_squared,
                models.ProviderAssetGroupAttribute.linear_fit_r_squared_adj.name: r_squared_adj,
                models.ProviderAssetGroupAttribute.ou_theta.name: theta,
                models.ProviderAssetGroupAttribute.ou_mu.name: mu,
                models.ProviderAssetGroupAttribute.ou_sigma.name: sigma,
                models.ProviderAssetGroupAttribute.cointegration_p_value.name: p_value,
            }
        )
