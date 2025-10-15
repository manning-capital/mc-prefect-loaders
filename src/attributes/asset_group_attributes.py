import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import datetime as dt
import itertools

import numpy as np
import polars as pl
import statsmodels.api as sm
import mc_postgres_db.models as models
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import coint
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

    @property
    def group_size(self) -> int:
        return 2

    @property
    def windows(self) -> list[dt.timedelta]:
        return [dt.timedelta(days=7), dt.timedelta(days=30), dt.timedelta(days=90)]

    @property
    def step(self) -> dt.timedelta:
        return dt.timedelta(hours=1)

    @property
    def maximum_provider_asset_market_pairs(self) -> int:
        return 5000

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
                    name="-".join(
                        [f"{pair[2].name}{pair[1].name}" for pair in combination]
                    ),
                    description="-".join(
                        [f"{pair[2].name}{pair[1].name}" for pair in combination]
                    ),
                    is_active=True,
                    members=[
                        models.ProviderAssetGroupMember(
                            provider_id=pair[0].id,
                            provider=pair[0],
                            from_asset_id=pair[1].id,
                            from_asset=pair[1],
                            to_asset_id=pair[2].id,
                            to_asset=pair[2],
                            order=i + 1,
                        )
                        for i, pair in enumerate(combination)
                    ],
                )
                for combination in combinations
            )

    def calculate_group_attributes(
        self, window: dt.timedelta, step: dt.timedelta, group_market_df: pl.DataFrame
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

            # Get the close columns.
            close_1 = group_market_df_window["close_1"].to_numpy()
            close_2 = group_market_df_window["close_2"].to_numpy()

            # Perform a linear regression over the window of data.
            X = sm.add_constant(close_1)
            y = close_2
            linear_regression = OLS(y, X).fit()

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

            # Run the cointegration test.
            p_value[i] = coint(close_1, close_2)[1]

            # Fit the residuals to the Ornstein-Uhlenbeck process using AR(1) model.
            # The AR(1) process: X_t = phi * X_{t-1} + c + e_t
            # OU process: dX_t = theta*(mu - X_t)dt + sigma*dW_t
            # The mapping is: theta = -ln(phi), mu = c/(1-phi), sigma = std(residuals) * sqrt(2*theta/(1-phi**2))
            arima = ARIMA(residuals, order=(1, 0, 0)).fit()
            phi = arima.params.get("ar.L1", arima.params[1])  # AR(1) coefficient
            c = arima.params.get("const", arima.params[0])  # Intercept

            # Ensure phi is less than 1 for stationarity
            if abs(phi) < 1:
                theta[i] = -np.log(phi)
                mu[i] = c / (1 - phi)
                # Estimate sigma of noise
                sigma_e = np.sqrt(arima.sigma2)
                sigma[i] = sigma_e * np.sqrt(2 * theta[i] / (1 - phi**2))
            else:
                theta[i] = np.nan
                mu[i] = np.nan
                sigma[i] = np.nan

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
