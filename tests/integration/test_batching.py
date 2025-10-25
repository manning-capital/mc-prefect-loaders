import datetime as dt
from unittest.mock import patch

import pytest
import mc_postgres_db.models as models
from sqlalchemy.orm import Session
from mc_postgres_db.prefect.asyncio.tasks import set_data, get_engine

from tests.utils import (
    sample_asset_data,
    sample_provider_data,
    generate_market_data_dataframe,
)
from src.attributes.asset_group_attributes import StatisticalPairsTrading
from src.attributes.provider_asset_attribute_flows import refresh_by_asset_group_type


@pytest.mark.asyncio
async def test_batching_creates_provider_asset_group_attributes():
    """
    Test that batching correctly creates ProviderAssetGroupAttribute records.
    This verifies that the batching logic works and produces the expected output.
    """
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [dt.timedelta(hours=1)],
        ),
        patch.object(StatisticalPairsTrading, "batch_size", return_value=2),
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            _,
            _,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create the pairs trading asset group type.
        with Session(engine) as session:
            pairs_trading_asset_group_type = models.AssetGroupType(
                symbol="STATISTICAL_PAIRS_TRADING",
                name="Statistical Pairs Trading",
                description="Group type for pairs trading attributes like the cointegration and hedge ratio.",
                is_active=True,
            )
            session.add(pairs_trading_asset_group_type)
            session.commit()
            session.refresh(pairs_trading_asset_group_type)

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,
                "volatility": 0.2,
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create StatisticalPairsTrading instance
        asset_group_type = StatisticalPairsTrading(engine)

        # Set up test parameters
        start = df["timestamp"].min()
        end = df["timestamp"].max()

        # Run the refresh function (batch_size=2 is already patched)
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Verify that ProviderAssetGroupAttribute records were created
        with Session(engine) as session:
            attribute_records = session.query(models.ProviderAssetGroupAttribute).all()

            # Should have at least one attribute record
            assert len(attribute_records) > 0, (
                "No ProviderAssetGroupAttribute records were created"
            )

            # Verify the records have the expected fields
            for record in attribute_records:
                assert record.provider_asset_group_id is not None, (
                    "provider_asset_group_id should not be None"
                )
                assert record.timestamp is not None, "timestamp should not be None"
                assert record.lookback_window_seconds is not None, (
                    "lookback_window_seconds should not be None"
                )
                assert record.linear_fit_beta is not None, (
                    "linear_fit_beta should not be None"
                )
                assert record.linear_fit_alpha is not None, (
                    "linear_fit_alpha should not be None"
                )
                assert record.ou_mu is not None, "ou_mu should not be None"
                assert record.ou_theta is not None, "ou_theta should not be None"
                assert record.ou_sigma is not None, "ou_sigma should not be None"


@pytest.mark.asyncio
async def test_attribute_data_consistency_across_batches():
    """
    Test that attribute data is set consistently regardless of batch size.
    This verifies that batching doesn't affect the final results.
    """
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [dt.timedelta(hours=1)],
        ),
        patch.object(StatisticalPairsTrading, "batch_size", return_value=2),
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            _,
            _,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create the pairs trading asset group type.
        with Session(engine) as session:
            pairs_trading_asset_group_type = models.AssetGroupType(
                symbol="STATISTICAL_PAIRS_TRADING",
                name="Statistical Pairs Trading",
                description="Group type for pairs trading attributes like the cointegration and hedge ratio.",
                is_active=True,
            )
            session.add(pairs_trading_asset_group_type)
            session.commit()
            session.refresh(pairs_trading_asset_group_type)

        # Generate market data with 1 cointegrated pair (BTC/USD and ETH/USD)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=1000,
            n_cointegrated_pairs=1,
            provider_id=kraken_provider.id,
            from_asset_id=usd_asset.id,
            cointegrated_params={
                "alpha": 10.0,
                "beta": 1.5,
                "drift": 0.05,
                "volatility": 0.2,
                "theta": 0.5,
                "mu": 0.1,
                "sigma": 2.0,
                "start_price": 100.0,
            },
        )

        # Set the data.
        await set_data(models.ProviderAssetMarket.__tablename__, df)

        # Create StatisticalPairsTrading instance
        asset_group_type = StatisticalPairsTrading(engine)

        # Set up test parameters
        start = df["timestamp"].min()
        end = df["timestamp"].max()

        # First, run with default batch size (100)
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Get the results from the first run
        with Session(engine) as session:
            default_batch_results = (
                session.query(models.ProviderAssetGroupAttribute)
                .order_by(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id,
                    models.ProviderAssetGroupAttribute.timestamp,
                )
                .all()
            )

        # Clear the attribute data
        with Session(engine) as session:
            session.query(models.ProviderAssetGroupAttribute).delete()
            session.commit()

        # Now run with a smaller batch size (2)
        with patch.object(StatisticalPairsTrading, "batch_size", return_value=2):
            await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Get the results from the second run
        with Session(engine) as session:
            small_batch_results = (
                session.query(models.ProviderAssetGroupAttribute)
                .order_by(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id,
                    models.ProviderAssetGroupAttribute.timestamp,
                )
                .all()
            )

        # Verify both runs produced the same number of results
        assert len(default_batch_results) == len(small_batch_results), (
            f"Different number of results: default batch={len(default_batch_results)}, "
            f"small batch={len(small_batch_results)}"
        )

        # Verify the results are identical
        for i, (default_result, small_result) in enumerate(
            zip(default_batch_results, small_batch_results)
        ):
            assert (
                default_result.provider_asset_group_id
                == small_result.provider_asset_group_id
            ), (
                f"Different provider_asset_group_id at index {i}: "
                f"default={default_result.provider_asset_group_id}, small={small_result.provider_asset_group_id}"
            )
            assert default_result.timestamp == small_result.timestamp, (
                f"Different timestamp at index {i}: "
                f"default={default_result.timestamp}, small={small_result.timestamp}"
            )
            assert (
                abs(default_result.linear_fit_beta - small_result.linear_fit_beta)
                < 1e-10
            ), (
                f"Different linear_fit_beta at index {i}: "
                f"default={default_result.linear_fit_beta}, small={small_result.linear_fit_beta}"
            )
            assert (
                abs(default_result.linear_fit_alpha - small_result.linear_fit_alpha)
                < 1e-10
            ), (
                f"Different linear_fit_alpha at index {i}: "
                f"default={default_result.linear_fit_alpha}, small={small_result.linear_fit_alpha}"
            )
            assert abs(default_result.ou_mu - small_result.ou_mu) < 1e-10, (
                f"Different ou_mu at index {i}: "
                f"default={default_result.ou_mu}, small={small_result.ou_mu}"
            )
            assert abs(default_result.ou_theta - small_result.ou_theta) < 1e-10, (
                f"Different ou_theta at index {i}: "
                f"default={default_result.ou_theta}, small={small_result.ou_theta}"
            )
            assert abs(default_result.ou_sigma - small_result.ou_sigma) < 1e-10, (
                f"Different ou_sigma at index {i}: "
                f"default={default_result.ou_sigma}, small={small_result.ou_sigma}"
            )
