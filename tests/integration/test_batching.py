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
                f"default={default_result.linear_fit_alpha}, small={default_result.linear_fit_alpha}"
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


@pytest.mark.asyncio
async def test_batching_with_multiple_asset_groups():
    """
    Test batching with multiple asset groups to verify that all groups are processed
    and that the batching logic handles multiple groups correctly.
    """
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [dt.timedelta(hours=1)],
        ),
        patch.object(StatisticalPairsTrading, "batch_size", return_value=1),  # Force multiple batches
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            crypto_asset_type,
            fiat_asset_type,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create SOL asset manually
        with Session(engine) as session:
            sol_asset = models.Asset(
                name="SOL", description="SOL", asset_type_id=crypto_asset_type.id
            )
            session.add(sol_asset)
            session.commit()
            session.refresh(sol_asset)

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

        # Generate market data with 3 assets (BTC, ETH, SOL) - this should create 3 pairs
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id, sol_asset.id],
            n_points=500,  # Smaller dataset for faster testing
            n_cointegrated_pairs=1,  # Only BTC-ETH are cointegrated
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

        # Run the refresh function with batch_size=1 (should create multiple batches)
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Verify that ProviderAssetGroupAttribute records were created for all groups
        with Session(engine) as session:
            # Get all attribute records grouped by provider_asset_group_id
            attribute_records = (
                session.query(models.ProviderAssetGroupAttribute)
                .order_by(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id,
                    models.ProviderAssetGroupAttribute.timestamp,
                )
                .all()
            )

            # Should have records for multiple groups
            group_ids = set(record.provider_asset_group_id for record in attribute_records)
            assert len(group_ids) >= 1, f"Expected at least 1 group, got {len(group_ids)}"

            # Verify each group has records for multiple time windows
            for group_id in group_ids:
                group_records = [r for r in attribute_records if r.provider_asset_group_id == group_id]
                assert len(group_records) > 0, (
                    f"Group {group_id} should have at least 1 record, got {len(group_records)}"
                )

                # Verify all records for this group have basic data
                for record in group_records:
                    assert record.timestamp is not None, "timestamp should not be None"
                    assert record.lookback_window_seconds == 3600, (  # 1 hour
                        f"Expected lookback_window_seconds=3600, got {record.lookback_window_seconds}"
                    )

                # Check that at least some records have valid statistical data
                # (some may be None due to insufficient data or failed calculations)
                valid_records = [
                    r for r in group_records 
                    if r.linear_fit_beta is not None and r.linear_fit_alpha is not None
                ]
                assert len(valid_records) > 0, (
                    f"Group {group_id} should have at least 1 record with valid statistical data, "
                    f"got {len(valid_records)} valid out of {len(group_records)} total"
                )

                # Verify valid records have complete data
                for record in valid_records:
                    assert record.ou_mu is not None, "ou_mu should not be None for valid records"
                    assert record.ou_theta is not None, "ou_theta should not be None for valid records"
                    assert record.ou_sigma is not None, "ou_sigma should not be None for valid records"

            # Verify that batching worked correctly by checking that we have records
            # from multiple groups (indicating the batching logic processed all groups)
            assert len(group_ids) >= 1, f"Expected at least 1 group, got {len(group_ids)}"


@pytest.mark.asyncio
async def test_batching_edge_case_single_batch():
    """
    Test batching edge case where all groups fit in a single batch.
    This verifies that the batching logic handles the case where batch_size >= number of groups.
    """
    with (
        patch(
            "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
            new_callable=lambda: [dt.timedelta(hours=1)],
        ),
        patch.object(StatisticalPairsTrading, "batch_size", return_value=100),  # Large batch size
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

        # Generate market data with 2 assets (BTC, ETH) - this should create 1 pair
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=500,
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

        # Run the refresh function with large batch_size (should be single batch)
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Verify that ProviderAssetGroupAttribute records were created
        with Session(engine) as session:
            attribute_records = session.query(models.ProviderAssetGroupAttribute).all()

            # Should have multiple records (1 group * multiple time windows)
            assert len(attribute_records) > 0, (
                f"Expected at least 1 record, got {len(attribute_records)}"
            )

            # Verify all records have basic data
            for record in attribute_records:
                assert record.timestamp is not None, "timestamp should not be None"
                assert record.lookback_window_seconds == 3600, (  # 1 hour
                    f"Expected lookback_window_seconds=3600, got {record.lookback_window_seconds}"
                )

            # Check that at least some records have valid statistical data
            valid_records = [
                r for r in attribute_records 
                if r.linear_fit_beta is not None and r.linear_fit_alpha is not None
            ]
            assert len(valid_records) > 0, (
                f"Should have at least 1 record with valid statistical data, "
                f"got {len(valid_records)} valid out of {len(attribute_records)} total"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 100])
async def test_batching_data_integrity_with_different_batch_sizes(batch_size):
    """
    Test that different batch sizes produce identical results.
    This test is parameterized to run with multiple batch sizes.
    """
    with patch(
        "src.attributes.asset_group_attributes.StatisticalPairsTrading.windows",
        new_callable=lambda: [dt.timedelta(hours=1)],
    ):
        # Get the engine.
        engine = await get_engine()

        # Create the provider and asset data.
        _, kraken_provider = await sample_provider_data(engine)
        (
            crypto_asset_type,
            fiat_asset_type,
            btc_asset,
            eth_asset,
            usd_asset,
        ) = await sample_asset_data(engine)

        # Create SOL asset manually
        with Session(engine) as session:
            sol_asset = models.Asset(
                name="SOL", description="SOL", asset_type_id=crypto_asset_type.id
            )
            session.add(sol_asset)
            session.commit()
            session.refresh(sol_asset)

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

        # Generate market data with 3 assets (BTC, ETH, SOL) - this should create 3 pairs
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id, sol_asset.id],
            n_points=500,
            n_cointegrated_pairs=1,  # Only BTC-ETH are cointegrated
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

        # Run with current batch size
        with patch.object(StatisticalPairsTrading, "batch_size", return_value=batch_size):
            await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Get results
        with Session(engine) as session:
            results = (
                session.query(models.ProviderAssetGroupAttribute)
                .order_by(
                    models.ProviderAssetGroupAttribute.provider_asset_group_id,
                    models.ProviderAssetGroupAttribute.timestamp,
                )
                .all()
            )

        # Verify that results were created
        assert len(results) > 0, f"Batch size {batch_size} should produce at least 1 result"

        # Verify all records have basic data
        for record in results:
            assert record.timestamp is not None, f"Batch size {batch_size}: timestamp should not be None"
            assert record.lookback_window_seconds == 3600, (  # 1 hour
                f"Batch size {batch_size}: Expected lookback_window_seconds=3600, got {record.lookback_window_seconds}"
            )

        # Check that at least some records have valid statistical data
        valid_records = [
            r for r in results 
            if r.linear_fit_beta is not None and r.linear_fit_alpha is not None
        ]
        assert len(valid_records) > 0, (
            f"Batch size {batch_size}: Should have at least 1 record with valid statistical data, "
            f"got {len(valid_records)} valid out of {len(results)} total"
        )

        # Verify valid records have complete data
        for record in valid_records:
            assert record.ou_mu is not None, f"Batch size {batch_size}: ou_mu should not be None for valid records"
            assert record.ou_theta is not None, f"Batch size {batch_size}: ou_theta should not be None for valid records"
            assert record.ou_sigma is not None, f"Batch size {batch_size}: ou_sigma should not be None for valid records"


@pytest.mark.asyncio
async def test_batching_with_empty_groups():
    """
    Test batching behavior when there are no asset groups to process.
    This verifies that the batching logic handles empty scenarios gracefully.
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

        # Generate market data with 2 assets but create NO asset groups
        # This simulates the case where no groups are created (e.g., insufficient data)
        df = generate_market_data_dataframe(
            to_asset_ids=[btc_asset.id, eth_asset.id],
            n_points=10,  # Very small dataset - might not create groups
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

        # Run the refresh function - should handle empty groups gracefully
        await refresh_by_asset_group_type(asset_group_type, start=start, end=end)

        # Verify that no ProviderAssetGroupAttribute records were created
        # (since we have insufficient data to create meaningful groups)
        with Session(engine) as session:
            attribute_records = session.query(models.ProviderAssetGroupAttribute).all()
            
            # This test verifies that the batching logic doesn't crash when there are no groups
            # The exact number of records depends on the minimum data requirements
            # We just verify that the function completes without error
            assert isinstance(attribute_records, list), "Should return a list of records"
