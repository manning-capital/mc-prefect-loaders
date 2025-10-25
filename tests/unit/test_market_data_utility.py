import datetime as dt

import pandas as pd
import pytest

from tests.utils import generate_market_data_dataframe


def test_generate_market_data_dataframe_basic():
    """Test basic functionality of generate_market_data_dataframe."""
    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
    )

    # Check basic structure
    assert len(df) == 200, f"Expected 200 rows (2 assets * 100 points), got {len(df)}"
    assert list(df.columns) == [
        "timestamp",
        "provider_id",
        "from_asset_id",
        "to_asset_id",
        "close",
    ]
    assert set(df["to_asset_id"].unique()) == {2, 3}
    assert df["provider_id"].nunique() == 1
    assert df["from_asset_id"].nunique() == 1
    assert df["provider_id"].iloc[0] == 1
    assert df["from_asset_id"].iloc[0] == 1


def test_generate_market_data_dataframe_multiple_assets():
    """Test with multiple assets and cointegrated pairs."""
    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3, 4, 5],
        n_points=50,
        n_cointegrated_pairs=2,
        provider_id=1,
        from_asset_id=1,
    )

    # Check structure
    assert len(df) == 250, (
        f"Expected 250 rows (asset 2: 100 rows + assets 3,4,5: 50 rows each), got {len(df)}"
    )
    assert set(df["to_asset_id"].unique()) == {2, 3, 4, 5}

    # Check that each asset has the expected number of rows
    # Asset 2 appears in both pairs, so it should have 100 rows
    assert len(df[df["to_asset_id"] == 2]) == 100, (
        "Asset 2 should have 100 rows (appears in both pairs)"
    )
    # Assets 3, 4, 5 should each have 50 rows
    for asset_id in [3, 4, 5]:
        count = len(df[df["to_asset_id"] == asset_id])
        assert count == 50, f"Asset {asset_id} should have 50 rows, got {count}"


def test_generate_market_data_dataframe_validation():
    """Test input validation."""
    # Test too many cointegrated pairs
    with pytest.raises(ValueError, match="cannot exceed maximum possible pairs"):
        generate_market_data_dataframe(
            to_asset_ids=[2, 3],
            n_points=100,
            n_cointegrated_pairs=2,  # This should fail - only 1 pair possible with 2 assets
            provider_id=1,
            from_asset_id=1,
        )


def test_generate_market_data_dataframe_custom_params():
    """Test with custom cointegrated parameters."""
    custom_params = {
        "alpha": 5.0,
        "beta": 2.0,
        "drift": 0.1,
        "volatility": 0.3,
        "theta": 0.2,
        "mu": 0.05,
        "sigma": 1.5,
        "start_price": 50.0,
    }

    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        cointegrated_params=custom_params,
    )

    # Check that data was generated
    assert len(df) == 200
    assert df["close"].min() > 0, "All prices should be positive"
    assert df["close"].max() < 2000, "Prices should be reasonable"


def test_generate_market_data_dataframe_timestamps():
    """Test timestamp generation."""
    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=10,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        resolution=dt.timedelta(minutes=5),
    )

    # Check timestamp structure
    timestamps = df["timestamp"].unique()
    assert len(timestamps) == 10, (
        f"Expected 10 unique timestamps, got {len(timestamps)}"
    )

    # Check timestamp ordering
    timestamps_sorted = sorted(timestamps)
    assert list(timestamps) == timestamps_sorted, "Timestamps should be in order"

    # Check timestamp resolution
    if len(timestamps) > 1:
        diff = timestamps[1] - timestamps[0]
        assert diff == dt.timedelta(minutes=5), (
            f"Expected 5-minute resolution, got {diff}"
        )


def test_generate_market_data_dataframe_seed_reproducibility():
    """Test that the same seed produces the same results."""
    df1 = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        seed=42,
    )

    df2 = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        seed=42,
    )

    # Results should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_market_data_dataframe_different_seeds():
    """Test that different seeds produce different results."""
    df1 = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        seed=42,
    )

    df2 = generate_market_data_dataframe(
        to_asset_ids=[2, 3],
        n_points=100,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=1,
        seed=123,
    )

    # Results should be different (at least the close prices)
    assert not df1["close"].equals(df2["close"]), (
        "Different seeds should produce different prices"
    )


def test_generate_market_data_dataframe_edge_cases():
    """Test edge cases."""
    # Test with minimum values
    df = generate_market_data_dataframe(
        to_asset_ids=[2],
        n_points=1,
        n_cointegrated_pairs=0,  # No cointegrated pairs
        provider_id=1,
        from_asset_id=1,
    )

    assert len(df) == 1
    assert df["to_asset_id"].iloc[0] == 2

    # Test with maximum cointegrated pairs
    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3, 4],
        n_points=10,
        n_cointegrated_pairs=1,  # Maximum for 3 assets
        provider_id=1,
        from_asset_id=1,
    )

    assert len(df) == 30  # 3 assets * 10 points
    assert set(df["to_asset_id"].unique()) == {2, 3, 4}


def test_generate_market_data_dataframe_all_pairs_cointegrated():
    """Test that n_cointegrated_pairs=None makes all pairs cointegrated."""
    df = generate_market_data_dataframe(
        to_asset_ids=[2, 3, 4],  # 3 assets = 3 possible pairs: (2,3), (2,4), (3,4)
        n_points=10,
        n_cointegrated_pairs=None,  # All pairs should be cointegrated
        provider_id=1,
        from_asset_id=1,
    )

    # Should have all 3 assets, each appearing in 2 pairs (so 20 rows each)
    assert len(df) == 60  # 3 assets * 20 rows each (each asset appears in 2 pairs)
    assert set(df["to_asset_id"].unique()) == {2, 3, 4}

    # Each asset should appear in 2 pairs, so 20 rows each
    for asset_id in [2, 3, 4]:
        count = len(df[df["to_asset_id"] == asset_id])
        assert count == 20, (
            f"Asset {asset_id} should have 20 rows (appears in 2 pairs), got {count}"
        )


def test_generate_market_data_dataframe_explicit_to_asset_ids():
    """Test explicit to_asset_ids parameter."""
    df = generate_market_data_dataframe(
        to_asset_ids=[11, 12, 13],  # Explicitly provide asset IDs
        n_points=50,
        n_cointegrated_pairs=1,
        provider_id=1,
        from_asset_id=10,
    )

    # Should use the provided asset IDs
    expected_ids = {11, 12, 13}
    assert set(df["to_asset_id"].unique()) == expected_ids
