import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.shared.utils import compare_dataframes


def test_basic_comparison():
    """Test basic comparison with new, dropped, and changed records."""
    # Create test data
    old_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "city": ["NYC", "LA", "Chicago", "Boston"],
        }
    )

    new_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 5],
            "name": ["Alice", "Bob Updated", "Charlie", "Eve"],
            "age": [25, 31, 35, 28],
            "city": ["NYC", "LA", "Chicago", "Miami"],
        }
    )

    # Compare dataframes
    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["id"]
    )

    # Test records in old but not new (dropped)
    assert len(in_1_not_2) == 1
    assert in_1_not_2.iloc[0]["id"] == 4
    assert in_1_not_2.iloc[0]["name"] == "David"

    # Test records in new but not old (new)
    assert len(in_2_not_1) == 1
    assert in_2_not_1.iloc[0]["id"] == 5
    assert in_2_not_1.iloc[0]["name"] == "Eve"

    # Test exact matches
    assert len(exact) == 2
    exact_ids = set(exact["id"].tolist())
    assert exact_ids == {1, 3}  # Alice and Charlie

    # Test different records
    assert len(different) == 1
    assert different.iloc[0]["id"] == 2
    assert different.iloc[0]["name"] == "Bob Updated"  # Should use new value
    assert different.iloc[0]["age"] == 31  # Should use new value


def test_empty_dataframes():
    """Test comparison with empty DataFrames."""
    empty_df = pd.DataFrame(columns=pd.Index(["id", "name"]))
    data_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

    # Both empty
    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        empty_df, empty_df, key_columns=["id"]
    )
    assert len(in_1_not_2) == 0
    assert len(in_2_not_1) == 0
    assert len(exact) == 0
    assert len(different) == 0

    # One empty
    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        empty_df, data_df, key_columns=["id"]
    )
    assert len(in_1_not_2) == 0
    assert len(in_2_not_1) == 2
    assert len(exact) == 0
    assert len(different) == 0


def test_multiple_key_columns():
    """Test comparison with multiple key columns."""
    old_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "role": ["admin", "user", "admin", "user"],
            "name": ["Alice", "Alice", "Bob", "Bob"],
            "permissions": ["all", "read", "all", "write"],
        }
    )

    new_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "role": ["admin", "user", "admin", "admin"],
            "name": ["Alice", "Alice Updated", "Bob", "Charlie"],
            "permissions": ["all", "read", "all", "all"],
        }
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["user_id", "role"]
    )

    # Test records in old but not new
    assert len(in_1_not_2) == 1  # user_id=2, role=user

    # Test records in new but not old
    assert len(in_2_not_1) == 1  # user_id=3, role=admin

    # Test exact matches
    assert len(exact) == 2  # user_id=1, role=admin and user_id=2, role=user

    # Test different records
    assert len(different) == 1  # user_id=2, role=user (name changed)


def test_nan_values():
    """Test comparison with NaN values."""
    old_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", None],
            "age": [25, None, 35],
            "city": ["NYC", "LA", None],
        }
    )

    new_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", None, "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"],
        }
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["id"]
    )

    # Test exact matches (where values are the same, including NaN)
    assert len(exact) == 1  # id=1 (Alice, 25, NYC)

    # Test different records (where values changed, including NaN changes)
    assert len(different) == 2  # id=2 and id=3 have changes


def test_different_column_sets():
    """Test comparison when DataFrames have different columns."""
    old_df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    new_df = pd.DataFrame(
        {
            "id": [1, 2, 4],
            "name": ["Alice", "Bob Updated", "David"],
            "city": ["NYC", "LA", "Chicago"],  # Different column
        }
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["id"]
    )

    # Test records in old but not new
    assert len(in_1_not_2) == 1  # id=3

    # Test records in new but not old
    assert len(in_2_not_1) == 1  # id=4

    # Test exact matches
    assert len(exact) == 1  # id=1 (only name matches)

    # Test different records
    assert len(different) == 1  # id=2 (name changed)


def test_missing_key_columns():
    """Test that function raises error when key columns are missing."""
    df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    df2 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

    # Test missing key column in first DataFrame
    with pytest.raises(ValueError):
        compare_dataframes(df1, df2, key_columns=["missing_key"])

    # Test missing key column in second DataFrame
    with pytest.raises(ValueError):
        compare_dataframes(df1, df2, key_columns=["id", "missing_key"])


def test_all_records_different():
    """Test when all records have differences."""
    old_df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    new_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice Updated", "Bob Updated", "Charlie Updated"],
            "age": [26, 31, 36],
        }
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["id"]
    )

    assert len(in_1_not_2) == 0
    assert len(in_2_not_1) == 0
    assert len(exact) == 0
    assert len(different) == 3  # All records have changes


def test_all_records_exact_match():
    """Test when all records are exact matches."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        df, df, key_columns=["id"]
    )

    assert len(in_1_not_2) == 0
    assert len(in_2_not_1) == 0
    assert len(exact) == 3  # All records match exactly
    assert len(different) == 0


def test_complex_data_types():
    """Test comparison with different data types."""
    old_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "is_active": [True, False, True],
            "score": [95.5, 87.2, 92.1],
        }
    )

    new_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie Updated"],
            "age": [25, 30, 35],
            "is_active": [True, True, True],  # Changed
            "score": [95.5, 87.2, 92.1],
        }
    )

    in_1_not_2, in_2_not_1, exact, different = compare_dataframes(
        old_df, new_df, key_columns=["id"]
    )

    assert len(in_1_not_2) == 0
    assert len(in_2_not_1) == 0
    assert len(exact) == 1  # id=1 (only name and age match exactly)
    assert len(different) == 2  # id=2 and id=3 (name and is_active changed)

    # Verify the different record uses new values
    different_record = different.loc[different["id"] == 2].iloc[0]  # type: ignore
    assert different_record["name"] == "Bob"
    assert different_record["is_active"]
    different_record = different.loc[different["id"] == 3].iloc[0]  # type: ignore
    assert different_record["name"] == "Charlie Updated"
    assert different_record["is_active"]
