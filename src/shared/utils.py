from prefect import task
import pandas as pd
from typing import List, Tuple
from prefect.settings import PREFECT_API_URL


@task()
async def get_api_url() -> str:
    """
    Get the base URL for the Kraken API.
    """
    return PREFECT_API_URL.value()


@task()
async def get_base_url() -> str:
    """
    Get the base URL for the Kraken API.
    """
    api_url: str = await get_api_url()
    if not api_url:
        raise ValueError("PREFECT_API_URL environment variable is not set.")
    if api_url.endswith("/api"):
        api_url = api_url[:-4]  # Remove the "/api" suffix
    return api_url


def compare_dataframes(
    table_1: pd.DataFrame,
    table_2: pd.DataFrame,
    key_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare two DataFrames based on primary keys and return different types of results.

    Args:
        table_1: First DataFrame (old/existing data)
        table_2: Second DataFrame (new data)
        key_columns: List of column names that form the primary key

    Returns:
        Tuple containing:
        - records_in_1_not_2: Records that exist in table_1 but not in table_2
        - records_in_2_not_1: Records that exist in table_2 but not in table_1
        - exact_matches: Records that exist in both tables with identical values
        - different_records: Records that exist in both tables but have different values in non-key columns
    """
    # Ensure both DataFrames have the required key columns
    for key in key_columns:
        if key not in table_1.columns:
            raise ValueError(f"Key column '{key}' not found in table_1")
        if key not in table_2.columns:
            raise ValueError(f"Key column '{key}' not found in table_2")

    # Get all columns except key columns for comparison
    all_columns = list(set(table_1.columns) | set(table_2.columns))
    comparison_columns = [col for col in all_columns if col not in key_columns]

    # Check if the the columns are the same in both dataframes.
    symmetric_difference = set(table_1.columns) ^ set(table_2.columns)
    if len(symmetric_difference) != 0:
        raise ValueError(
            "Columns are not the same in both dataframes: {column_differences}".format(
                column_differences=", ".join(symmetric_difference)
            )
        )

    # Check that the comparison columns have the same data types in both dataframes. Ignore
    for col in comparison_columns:
        if table_1[col].dtype != table_2[col].dtype:
            raise ValueError(
                f"Column '{col}' has the type {table_1[col].dtype} in table_1 and {table_2[col].dtype} in table_2."
            )

    # Merge DataFrames on key columns with outer join and indicators
    merged_df = table_1.merge(
        table_2, on=key_columns, how="outer", suffixes=("_1", "_2"), indicator=True
    )

    # Records in table_1 but not in table_2
    records_in_1_not_2 = merged_df[merged_df["_merge"] == "left_only"].copy()
    records_in_1_not_2 = records_in_1_not_2.drop(
        columns=[
            col
            for col in records_in_1_not_2.columns
            if col.endswith("_2") or col == "_merge"
        ]
    )
    records_in_1_not_2.columns = [
        col.replace("_1", "") for col in records_in_1_not_2.columns
    ]

    # Records in table_2 but not in table_1
    records_in_2_not_1 = merged_df[merged_df["_merge"] == "right_only"].copy()
    records_in_2_not_1 = records_in_2_not_1.drop(
        columns=[
            col
            for col in records_in_2_not_1.columns
            if col.endswith("_1") or col == "_merge"
        ]
    )
    records_in_2_not_1.columns = [
        col.replace("_2", "") for col in records_in_2_not_1.columns
    ]

    # Records that exist in both tables
    common_records = merged_df[merged_df["_merge"] == "both"].copy()

    if not common_records.empty:
        # Check for differences in comparison columns
        different_mask = pd.Series(False, index=common_records.index)

        for col in comparison_columns:
            col_1 = f"{col}_1"
            col_2 = f"{col}_2"

            if col_1 in common_records.columns and col_2 in common_records.columns:
                # Compare values, handling NaN values properly
                series_1 = common_records[col_1]
                series_2 = common_records[col_2]

                # Check if values are different (but not both NaN)
                col_different = (series_1 != series_2) & ~(
                    pd.isna(series_1) & pd.isna(series_2)
                )
                different_mask = different_mask | col_different

        # Split into exact matches and different records
        exact_matches = common_records[~different_mask].copy()  # type: ignore
        different_records = common_records[different_mask].copy()  # type: ignore

        # Clean up exact_matches - use table_1 values (or table_2, doesn't matter since they're identical)
        for col in comparison_columns:
            col_1 = f"{col}_1"
            if col_1 in exact_matches.columns:  # type: ignore
                exact_matches[col] = exact_matches[col_1]  # type: ignore
        exact_matches = exact_matches.drop(  # type: ignore
            columns=[  # type: ignore
                col
                for col in exact_matches.columns  # type: ignore
                if col.endswith("_1") or col.endswith("_2") or col == "_merge"
            ]
        )

        # Clean up different_records - use table_2 values (newer data)
        for col in comparison_columns:
            col_2 = f"{col}_2"
            if col_2 in different_records.columns:  # type: ignore
                different_records[col] = different_records[col_2]  # type: ignore
        different_records = different_records.drop(  # type: ignore
            columns=[  # type: ignore
                col
                for col in different_records.columns  # type: ignore
                if col.endswith("_1") or col.endswith("_2") or col == "_merge"
            ]
        )
    else:
        # No common records
        exact_matches = pd.DataFrame(
            {
                column_name: pd.Series(dtype=table_1[column_name].dtype)
                for column_name in table_1.columns
            }
        )
        different_records = pd.DataFrame(
            {
                column_name: pd.Series(dtype=table_1[column_name].dtype)
                for column_name in table_1.columns
            }
        )

    return records_in_1_not_2, records_in_2_not_1, exact_matches, different_records  # type: ignore
