import datetime as dt

from src.attributes.abstract import align_timestamp_to_resolution


class TestTimeAlignment:
    """Test cases for the align_timestamp_to_resolution function."""

    def test_basic_resolutions(self):
        """Test basic time resolutions (second, minute, hour, day)."""
        test_timestamp = dt.datetime(2024, 1, 15, 12, 34, 56, 789000)

        test_cases = [
            (dt.timedelta(seconds=1), dt.datetime(2024, 1, 15, 12, 34, 56)),
            (dt.timedelta(minutes=1), dt.datetime(2024, 1, 15, 12, 34, 0)),
            (dt.timedelta(hours=1), dt.datetime(2024, 1, 15, 12, 0, 0)),
            (dt.timedelta(days=1), dt.datetime(2024, 1, 15, 0, 0, 0)),
        ]

        for resolution, expected in test_cases:
            result = align_timestamp_to_resolution(test_timestamp, resolution)
            assert result == expected, (
                f"Failed for {resolution}: got {result}, expected {expected}"
            )

    def test_intermediate_minute_resolutions(self):
        """Test intermediate minute resolutions that evenly divide into 60 minutes."""
        test_timestamp = dt.datetime(2024, 1, 15, 12, 34, 56, 789000)

        test_cases = [
            # (resolution, expected_result)
            (
                dt.timedelta(minutes=5),
                dt.datetime(2024, 1, 15, 12, 30, 0),
            ),  # 12:34 → 12:30
            (
                dt.timedelta(minutes=10),
                dt.datetime(2024, 1, 15, 12, 30, 0),
            ),  # 12:34 → 12:30
            (
                dt.timedelta(minutes=15),
                dt.datetime(2024, 1, 15, 12, 30, 0),
            ),  # 12:34 → 12:30
            (
                dt.timedelta(minutes=20),
                dt.datetime(2024, 1, 15, 12, 20, 0),
            ),  # 12:34 → 12:20
            (
                dt.timedelta(minutes=30),
                dt.datetime(2024, 1, 15, 12, 30, 0),
            ),  # 12:34 → 12:30
        ]

        for resolution, expected in test_cases:
            result = align_timestamp_to_resolution(test_timestamp, resolution)
            assert result == expected, (
                f"Failed for {resolution}: got {result}, expected {expected}"
            )

    def test_intermediate_hour_resolutions(self):
        """Test intermediate hour resolutions that evenly divide into 24 hours."""
        test_timestamp = dt.datetime(2024, 1, 15, 12, 34, 56, 789000)

        test_cases = [
            # (resolution, expected_result)
            (
                dt.timedelta(hours=2),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:34 → 12:00
            (
                dt.timedelta(hours=3),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:34 → 12:00
            (
                dt.timedelta(hours=4),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:34 → 12:00
            (
                dt.timedelta(hours=6),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:34 → 12:00
            (dt.timedelta(hours=8), dt.datetime(2024, 1, 15, 8, 0, 0)),  # 12:34 → 08:00
            (
                dt.timedelta(hours=12),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:34 → 12:00
        ]

        for resolution, expected in test_cases:
            result = align_timestamp_to_resolution(test_timestamp, resolution)
            assert result == expected, (
                f"Failed for {resolution}: got {result}, expected {expected}"
            )

    def test_six_hour_alignment_boundaries(self):
        """Test 6-hour alignment at different times of day."""
        test_cases = [
            (
                dt.datetime(2024, 1, 15, 2, 30, 0),
                dt.datetime(2024, 1, 15, 0, 0, 0),
            ),  # 02:30 → 00:00
            (
                dt.datetime(2024, 1, 15, 8, 30, 0),
                dt.datetime(2024, 1, 15, 6, 0, 0),
            ),  # 08:30 → 06:00
            (
                dt.datetime(2024, 1, 15, 14, 30, 0),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 14:30 → 12:00
            (
                dt.datetime(2024, 1, 15, 20, 30, 0),
                dt.datetime(2024, 1, 15, 18, 0, 0),
            ),  # 20:30 → 18:00
        ]

        for test_time, expected in test_cases:
            result = align_timestamp_to_resolution(test_time, dt.timedelta(hours=6))
            assert result == expected, (
                f"Failed for {test_time}: got {result}, expected {expected}"
            )

    def test_fifteen_minute_alignment_boundaries(self):
        """Test 15-minute alignment at different times."""
        test_cases = [
            (
                dt.datetime(2024, 1, 15, 12, 7, 0),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 12:07 → 12:00
            (
                dt.datetime(2024, 1, 15, 12, 22, 0),
                dt.datetime(2024, 1, 15, 12, 15, 0),
            ),  # 12:22 → 12:15
            (
                dt.datetime(2024, 1, 15, 12, 37, 0),
                dt.datetime(2024, 1, 15, 12, 30, 0),
            ),  # 12:37 → 12:30
            (
                dt.datetime(2024, 1, 15, 12, 52, 0),
                dt.datetime(2024, 1, 15, 12, 45, 0),
            ),  # 12:52 → 12:45
        ]

        for test_time, expected in test_cases:
            result = align_timestamp_to_resolution(test_time, dt.timedelta(minutes=15))
            assert result == expected, (
                f"Failed for {test_time}: got {result}, expected {expected}"
            )

    def test_non_evenly_dividing_resolutions(self):
        """Test resolutions that don't evenly divide into larger units."""
        test_timestamp = dt.datetime(2024, 1, 15, 12, 34, 56, 789000)

        test_cases = [
            # Non-evenly dividing resolutions should fall back to standard alignment
            (
                dt.timedelta(minutes=7),
                dt.datetime(2024, 1, 15, 12, 34, 0),
            ),  # 7 min → minute alignment
            (
                dt.timedelta(hours=5),
                dt.datetime(2024, 1, 15, 12, 0, 0),
            ),  # 5 hour → hour alignment
            (
                dt.timedelta(minutes=13),
                dt.datetime(2024, 1, 15, 12, 34, 0),
            ),  # 13 min → minute alignment
        ]

        for resolution, expected in test_cases:
            result = align_timestamp_to_resolution(test_timestamp, resolution)
            assert result == expected, (
                f"Failed for {resolution}: got {result}, expected {expected}"
            )

    def test_timezone_aware_timestamps(self):
        """Test that timezone-aware timestamps are handled correctly."""
        tz_timestamp = dt.datetime(
            2024, 1, 15, 12, 34, 56, 789000, tzinfo=dt.timezone.utc
        )
        expected = dt.datetime(2024, 1, 15, 12, 30, 0)

        result = align_timestamp_to_resolution(tz_timestamp, dt.timedelta(minutes=15))
        assert result == expected, (
            f"Failed for timezone-aware timestamp: got {result}, expected {expected}"
        )

    def test_already_aligned_timestamps(self):
        """Test that already aligned timestamps remain unchanged."""
        test_cases = [
            (dt.datetime(2024, 1, 15, 12, 0, 0, 0), dt.timedelta(hours=1)),
            (dt.datetime(2024, 1, 15, 0, 0, 0, 0), dt.timedelta(days=1)),
            (dt.datetime(2024, 1, 15, 12, 30, 0, 0), dt.timedelta(minutes=15)),
        ]

        for aligned_timestamp, resolution in test_cases:
            result = align_timestamp_to_resolution(aligned_timestamp, resolution)
            assert result == aligned_timestamp, (
                f"Failed for already aligned {aligned_timestamp}: got {result}"
            )

    def test_edge_cases(self):
        """Test edge cases like midnight, end of day, etc."""
        test_cases = [
            # Midnight
            (
                dt.datetime(2024, 1, 15, 0, 0, 0, 0),
                dt.timedelta(days=1),
                dt.datetime(2024, 1, 15, 0, 0, 0, 0),
            ),
            # End of day
            (
                dt.datetime(2024, 1, 15, 23, 59, 59, 999999),
                dt.timedelta(hours=1),
                dt.datetime(2024, 1, 15, 23, 0, 0, 0),
            ),
            # Leap year
            (
                dt.datetime(2024, 2, 29, 12, 34, 56),
                dt.timedelta(days=1),
                dt.datetime(2024, 2, 29, 0, 0, 0),
            ),
        ]

        for test_time, resolution, expected in test_cases:
            result = align_timestamp_to_resolution(test_time, resolution)
            assert result == expected, (
                f"Failed for edge case {test_time}: got {result}, expected {expected}"
            )

    def test_weekly_resolution(self):
        """Test weekly resolution alignment."""
        test_timestamp = dt.datetime(2024, 1, 15, 12, 34, 56, 789000)  # Monday
        expected = dt.datetime(2024, 1, 15, 0, 0, 0, 0)  # Should align to midnight

        result = align_timestamp_to_resolution(test_timestamp, dt.timedelta(days=7))
        assert result == expected, (
            f"Failed for weekly resolution: got {result}, expected {expected}"
        )
