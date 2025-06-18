import pytest

from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


def test_placeholder():
    # This is a placeholder test to ensure the test suite runs without errors.
    # Replace with actual tests as needed.
    assert True
