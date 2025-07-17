import pytest
from prefect.logging import disable_run_logger
from mc_postgres_db.testing.utilities import postgres_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with disable_run_logger():
        with postgres_test_harness(prefect_server_startup_timeout=60):
            yield
