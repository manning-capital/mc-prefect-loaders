from typing import Any, Generator

import pytest
from sqlalchemy import Engine
from prefect.logging import disable_run_logger
from mc_postgres_db.prefect.tasks import get_engine
from mc_postgres_db.testing.utilities import clear_database, postgres_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with disable_run_logger():
        with postgres_test_harness(prefect_server_startup_timeout=60):
            yield


@pytest.fixture(autouse=True, scope="function")
def clear_db_fixture(prefect_test_fixture: Generator[None, Any, None]):
    """Automatically clear the database before and after each test."""

    # Get the engine.
    engine: Engine = get_engine()

    # Clear the database.
    clear_database(engine)

    # Yield the engine.
    yield engine

    # Clear the database.
    clear_database(engine)
