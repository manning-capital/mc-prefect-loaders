import os
import sys

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from src.content.coin_desk_content_flows import (
    get_recent_coin_desk_content,
    get_current_coin_desk_content,
    save_coin_desk_content,
    pull_coin_desk_content,
)
from tests.mock_database import MockDatabase
