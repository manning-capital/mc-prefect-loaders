import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from src.market.provider_asset_market_flows import pull_provider_asset_market_data


@pytest.mark.asyncio
async def test_pull_provider_asset_market_data():
    pass