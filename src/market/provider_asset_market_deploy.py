import os
import sys
import datetime as dt

# Ensure the parent directory is in the Python path.
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from prefect import flow
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyLimitStrategy,
)
from prefect.docker import DockerImage
from prefect.schedules import Interval


if __name__ == "__main__":
    branch = os.getenv("GITHUB_BRANCH", "main")
    flow.from_source(
        source=".",
        entrypoint="src/market/provider_asset_market_flows.py:pull_provider_asset_market_data",
    ).deploy(  # type: ignore
        image=DockerImage(
            name="ghcr.io/manning-capital/mc-prefect-loaders",
            tag=branch,
            dockerfile="Dockerfile",
        ),
        name="pull_provider_asset_market_data",
        work_pool_name="kubernetes-default",
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
        schedule=Interval(
            dt.timedelta(minutes=30), anchor_date=dt.datetime(2000, 1, 1, 0, 0, 0)
        ),
        build=False,
        push=False,
    )
