import os
import sys
from datetime import datetime

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
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval
from prefect_github import GitHubCredentials

from src.order.kraken_trade_book_flows import INTERVAL_SECONDS

if __name__ == "__main__":
    branch = os.getenv("GITHUB_BRANCH", "main")
    source = GitRepository(
        url="https://github.com/manning-capital/mc-prefect-loaders",
        credentials=GitHubCredentials.load("github-credentials"),
        branch=branch,
    )
    flow.from_source(
        source=source,
        entrypoint="src/order/kraken_trade_book_flows.py:pull_kraken_orders",
    ).deploy(
        image=DockerImage(
            name="ghcr.io/manning-capital/mc-prefect-loaders",
            tag=branch,
            dockerfile="Dockerfile",
        ),
        name="pull_kraken_orders",
        work_pool_name="kubernetes-default",
        parameters={
            "from_asset_ids": [1, 3, 3],
            "to_asset_ids": [2, 2, 1],
        },
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
        schedule=Interval(INTERVAL_SECONDS, anchor_date=datetime(2000, 1, 1, 0, 0, 0)),
        build=False,
        push=False,
    )
