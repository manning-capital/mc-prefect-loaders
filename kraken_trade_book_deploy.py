from datetime import datetime

from prefect import flow
from prefect.docker import DockerImage
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval
from prefect_github import GitHubCredentials
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyLimitStrategy,
)

if __name__ == "__main__":
    source = GitRepository(
        url="https://github.com/glynfinck/sentiment",
        credentials=GitHubCredentials.load("github-credentials"),
        branch="main",
    )
    flow.from_source(
        source=source, entrypoint="kraken_trade_book_flows.py:pull_kraken_orders"
    ).deploy(
        image=DockerImage(
            name="glynfinck/sentiment", tag="latest", dockerfile="Dockerfile"
        ),
        name="pull_kraken_orders",
        work_pool_name="kubernetes-default",
        parameters={
            "from_asset_ids": [1],
            "to_asset_ids": [2],
        },
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
        schedule=Interval(30, anchor_date=datetime(2000, 1, 1, 0, 0, 0)),
        build=False,
        push=False,
    )
