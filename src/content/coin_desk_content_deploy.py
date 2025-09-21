import os
import sys
import datetime as dt

# Ensure the parent directory is in the Python path.
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from prefect import flow
from prefect.docker import DockerImage
from prefect_github import GitHubCredentials
from prefect.schedules import Interval
from prefect.runner.storage import GitRepository
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyLimitStrategy,
)

if __name__ == "__main__":
    branch = os.getenv("GITHUB_BRANCH", "main")
    credentials = GitHubCredentials.load("github-credentials")  # type: ignore
    source = GitRepository(
        url="https://github.com/manning-capital/mc-prefect-loaders",
        credentials=credentials,  # type: ignore
        branch=branch,
    )
    flow.from_source(
        source=source,
        entrypoint="src/content/coin_desk_content_flows.py:pull_coindesk_news_content",
    ).deploy(  # type: ignore
        image=DockerImage(
            name="ghcr.io/manning-capital/mc-prefect-loaders",
            tag=branch,
            dockerfile="Dockerfile",
        ),
        name="pull_coindesk_news_content",
        work_pool_name="kubernetes-default",
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
        schedule=Interval(
            dt.timedelta(hours=1), anchor_date=dt.datetime(2000, 1, 1, 0, 0, 0)
        ),
        build=False,
        push=False,
    )
