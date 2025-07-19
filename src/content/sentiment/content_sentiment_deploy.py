import os
import sys
import datetime as dt

# Ensure the parent directory is in the Python path.
sys.path.append(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

from prefect import flow
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyLimitStrategy,
)
from prefect.docker import DockerImage
from prefect.runner.storage import GitRepository
from prefect.events import DeploymentEventTrigger
from prefect_github import GitHubCredentials


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
        entrypoint="src/content/sentiment/content_sentiment_flows.py:refresh_content_sentiment",
    ).deploy(  # type: ignore
        image=DockerImage(
            name="ghcr.io/manning-capital/mc-prefect-loaders",
            tag=branch,
            dockerfile="Dockerfile",
        ),
        name="refresh_content_sentiment",
        work_pool_name="kubernetes-default",
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
        triggers=[
             DeploymentEventTrigger(
                name="Refresh content sentiment after coindesk content is refreshed",
                description="This trigger will refresh the content sentiment after the coindesk content is refreshed.",
                enabled=True,
                match={"prefect.resource.id": "prefect.flow-run.*"}, # type: ignore
                expect={"prefect.flow-run.Completed"},
                match_related={
                    "prefect.resource.name": "pull_coindesk_news_content",
                    "prefect.resource.role": "deployment"
                }, # type: ignore
                for_each=[
                    "prefect.resource.id"
                ], # type: ignore
            )
        ], # type: ignore
        build=False,
        push=False,
    )