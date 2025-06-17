from prefect import flow
from prefect.docker import DockerImage
from prefect_github import GitHubCredentials
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval
from datetime import datetime

if __name__ == "__main__":
    source = GitRepository(
        url="https://github.com/glynfinck/sentiment",
        credentials=GitHubCredentials.load("github-credentials"),
        branch="main",
    )
    flow.from_source(
        source=source, entrypoint="kraken_trade_book_flows.py:pull_kraken_trade_book"
    ).deploy(
        image=DockerImage(
            name="glynfinck/sentiment", tag="latest", dockerfile="Dockerfile"
        ),
        name="pull_kraken_trade_book_xbtusd",
        work_pool_name="kubernetes-default",
        parameters={"pairs": ["XBTUSD", "XBTEUR"], "count": 500},  # Default parameters
        interval=Interval(60, anchor_date=datetime(2000, 1, 1, 0, 0, 0)),
        build=False,
        push=False,
    )
