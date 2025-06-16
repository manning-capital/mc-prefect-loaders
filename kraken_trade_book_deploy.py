from prefect import flow
from prefect.docker import DockerImage
from prefect_github import GitHubCredentials
from prefect.runner.storage import LocalStorage
from prefect.runner.storage import GitRepository
from kraken_trade_book_flows import pull_kraken_trade_book


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
            name="glynfinck/sentiment",
            tag="latest",
            dockerfile="Dockerfile",
        ),
        name="pull_kraken_trade_book",
        work_pool_name="kubernetes-default",
        build=False,
        push=False,
    )
    # pull_kraken_trade_book_deployment = pull_kraken_trade_book.deploy(
    #     name="pull_kraken_trade_book_xbtusd",
    #     parameters={"pair": "XBTUSD", "count": 500},  # Default parameters
    #     image=DockerImage(
    #         name="glynfinck/sentiment",
    #         tag="latest",
    #         dockerfile="Dockerfile",
    #     ),
    #     work_pool_name="kube-default",
    # )
