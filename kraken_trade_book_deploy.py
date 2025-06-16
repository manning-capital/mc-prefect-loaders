from prefect.docker import DockerImage
from prefect import flow
from prefect.runner.storage import GitRepository
from prefect_github import GitHubCredentials

if __name__ == "__main__":
    source = GitRepository(
        url="https://github.com/org/private-repo.git",
        credentials=GitHubCredentials.load("my-github-credentials-block"),
        branch="main",
    )
    flow.from_source(source=source, entrypoint="my_file.py:my_flow").deploy(
         image=DockerImage(
            name="glynfinck/sentiment",
            tag="latest",
            dockerfile="Dockerfile",
        ),
        name="pull_kraken_trade_book",
        work_pool_name="kube-default",
        push=False,
        build=False,
    )
