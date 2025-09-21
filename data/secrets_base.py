import os

from prefect_github import GitHubCredentials
from prefect.blocks.system import Secret

if __name__ == "__main__":
    GitHubCredentials(token=os.getenv("PREFECT_GITHUB_CREDENTIALS")).save(  # type: ignore
        "github-credentials", overwrite=True
    )
    Secret(value=os.getenv("POSTGRES_URL")).save("postgres-url", overwrite=True)  # type: ignore
