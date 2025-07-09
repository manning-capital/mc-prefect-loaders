import os
from prefect.blocks.system import Secret
from prefect_github import GitHubCredentials

if __name__ == "__main__":
    GitHubCredentials(token=os.getenv("PREFECT_GITHUB_CREDENTIALS")).save(
        "github-credentials", overwrite=True
    )
