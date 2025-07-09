import os
from prefect.blocks.system import Secret

if __name__ == "__main__":
    Secret(value=os.getenv("GITHUB_CREDENTIALS")).save(
        "github-credentials", overwrite=True
    )
