import os
from prefect.variables import Variable

if __name__ == "__main__":
    Variable.set(
        name="github-branch",
        value=os.getenv("GITHUB_BRANCH"),
        overwrite=True,
    )
