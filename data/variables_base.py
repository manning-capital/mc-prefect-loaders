import os

from prefect import get_client
from prefect.variables import Variable
from prefect.exceptions import ObjectNotFound
from prefect.client.schemas.actions import (
    GlobalConcurrencyLimitCreate,
    GlobalConcurrencyLimitUpdate,
)


def does_global_concurrency_limit_exist(name: str) -> bool:
    """
    Check if a global concurrency limit with the given name exists.
    """
    with get_client(sync_client=True) as client:
        try:
            client.read_global_concurrency_limit_by_name(name=name)
            return True
        except ObjectNotFound:
            return False


def create_global_concurrency_limit(name: str, limit: int, slot_decay_per_second: int):
    """
    Create a global concurrency limit for the Kraken API.
    """
    with get_client(sync_client=True) as client:
        if does_global_concurrency_limit_exist(name):
            print(f"Updating existing global concurrency limit '{name}'.")
            client.update_global_concurrency_limit(
                name=name,
                concurrency_limit=GlobalConcurrencyLimitUpdate(
                    name=name,
                    limit=limit,
                    active=True,
                    slot_decay_per_second=slot_decay_per_second,
                ),
            )
        else:
            print(f"Creating new global concurrency limit '{name}'.")
            client.create_global_concurrency_limit(
                GlobalConcurrencyLimitCreate(
                    name=name,
                    limit=limit,
                    active=True,
                    slot_decay_per_second=slot_decay_per_second,
                )
            )


if __name__ == "__main__":
    Variable.set(
        name="github-branch",
        value=os.getenv("GITHUB_BRANCH"),
        overwrite=True,
    )
    create_global_concurrency_limit(
        name="kraken-api",
        limit=1,
        slot_decay_per_second=1,
    )
    create_global_concurrency_limit(
        name="coindesk-api",
        limit=1,
        slot_decay_per_second=1,
    )
