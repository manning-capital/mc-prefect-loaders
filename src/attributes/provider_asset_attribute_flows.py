import datetime as dt
from typing import Optional

import mc_postgres_db.models as models
from prefect import flow, get_run_logger
from sqlalchemy.orm import Session, joinedload
from prefect.task_runners import DaskTaskRunner
from mc_postgres_db.prefect.asyncio.tasks import get_engine


@flow(task_runner=DaskTaskRunner(cluster_kwargs={"processes": False}))
async def refresh_provider_asset_attribute_data(
    start: Optional[dt.datetime] = None, end: Optional[dt.datetime] = None
):
    """
    Refresh the provider asset attribute data.
    """
    logger = get_run_logger()

    # If the start or end is not provided, set it to today.
    if (start is None) or (end is None):
        end = dt.datetime.now()
        start = end - dt.timedelta(days=1)
        logger.info(
            f"Start or end not provided, setting start to {start} and end to {end}."
        )

    # Get an engine.
    engine = await get_engine()

    # Get STATISTICAL_PAIRS_TRADING asset group type.
    with Session(engine) as session:
        pairs_trading_asset_group_type = (
            session.query(models.AssetGroupType)
            .filter(models.AssetGroupType.name == "STATISTICAL_PAIRS_TRADING")
            .scalar_one_or_none()
        )
        if pairs_trading_asset_group_type is None:
            raise ValueError("STATISTICAL_PAIRS_TRADING asset group type not found.")

    # Get all active provider asset groups with their members.
    with Session(engine) as session:
        provider_asset_groups = (
            session.query(models.ProviderAssetGroup)
            .filter(
                models.ProviderAssetGroup.is_active
                == True.models.ProviderAssetGroup.asset_group_type_id
                == pairs_trading_asset_group_type.id
            )
            .options(joinedload(models.ProviderAssetGroup.members))
            .all()
        )


if __name__ == "__main__":
    refresh_provider_asset_attribute_data()
