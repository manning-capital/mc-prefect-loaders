import math
import datetime as dt
import itertools

import numpy as np
import mc_postgres_db.models as models
from prefect import flow, task, get_run_logger
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased, joinedload
from mc_postgres_db.prefect.asyncio.tasks import get_engine


def convert_provider_asset_groups_to_tuples(
    provider_asset_groups: list[models.ProviderAssetGroup],
) -> set[tuple[tuple[int, int, int], ...]]:
    """
    Convert the provider asset groups to a set of tuples of (provider_id, from_asset_id, to_asset_id).

    Args:
        provider_asset_groups: List of provider asset groups to convert.

    Returns:
        Set of tuples representing provider asset groups.
    """
    return {
        tuple(
            sorted(
                [
                    (member.provider_id, member.from_asset_id, member.to_asset_id)
                    for member in provider_asset_group.members
                ],
                key=lambda x: x,
            )
        )
        for provider_asset_group in provider_asset_groups
    }


@task()
async def get_desired_provider_asset_groups(
    start_date: dt.date, end_date: dt.date, provider_ids: list[int]
) -> set[models.ProviderAssetGroup]:
    """
    Get the new provider asset groups based on the provider asset market data in the database.
    """
    logger = get_run_logger()

    # Get the engine.
    engine = await get_engine()

    with Session(engine) as session:
        # Get the asset group type for STATISTICAL_PAIRS_TRADING
        asset_group_type = session.execute(
            select(models.AssetGroupType).where(
                models.AssetGroupType.symbol == "STATISTICAL_PAIRS_TRADING"
            )
        ).scalar_one()

        # Get the distinct provider asset market pairs in the given date range.
        provider = aliased(models.Provider)
        from_asset = aliased(models.Asset)
        to_asset = aliased(models.Asset)
        provider_asset_market_group_members = set(
            session.execute(
                select(
                    provider,
                    from_asset,
                    to_asset,
                )
                .where(
                    models.ProviderAssetMarket.provider_id.in_(provider_ids),
                    models.ProviderAssetMarket.timestamp >= start_date,
                    models.ProviderAssetMarket.timestamp <= end_date,
                )
                .select_from(models.ProviderAssetMarket)
                .join(
                    provider,
                    models.ProviderAssetMarket.provider_id == provider.id,
                )
                .join(
                    from_asset,
                    models.ProviderAssetMarket.from_asset_id == from_asset.id,
                )
                .join(
                    to_asset,
                    models.ProviderAssetMarket.to_asset_id == to_asset.id,
                )
                .distinct()
            ).tuples()
        )

        # Check the number of combinations. This number will overflow if it is greater than 135805301026.
        n_combinations: np.int64 = (
            np.int64(math.comb(len(provider_asset_market_group_members), 2))
            if len(provider_asset_market_group_members) < 135805301026
            else np.inf
        )

        # Group members by from_asset to ensure we only pair assets with the same from_asset
        # This allows pairs to form across different providers as long as they have the same from_asset
        grouped_members = {}
        for provider, from_asset, to_asset in provider_asset_market_group_members:
            if from_asset not in grouped_members:
                grouped_members[from_asset] = []
            grouped_members[from_asset].append((provider, from_asset, to_asset))

        # Generate combinations only within each group (same from_asset)
        all_combinations = []
        for group_members in grouped_members.values():
            if len(group_members) >= 2:  # Need at least 2 members to form a pair
                group_combinations = itertools.combinations(group_members, 2)
                all_combinations.extend(group_combinations)

        # Limit the combinations to the maximum allowed if needed
        maximum_provider_asset_market_pairs = 15000
        if len(all_combinations) > maximum_provider_asset_market_pairs:
            all_combinations = all_combinations[:maximum_provider_asset_market_pairs]

        return set(
            models.ProviderAssetGroup(
                asset_group_type_id=asset_group_type.id,
                is_active=False,
                members=[
                    models.ProviderAssetGroupMember(
                        provider_id=member[0].id,
                        provider=member[0],
                        from_asset_id=member[1].id,
                        from_asset=member[1],
                        to_asset_id=member[2].id,
                        to_asset=member[2],
                        order=i + 1,
                    )
                    for i, member in enumerate(combination)
                ],
            )
            for combination in all_combinations
        )


@task()
async def get_current_pairs_trading_groups() -> list[models.ProviderAssetGroup]:
    """
    Get the current pairs trading provider asset groups from the database.

    Returns:
        List of current provider asset groups (both active and inactive).
    """
    logger = get_run_logger()
    engine = await get_engine()

    with Session(engine) as session:
        # Get the asset group type for STATISTICAL_PAIRS_TRADING
        logger.info("Getting STATISTICAL_PAIRS_TRADING asset group type...")
        asset_group_type = session.execute(
            select(models.AssetGroupType).where(
                models.AssetGroupType.symbol == "STATISTICAL_PAIRS_TRADING"
            )
        ).scalar_one()

        # Get current provider asset groups (all of them, including inactive)
        logger.info("Getting current provider asset groups from database...")
        current_provider_asset_groups = (
            session.query(models.ProviderAssetGroup)
            .filter(
                models.ProviderAssetGroup.asset_group_type_id == asset_group_type.id,
            )
            .options(
                joinedload(models.ProviderAssetGroup.members).joinedload(
                    models.ProviderAssetGroupMember.provider
                )
            )
            .options(
                joinedload(models.ProviderAssetGroup.members).joinedload(
                    models.ProviderAssetGroupMember.from_asset
                )
            )
            .options(
                joinedload(models.ProviderAssetGroup.members).joinedload(
                    models.ProviderAssetGroupMember.to_asset
                )
            )
            .all()
        )

        logger.info(
            f"Found {len(current_provider_asset_groups)} existing provider asset groups."
        )

        return current_provider_asset_groups


@flow()
async def refresh_pairs_trading_groups(
    start_date: dt.date,
    end_date: dt.date,
):
    """
    Refresh the pairs trading provider asset groups by creating new groups that don't already exist.
    New groups are created as inactive (is_active=False) by default.

    Args:
        start_date: The start date to query provider asset market data from.
        end_date: The end date to query provider asset market data from.
    """
    logger = get_run_logger()

    # Get desired and current provider asset groups
    desired_combinations = await get_desired_pairs_trading_groups(
        start_date=start_date,
        end_date=end_date,
    )
    current_provider_asset_groups = await get_current_pairs_trading_groups()

    # Convert desired combinations to tuple representation for comparison
    desired_provider_asset_tuples = set()
    for combination in desired_combinations:
        tuple_repr = tuple(
            sorted(
                [(member[0].id, member[1].id, member[2].id) for member in combination],
                key=lambda x: x,
            )
        )
        desired_provider_asset_tuples.add(tuple_repr)

    logger.info(
        f"Created {len(desired_provider_asset_tuples)} unique desired provider asset tuples."
    )

    # Convert current groups to tuple representation
    current_provider_asset_tuples = convert_provider_asset_groups_to_tuples(
        current_provider_asset_groups
    )

    # Get the new provider asset groups
    new_provider_asset_tuples = (
        desired_provider_asset_tuples - current_provider_asset_tuples
    )

    logger.info(
        f"Found {len(new_provider_asset_tuples)} new provider asset groups to create."
    )

    # Create the new provider asset groups with is_active=False
    if len(new_provider_asset_tuples) > 0:
        engine = await get_engine()
        with Session(engine) as session:
            # Get the asset group type for STATISTICAL_PAIRS_TRADING
            asset_group_type = session.execute(
                select(models.AssetGroupType).where(
                    models.AssetGroupType.symbol == "STATISTICAL_PAIRS_TRADING"
                )
            ).scalar_one()

            for provider_asset_tuple in new_provider_asset_tuples:
                provider_asset_group_members = [
                    models.ProviderAssetGroupMember(
                        provider_id=provider_asset_group_member_tuple[0],
                        from_asset_id=provider_asset_group_member_tuple[1],
                        to_asset_id=provider_asset_group_member_tuple[2],
                        order=i + 1,
                    )
                    for i, provider_asset_group_member_tuple in enumerate(
                        provider_asset_tuple
                    )
                ]
                session.add(
                    models.ProviderAssetGroup(
                        asset_group_type_id=asset_group_type.id,
                        is_active=False,
                        members=provider_asset_group_members,
                    )
                )
            session.commit()
            logger.info(
                f"Successfully created {len(new_provider_asset_tuples)} new inactive provider asset groups."
            )
    else:
        logger.info("No new provider asset groups to create.")
