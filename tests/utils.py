import mc_postgres_db.models as models
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session


async def sample_provider_data(
    engine: Engine,
) -> tuple[models.ProviderType, models.Provider]:
    # Create the exchange provider type.
    with Session(engine) as session:
        exchange_provider_type = models.ProviderType(
            name="Exchange", description="Exchange", is_active=True
        )
        session.add(exchange_provider_type)
        session.commit()
        session.refresh(exchange_provider_type)

    # Create the exchange provider.
    with Session(engine) as session:
        kraken_provider = models.Provider(
            name="Kraken",
            description="Kraken",
            provider_type_id=exchange_provider_type.id,
            is_active=True,
        )
        session.add(kraken_provider)
        session.commit()
        session.refresh(kraken_provider)

    return exchange_provider_type, kraken_provider


async def sample_asset_data(
    engine: Engine,
) -> tuple[models.ProviderType, models.AssetType, models.AssetType]:
    # Create the crypto asset type.
    with Session(engine) as session:
        crypto_asset_type = models.AssetType(
            name="CryptoCurrency", description="CryptoCurrency", is_active=True
        )
        session.add(crypto_asset_type)
        session.commit()
        session.refresh(crypto_asset_type)

    # Create the fiat asset type.
    with Session(engine) as session:
        fiat_asset_type = models.AssetType(
            name="FiatCurrency", description="FiatCurrency", is_active=True
        )
        session.add(fiat_asset_type)
        session.commit()
        session.refresh(fiat_asset_type)

    # Create BTC asset.
    with Session(engine) as session:
        btc_asset = models.Asset(
            name="BTC", description="BTC", asset_type_id=crypto_asset_type.id
        )
        session.add(btc_asset)
        session.commit()
        session.refresh(btc_asset)

    # Create ETH asset.
    with Session(engine) as session:
        eth_asset = models.Asset(
            name="ETH", description="ETH", asset_type_id=crypto_asset_type.id
        )
        session.add(eth_asset)
        session.commit()
        session.refresh(eth_asset)

    # Create USD asset.
    with Session(engine) as session:
        usd_asset = models.Asset(
            name="USD", description="USD", asset_type_id=fiat_asset_type.id
        )
        session.add(usd_asset)
        session.commit()
        session.refresh(usd_asset)

    return crypto_asset_type, fiat_asset_type, btc_asset, eth_asset, usd_asset


async def create_base_data(
    engine: Engine,
) -> tuple[
    models.ProviderType, models.Provider, models.ContentType, models.SentimentType
]:
    # Create the provider type.
    with Session(engine) as session:
        provider_type = models.ProviderType(
            name="NEWS_PROVIDER", description="News provider", is_active=True
        )
        session.add(provider_type)
        session.commit()

        # Create the provider.
        provider = models.Provider(
            name="Coindesk", provider_type_id=provider_type.id, is_active=True
        )
        session.add(provider)
        session.commit()

        # Create the content type.
        content_type = models.ContentType(
            name="NEWS", description="News", is_active=True
        )
        session.add(content_type)
        session.commit()

        # Create the sentiment type.
        sentiment_type = models.SentimentType(
            name="NLTKVader", description="NLTKVader", is_active=True
        )
        session.add(sentiment_type)
        session.commit()

        # Get the ORM objects for all of the above.
        provider_type = session.execute(
            select(models.ProviderType).where(
                models.ProviderType.name == "NEWS_PROVIDER"
            )
        ).scalar_one()
        provider = session.execute(
            select(models.Provider).where(models.Provider.name == "Coindesk")
        ).scalar_one()
        content_type = session.execute(
            select(models.ContentType).where(models.ContentType.name == "NEWS")
        ).scalar_one()
        sentiment_type = session.execute(
            select(models.SentimentType).where(models.SentimentType.name == "NLTKVader")
        ).scalar_one()

        return provider_type, provider, content_type, sentiment_type
