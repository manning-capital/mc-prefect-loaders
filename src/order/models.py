import datetime
from typing import Optional

from sqlalchemy import Engine, ForeignKey, String, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy models.
    This class is used to define the base for all models in the application.
    It inherits from DeclarativeBase, which is a SQLAlchemy class that provides
    a declarative interface for defining models.
    """

    pass


class AssetType(Base):
    __tablename__ = "asset_type"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        nullable=False, server_onupdate=func.now(), server_default=func.now()
    )

    def __repr__(self):
        return f"{AssetType.__name__}({self.id}, {self.name})"


class Asset(Base):
    __tablename__ = "asset"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_type_id: Mapped[int] = mapped_column(
        ForeignKey("asset_type.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[Optional[str]]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        nullable=False, server_onupdate=func.now(), server_default=func.now()
    )

    def __repr__(self):
        return f"{Asset.__name__}({self.id}, {self.name})"


class ProviderType(Base):
    __tablename__ = "provider_type"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        nullable=False, server_onupdate=func.now(), server_default=func.now()
    )

    def __repr__(self):
        return f"{ProviderType.__name__}({self.id}, {self.name})"


class Provider(Base):
    __tablename__ = "provider"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_type_id: Mapped[int] = mapped_column(
        ForeignKey("provider_type.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        nullable=False, server_onupdate=func.now(), server_default=func.now()
    )

    def __repr__(self):
        return f"{Provider.__name__}({self.id}, {self.name})"

    def get_all_assets(
        self,
        engine: Engine,
        asset_ids: list[int] = [],
    ) -> set["ProviderAsset"]:
        with Session(engine) as session:
            # Subquery to get the latest date for each provider_id, asset_id combination
            latest_dates_subq = (
                select(
                    ProviderAsset.provider_id,
                    ProviderAsset.asset_id,
                    func.max(ProviderAsset.date).label("max_date"),
                )
                .where(ProviderAsset.provider_id == self.id, ProviderAsset.is_active)
                .group_by(ProviderAsset.provider_id, ProviderAsset.asset_id)
                .subquery()
            )

            # Query to get assets that have provider_asset entries with the latest dates
            query = (
                select(ProviderAsset)
                .join(Asset, ProviderAsset.asset_id == Asset.id)
                .join(
                    latest_dates_subq,
                    (ProviderAsset.provider_id == latest_dates_subq.c.provider_id)
                    & (ProviderAsset.asset_id == latest_dates_subq.c.asset_id)
                    & (ProviderAsset.date == latest_dates_subq.c.max_date),
                )
                .where(
                    ProviderAsset.provider_id == self.id,
                    ProviderAsset.is_active,
                    Asset.is_active,
                )
            )

            # Add asset ID filter if provided
            if asset_ids:
                query = query.where(ProviderAsset.asset_id.in_(asset_ids))

            # Execute query and return results as a set
            assets = session.scalars(query).all()
            return set(assets)


class ProviderAsset(Base):
    __tablename__ = "provider_asset"

    date: Mapped[datetime.date] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("provider.id"), nullable=False, primary_key=True
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("asset.id"), nullable=False, primary_key=True
    )
    asset_code: Mapped[str] = mapped_column(String(30), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        nullable=False, server_onupdate=func.now(), server_default=func.now()
    )

    def __repr__(self):
        return f"{ProviderAsset.__name__}({self.date}, {self.provider_id}, {self.asset_id})"


class ProviderAssetOrder(Base):
    __tablename__ = "provider_asset_order"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(nullable=False)
    provider_id: Mapped[int] = mapped_column(ForeignKey("provider.id"), nullable=False)
    from_asset_id: Mapped[int] = mapped_column(ForeignKey("asset.id"), nullable=False)
    to_asset_id: Mapped[int] = mapped_column(ForeignKey("asset.id"), nullable=False)
    price: Mapped[float] = mapped_column(nullable=True)
    volume: Mapped[float] = mapped_column(nullable=True)

    def __repr__(self):
        return f"{ProviderAssetOrder.__name__}(id={self.id}, timestamp={self.timestamp}, provider_id={self.provider_id}, from_asset_id={self.from_asset_id}, to_asset_id={self.to_asset_id}, price={self.price}, volume={self.volume})"


class ProviderAssetMarket(Base):
    __tablename__ = "provider_asset_market"

    timestamp: Mapped[datetime.datetime] = mapped_column(
        nullable=False, primary_key=True
    )
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("provider.id"), nullable=False, primary_key=True
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("asset.id"), nullable=False, primary_key=True
    )
    close: Mapped[float] = mapped_column(nullable=True, comment="Closing price")
    open: Mapped[float] = mapped_column(nullable=True, comment="Opening price")
    high: Mapped[float] = mapped_column(nullable=True, comment="Highest price")
    low: Mapped[float] = mapped_column(nullable=True, comment="Lowest price")
    volume: Mapped[float] = mapped_column(nullable=True, comment="Volume traded")
    best_bid: Mapped[float] = mapped_column(nullable=True, comment="Best bid price")
    best_ask: Mapped[float] = mapped_column(nullable=True, comment="Best ask price")

    def __repr__(self):
        return f"{ProviderAssetMarket.__name__}(id={self.id}, provider_id={self.provider_id}, asset_id={self.asset_id}, market_type={self.market_type})"


if __name__ == "__main__":
    Base.metadata.create_all()  # Create all tables in the database
