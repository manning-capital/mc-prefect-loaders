import os
import sys
import datetime
from typing import List, Tuple, Optional

# Ensure the parent directory is in the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Provider,
    Asset,
    ProviderAssetOrder,
    AssetType,
    ProviderType,
    ProviderAsset,
    ProviderAssetMarket,
)


# Extend MockDatabase with CRUD and query methods
class MockDatabase:
    """
    A mock database that simulates database operations for testing purposes.
    """

    def __init__(self):
        self.providers = {}
        self.assets = {}
        self.provider_asset_orders = {}
        self.asset_types = {}
        self.provider_types = {}
        self.provider_assets = {}
        self.provider_asset_markets = {}
        self._provider_id_counter = 1
        self._asset_id_counter = 1
        self._provider_asset_order_id_counter = 1
        self._asset_type_id_counter = 1
        self._provider_type_id_counter = 1

    # AssetType operations
    def add_asset_type(self, asset_type: AssetType) -> int:
        if not hasattr(asset_type, "id") or asset_type.id is None:
            asset_type.id = self._asset_type_id_counter
            self._asset_type_id_counter += 1
        if not hasattr(asset_type, "created_at") or asset_type.created_at is None:
            asset_type.created_at = datetime.datetime.now()
        if not hasattr(asset_type, "updated_at") or asset_type.updated_at is None:
            asset_type.updated_at = datetime.datetime.now()
        self.asset_types[asset_type.id] = asset_type
        return asset_type.id

    def get_asset_type(self, asset_type_id: int) -> Optional[AssetType]:
        return self.asset_types.get(asset_type_id)

    def get_asset_types(self, **filters) -> List[AssetType]:
        result = []
        for asset_type in self.asset_types.values():
            matches = True
            for key, value in filters.items():
                if getattr(asset_type, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(asset_type)
        return result

    # ProviderType operations
    def add_provider_type(self, provider_type: ProviderType) -> int:
        if not hasattr(provider_type, "id") or provider_type.id is None:
            provider_type.id = self._provider_type_id_counter
            self._provider_type_id_counter += 1
        if not hasattr(provider_type, "created_at") or provider_type.created_at is None:
            provider_type.created_at = datetime.datetime.now()
        if not hasattr(provider_type, "updated_at") or provider_type.updated_at is None:
            provider_type.updated_at = datetime.datetime.now()
        self.provider_types[provider_type.id] = provider_type
        return provider_type.id

    def get_provider_type(self, provider_type_id: int) -> Optional[ProviderType]:
        return self.provider_types.get(provider_type_id)

    def get_provider_types(self, **filters) -> List[ProviderType]:
        result = []
        for provider_type in self.provider_types.values():
            matches = True
            for key, value in filters.items():
                if getattr(provider_type, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(provider_type)
        return result

    # Provider operations
    def add_provider(self, provider: Provider) -> int:
        if not hasattr(provider, "id") or provider.id is None:
            provider.id = self._provider_id_counter
            self._provider_id_counter += 1
        if not hasattr(provider, "created_at") or provider.created_at is None:
            provider.created_at = datetime.datetime.now()
        if not hasattr(provider, "updated_at") or provider.updated_at is None:
            provider.updated_at = datetime.datetime.now()
        self.providers[provider.id] = provider
        return provider.id

    def get_provider(self, provider_id: int) -> Optional[Provider]:
        return self.providers.get(provider_id)

    def get_providers(self, **filters) -> List[Provider]:
        result = []
        for provider in self.providers.values():
            matches = True
            for key, value in filters.items():
                if getattr(provider, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(provider)
        return result

    def update_provider(self, provider_id: int, **updates) -> bool:
        provider = self.get_provider(provider_id)
        if provider:
            for key, value in updates.items():
                setattr(provider, key, value)
            return True
        return False

    def delete_provider(self, provider_id: int) -> bool:
        if provider_id in self.providers:
            del self.providers[provider_id]
            return True
        return False

    # Asset operations
    def add_asset(self, asset: Asset) -> int:
        if not hasattr(asset, "id") or asset.id is None:
            asset.id = self._asset_id_counter
            self._asset_id_counter += 1
        if not hasattr(asset, "created_at") or asset.created_at is None:
            asset.created_at = datetime.datetime.now()
        if not hasattr(asset, "updated_at") or asset.updated_at is None:
            asset.updated_at = datetime.datetime.now()
        self.assets[asset.id] = asset
        return asset.id

    def get_asset(self, asset_id: int) -> Optional[Asset]:
        return self.assets.get(asset_id)

    def get_assets(self, **filters) -> List[Asset]:
        result = []
        for asset in self.assets.values():
            matches = True
            for key, value in filters.items():
                if getattr(asset, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(asset)
        return result

    def update_asset(self, asset_id: int, **updates) -> bool:
        asset = self.get_asset(asset_id)
        if asset:
            for key, value in updates.items():
                setattr(asset, key, value)
            return True
        return False

    def delete_asset(self, asset_id: int) -> bool:
        if asset_id in self.assets:
            del self.assets[asset_id]
            return True
        return False

    # ProviderAssetOrder operations
    def add_provider_asset_order(self, provider_asset_order: ProviderAssetOrder) -> int:
        if not hasattr(provider_asset_order, "id") or provider_asset_order.id is None:
            provider_asset_order.id = self._provider_asset_order_id_counter
            self._provider_asset_order_id_counter += 1
        self.provider_asset_orders[provider_asset_order.id] = provider_asset_order
        return provider_asset_order.id

    def get_provider_asset_order(
        self, provider_asset_order_id: int
    ) -> Optional[ProviderAssetOrder]:
        return self.provider_asset_orders.get(provider_asset_order_id)

    def get_provider_asset_orders(self, **filters) -> List[ProviderAssetOrder]:
        result = []
        for provider_asset_order in self.provider_asset_orders.values():
            matches = True
            for key, value in filters.items():
                if getattr(provider_asset_order, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(provider_asset_order)
        return result

    def update_provider_asset_order(
        self, provider_asset_order_id: int, **updates
    ) -> bool:
        provider_asset_order = self.get_provider_asset_order(provider_asset_order_id)
        if provider_asset_order:
            for key, value in updates.items():
                setattr(provider_asset_order, key, value)
            return True
        return False

    def delete_provider_asset_order(self, provider_asset_order_id: int) -> bool:
        if provider_asset_order_id in self.provider_asset_orders:
            del self.provider_asset_orders[provider_asset_order_id]
            return True
        return False

    # ProviderAsset operations
    def add_provider_asset(
        self, provider_asset: ProviderAsset
    ) -> Tuple[datetime.date, int, int]:
        key = (provider_asset.date, provider_asset.provider_id, provider_asset.asset_id)
        if (
            not hasattr(provider_asset, "created_at")
            or provider_asset.created_at is None
        ):
            provider_asset.created_at = datetime.datetime.now()
        if (
            not hasattr(provider_asset, "updated_at")
            or provider_asset.updated_at is None
        ):
            provider_asset.updated_at = datetime.datetime.now()
        self.provider_assets[key] = provider_asset
        return key

    def get_provider_asset(
        self, date: datetime.date, provider_id: int, asset_id: int
    ) -> Optional[ProviderAsset]:
        key = (date, provider_id, asset_id)
        return self.provider_assets.get(key)

    def get_provider_assets(self, **filters) -> List[ProviderAsset]:
        result = []
        for provider_asset in self.provider_assets.values():
            matches = True
            for key, value in filters.items():
                if getattr(provider_asset, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(provider_asset)
        return result

    # ProviderAssetMarket operations
    def add_provider_asset_market(
        self, provider_asset_market: ProviderAssetMarket
    ) -> Tuple[datetime.datetime, int, int]:
        key = (
            provider_asset_market.timestamp,
            provider_asset_market.provider_id,
            provider_asset_market.asset_id,
        )
        self.provider_asset_markets[key] = provider_asset_market
        return key

    def get_provider_asset_market(
        self, timestamp: datetime.datetime, provider_id: int, asset_id: int
    ) -> Optional[ProviderAssetMarket]:
        key = (timestamp, provider_id, asset_id)
        return self.provider_asset_markets.get(key)

    def get_provider_asset_markets(self, **filters) -> List[ProviderAssetMarket]:
        result = []
        for provider_asset_market in self.provider_asset_markets.values():
            matches = True
            for key, value in filters.items():
                if getattr(provider_asset_market, key, None) != value:
                    matches = False
                    break
            if matches:
                result.append(provider_asset_market)
        return result
