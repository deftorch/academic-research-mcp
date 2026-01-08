from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.models import Paper
from ..utils import get_from_cache, set_to_cache

class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider."""
        pass

    @abstractmethod
    async def search(self, query: str, max_results: int, **kwargs) -> List[Paper]:
        """Search method to be implemented by providers."""
        pass

    def _get_cache(self, key: str) -> Optional[List[Paper]]:
        cached_data = get_from_cache(key)
        if cached_data:
            return [Paper(**item) for item in cached_data]
        return None

    def _set_cache(self, key: str, data: List[Paper]):
        set_to_cache(key, [p.model_dump(mode='json') for p in data])
