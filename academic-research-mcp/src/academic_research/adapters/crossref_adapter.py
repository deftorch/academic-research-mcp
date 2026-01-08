from typing import List, Optional

from ..ports.search_port import BaseSearchProvider
from ..domain.models import Paper
from ..utils import (
    CROSSREF_API,
    CONTACT_EMAIL,
    cache_key,
    clean_text,
    logger,
    make_api_request,
    ProviderError
)

class CrossRefProvider(BaseSearchProvider):
    name = "CrossRef"

    async def search(self, query: str, max_results: int, from_year: Optional[int] = None, **kwargs) -> List[Paper]:
        cache_k = cache_key("crossref", query, max_results)
        cached = self._get_cache(cache_k)
        if cached:
            return cached

        params = {"query": query, "rows": min(max_results, 100), "mailto": CONTACT_EMAIL}

        try:
            data = await make_api_request(CROSSREF_API, params=params)
            papers = []
            for item in data.get("message", {}).get("items", []):
                # CrossRef items structure varies
                title = item.get("title", [""])
                if isinstance(title, list) and len(title) > 0:
                    title = title[0]
                elif isinstance(title, str):
                    title = title
                else:
                    title = "Untitled"

                # Extract year
                published = item.get("published-print") or item.get("published-online")
                year = None
                if published and "date-parts" in published:
                    parts = published["date-parts"]
                    if parts and len(parts) > 0 and len(parts[0]) > 0:
                        year = parts[0][0]

                paper = Paper(
                    source=self.name,
                    title=clean_text(title),
                    doi=item.get("DOI"),
                    url=item.get("URL"),
                    citation_count=item.get("is-referenced-by-count", 0),
                    paper_id=item.get("DOI"), # Use DOI as ID
                    year=year
                )
                papers.append(paper)
            self._set_cache(cache_k, papers)
            return papers
        except Exception as e:
            logger.error(f"{self.name} search failed: {str(e)}")
            raise ProviderError(f"Failed to search {self.name}: {e}") from e
