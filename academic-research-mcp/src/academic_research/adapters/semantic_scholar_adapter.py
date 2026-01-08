from typing import List, Optional

from ..ports.search_port import BaseSearchProvider
from ..domain.models import Paper
from ..utils import (
    SEMANTIC_SCHOLAR_API,
    cache_key,
    clean_text,
    logger,
    make_api_request,
    ProviderError
)

class SemanticScholarProvider(BaseSearchProvider):
    name = "Semantic Scholar"

    async def search(
        self,
        query: str,
        max_results: int,
        year_filter: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        **kwargs,
    ) -> List[Paper]:
        cache_k = cache_key("semantic", query, max_results, year_filter)
        cached = self._get_cache(cache_k)
        if cached:
            return cached

        logger.info(f"Searching {self.name}: '{query}'")

        url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "paperId,title,authors,abstract,year,citationCount,influentialCitationCount,venue,openAccessPdf,externalIds",
        }

        if year_filter:
            params["year"] = year_filter
        if min_citation_count:
            params["minCitationCount"] = min_citation_count

        try:
            data = await make_api_request(url, params=params)

            papers = []
            for item in data.get("data", []):
                paper = Paper(
                    source=self.name,
                    paper_id=item.get("paperId"),
                    title=clean_text(item.get("title", "")),
                    authors=[a.get("name", "Unknown") for a in item.get("authors", [])],
                    abstract=clean_text(item.get("abstract", "")),
                    year=item.get("year"),
                    citation_count=item.get("citationCount", 0),
                    influential_citation_count=item.get("influentialCitationCount", 0),
                    venue=item.get("venue", ""),
                    pdf_url=item.get("openAccessPdf", {}).get("url") if item.get("openAccessPdf") else None,
                    doi=item.get("externalIds", {}).get("DOI"),
                    arxiv_id=item.get("externalIds", {}).get("ArXiv"),
                    url=f"https://www.semanticscholar.org/paper/{item.get('paperId')}" if item.get("paperId") else None,
                )
                papers.append(paper)

            self._set_cache(cache_k, papers)
            return papers
        except Exception as e:
            logger.error(f"{self.name} search failed: {str(e)}")
            raise ProviderError(f"Failed to search {self.name}: {e}") from e
