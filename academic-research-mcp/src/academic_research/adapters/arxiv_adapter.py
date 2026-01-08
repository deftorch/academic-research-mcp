from typing import List, Optional
import feedparser

from ..ports.search_port import BaseSearchProvider
from ..domain.models import Paper
from ..utils import (
    ARXIV_API,
    cache_key,
    clean_text,
    logger,
    make_api_request,
    ProviderError
)

class ArxivProvider(BaseSearchProvider):
    name = "arXiv"

    async def search(self, query: str, max_results: int, sort_by: str = "relevance", **kwargs) -> List[Paper]:
        cache_k = cache_key("arxiv", query, max_results, sort_by)
        cached = self._get_cache(cache_k)
        if cached:
            return cached

        logger.info(f"Searching {self.name}: '{query}'")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        try:
            # We bypass make_api_request here because Arxiv returns XML
            response_data = await make_api_request(ARXIV_API, params=params)

            text_content = response_data.get("text") if isinstance(response_data, dict) else response_data
            if not text_content:
                return []

            feed = feedparser.parse(text_content)

            papers = []
            for entry in feed.entries:
                arxiv_id = entry.id.split("/abs/")[-1] if "/abs/" in entry.id else entry.id.split("/")[-1]

                # Check for published date
                published = entry.published if hasattr(entry, "published") else None
                year = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    year = entry.published_parsed.tm_year

                paper = Paper(
                    source=self.name,
                    title=clean_text(entry.title),
                    authors=[author.name for author in entry.authors] if hasattr(entry, "authors") else [],
                    abstract=clean_text(entry.summary) if hasattr(entry, "summary") else None,
                    pdf_url=entry.id.replace("/abs/", "/pdf/") + ".pdf",
                    url=entry.id,
                    published=published,
                    year=year,
                    arxiv_id=arxiv_id,
                    categories=[tag.term for tag in entry.tags] if hasattr(entry, "tags") else [],
                    paper_id=arxiv_id # Use arxiv_id as paper_id for consistency
                )
                papers.append(paper)

            self._set_cache(cache_k, papers)
            logger.info(f"Found {len(papers)} papers from {self.name}")
            return papers
        except Exception as e:
            logger.error(f"{self.name} search failed: {str(e)}")
            raise ProviderError(f"Failed to search {self.name}: {e}") from e
