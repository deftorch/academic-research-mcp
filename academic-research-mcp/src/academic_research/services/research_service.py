import asyncio
import time
from typing import Dict, List, Optional
from pathlib import Path
import httpx

from ..domain.models import Paper
from ..adapters.arxiv_adapter import ArxivProvider
from ..adapters.semantic_scholar_adapter import SemanticScholarProvider
from ..adapters.crossref_adapter import CrossRefProvider
from ..utils import (
    calculate_similarity,
    make_api_request,
    logger,
    CONTACT_EMAIL,
    UNPAYWALL_API,
    SEMANTIC_SCHOLAR_API,
    clean_text
)

# Initialize providers
arxiv_provider = ArxivProvider()
semantic_provider = SemanticScholarProvider()
crossref_provider = CrossRefProvider()

# Service functions (Business Logic)

async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Paper]:
    """Search arXiv for papers by keyword."""
    try:
        return await arxiv_provider.search(query, max_results, sort_by=sort_by)
    except Exception:
        return []

async def search_semantic_scholar(
    query: str, max_results: int = 10, year_filter: Optional[str] = None, min_citation_count: Optional[int] = None
) -> List[Paper]:
    """Search Semantic Scholar with rich metadata and citation data."""
    try:
        return await semantic_provider.search(
            query, max_results, year_filter=year_filter, min_citation_count=min_citation_count
        )
    except Exception:
        return []

async def search_crossref(query: str, max_results: int = 10, from_year: Optional[int] = None) -> List[Paper]:
    """Search CrossRef for published papers with DOI."""
    try:
        return await crossref_provider.search(query, max_results, from_year=from_year)
    except Exception:
        return []

async def search_pubmed(query: str, max_results: int = 10) -> List[Paper]:
    """Search PubMed for biomedical literature."""
    # PubMed implementation placeholder
    return []

async def search_all_sources(
    query: str, max_results_per_source: int = 5, sources: Optional[List[str]] = None
) -> Dict[str, List[Paper]]:
    """Search multiple databases concurrently."""
    if sources is None:
        sources = ["arxiv", "semantic_scholar", "crossref"]

    tasks = []
    source_map = {}

    if "arxiv" in sources:
        tasks.append(search_arxiv(query, max_results_per_source))
        source_map[len(tasks) - 1] = "arxiv"

    if "semantic_scholar" in sources:
        tasks.append(search_semantic_scholar(query, max_results_per_source))
        source_map[len(tasks) - 1] = "semantic_scholar"

    if "crossref" in sources:
        tasks.append(search_crossref(query, max_results_per_source))
        source_map[len(tasks) - 1] = "crossref"

    results = await asyncio.gather(*tasks, return_exceptions=True)

    organized_results = {}
    for idx, result in enumerate(results):
        source_name = source_map.get(idx, f"source_{idx}")
        if isinstance(result, Exception):
            organized_results[source_name] = []
        else:
            organized_results[source_name] = result or []

    return organized_results

async def search_by_author(author_name: str, max_results: int = 10) -> List[Paper]:
    """Search papers by specific author."""
    url = f"{SEMANTIC_SCHOLAR_API}/author/search"
    params = {"query": author_name, "limit": 1}

    try:
        data = await make_api_request(url, params=params)
        if not data.get("data"):
            return []

        author_id = data["data"][0].get("authorId")
        # Need to request more fields to populate Paper object properly
        papers_url = f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers"
        params_papers = {
            "limit": max_results,
            "fields": "paperId,title,year,citationCount,venue,externalIds"
        }
        papers_data = await make_api_request(papers_url, params=params_papers)

        papers = []
        for p in papers_data.get("data", []):
            paper = Paper(
                source="Semantic Scholar",
                paper_id=p.get("paperId"),
                title=clean_text(p.get("title", "")),
                year=p.get("year"),
                citation_count=p.get("citationCount", 0),
                venue=p.get("venue"),
                doi=p.get("externalIds", {}).get("DOI"),
                arxiv_id=p.get("externalIds", {}).get("ArXiv"),
                url=f"https://www.semanticscholar.org/paper/{p.get('paperId')}" if p.get("paperId") else None
            )
            papers.append(paper)

        return papers
    except Exception as e:
        logger.error(f"Author search failed: {e}")
        return []

async def deduplicate_papers(
    papers: List[Paper], similarity_threshold: float = 0.90, quick_dedup: bool = False
) -> List[Paper]:
    """
    Remove duplicate papers using fuzzy matching.
    """
    if not papers:
        return []

    unique_papers = []
    seen_titles = []

    # Pre-normalization helper
    def normalize(t):
        return t.lower().strip() if t else ""

    if quick_dedup:
        # O(N) deduplication using set of normalized titles
        seen_set = set()
        for paper in papers:
            title = paper.title
            norm_title = normalize(title)

            if not norm_title:
                continue

            if norm_title not in seen_set:
                seen_set.add(norm_title)
                unique_papers.append(paper)

        return unique_papers

    # O(N^2) fuzzy deduplication with optimization
    for paper in papers:
        title = paper.title
        norm_title = normalize(title)

        if not norm_title:
            continue

        is_duplicate = False
        for seen_title in seen_titles:
            seen_norm = normalize(seen_title)

            # Optimization 1: Length check
            if abs(len(norm_title) - len(seen_norm)) / max(len(norm_title), len(seen_norm)) > 0.2:
                continue

            # Optimization 2: First character check
            if norm_title[0] != seen_norm[0]:
                continue

            if calculate_similarity(title, seen_title) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_papers.append(paper)
            seen_titles.append(title)

    return unique_papers

async def assess_paper_quality(papers: List[Paper]) -> List[Paper]:
    """Score papers based on multiple quality indicators."""
    for paper in papers:
        score = 0
        citations = paper.citation_count

        if citations >= 1000:
            score += 5
        elif citations >= 100:
            score += 3
        elif citations >= 10:
            score += 1

        if paper.pdf_url:
            score += 1

        if score >= 6:
            tier = "Excellent"
        elif score >= 3:
            tier = "Good"
        else:
            tier = "Fair"

        paper.quality_score = score
        paper.quality_tier = tier

    # Sort in place
    papers.sort(key=lambda x: x.quality_score, reverse=True)
    return papers

async def find_open_access(doi: str) -> Dict:
    """Find open access versions using Unpaywall."""
    try:
        url = f"{UNPAYWALL_API}/{doi}"
        data = await make_api_request(url, params={"email": CONTACT_EMAIL})

        return {
            "is_open_access": data.get("is_oa", False),
            "pdf_url": data.get("best_oa_location", {}).get("url_for_pdf"),
        }
    except Exception as e:
        logger.error(f"Open Access search failed: {e}")
        return {"is_open_access": False}

async def get_paper_citations(paper_id: str, max_citations: int = 10, max_references: int = 10) -> Dict:
    """Get citation network."""
    url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}"
    # FIX: Added paperId, citations.paperId, references.paperId to fields
    params = {"fields": "paperId,title,citations.paperId,citations.title,citations.year,references.paperId,references.title,references.year"}

    try:
        data = await make_api_request(url, params=params)
        citations = data.get("citations", [])[:max_citations]
        references = data.get("references", [])[:max_references]

        return {
            "paper_id": data.get("paperId"),
            "title": data.get("title"),
            "citing_papers": [{"paper_id": c.get("paperId"), "title": c.get("title"), "year": c.get("year")} for c in citations],
            "referenced_papers": [{"paper_id": r.get("paperId"), "title": r.get("title"), "year": r.get("year")} for r in references],
        }
    except Exception as e:
        logger.error(f"Citation network failed: {e}")
        return {}

async def generate_research_summary(papers: List[Paper]) -> str:
    """Generate structured research summary."""
    if not papers:
        return "# Research Summary\n\nNo papers to summarize."

    summary = f"# Research Summary\n\n**Total Papers:** {len(papers)}\n\n"

    for i, paper in enumerate(papers[:10], 1):
        summary += f"## {i}. {paper.title}\n"
        summary += f"- **Authors:** {', '.join(paper.authors[:3])}\n"
        summary += f"- **Year:** {paper.year or 'N/A'}\n"
        summary += f"- **Citations:** {paper.citation_count}\n\n"

    return summary

async def research_pipeline(
    query: str, max_results: int = 20, min_quality_score: int = 3, year_filter: Optional[str] = None
) -> Dict:
    """Complete automated research pipeline."""
    logger.info(f"Starting pipeline: '{query}'")
    start_time = time.time()

    # Step 1: Multi-source search
    results = await search_all_sources(query, max_results_per_source=10)
    all_papers = []
    for papers in results.values():
        all_papers.extend(papers)

    # Step 2: Deduplicate
    unique_papers = await deduplicate_papers(all_papers)

    # Step 3: Quality assessment
    scored_papers = await assess_paper_quality(unique_papers)

    # Step 4: Filter
    filtered = [p for p in scored_papers if p.quality_score >= min_quality_score]

    # Step 5: Limit results
    final = filtered[:max_results]

    return {
        "query": query,
        "statistics": {
            "initial_papers": len(all_papers),
            "after_dedup": len(unique_papers),
            "final_count": len(final),
            "processing_time": round(time.time() - start_time, 2),
        },
        # Convert to dict for JSON serialization if needed, or rely on Pydantic's serialization
        "final_papers": [p.to_dict() for p in final],
    }

async def batch_process_papers(paper_identifiers: List[str], operations: List[str]) -> List[Dict]:
    """Process multiple papers concurrently."""

    async def process_single_paper(identifier: str) -> Dict:
        result = {"identifier": identifier, "operations": {}}
        if "open_access" in operations:
            result["operations"]["open_access"] = await find_open_access(identifier)
        return result

    # Create tasks for all papers
    tasks = [process_single_paper(identifier) for identifier in paper_identifiers]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    return list(results)

async def extract_pdf_metadata(pdf_url: str) -> Dict:
    """Extract PDF metadata."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.head(pdf_url)
            return {
                "accessible": response.status_code == 200,
                "size_mb": int(response.headers.get("content-length", 0)) / (1024 * 1024),
            }
    except Exception as e:
        logger.error(f"PDF metadata extraction failed: {e}")
        return {"accessible": False}

async def export_to_bibtex(papers: List[Paper], output_file: str = "references.bib") -> str:
    """Export papers to BibTeX format."""
    bibtex = ""
    for i, paper in enumerate(papers, 1):
        bibtex += f"@article{{paper{i},\n"
        bibtex += f"  title = {{{paper.title}}},\n"
        bibtex += f"  year = {{{paper.year or 'N/A'}}},\n"
        bibtex += "}\n\n"

    Path(output_file).write_text(bibtex)
    return f"Exported {len(papers)} papers to {output_file}"
