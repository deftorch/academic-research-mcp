from typing import Dict, List, Optional
from .mcp_instance import mcp
from .models import Paper  # Keeping local import for now, but will redirect to src

# Import from the new core library
from academic_research.services import research_service
from academic_research.domain.models import Paper as CorePaper

# Adapter to convert CorePaper to LocalPaper if needed (though they are identical Pydantic models)
# Since we are refactoring, we should update the MCP tools to use the Core library directly.

# ============================================================================
# SEARCH TOOLS
# ============================================================================

@mcp.tool()
async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> List[CorePaper]:
    """Search arXiv for papers by keyword."""
    return await research_service.search_arxiv(query, max_results, sort_by)

@mcp.tool()
async def search_semantic_scholar(
    query: str, max_results: int = 10, year_filter: Optional[str] = None, min_citation_count: Optional[int] = None
) -> List[CorePaper]:
    """Search Semantic Scholar with rich metadata and citation data."""
    return await research_service.search_semantic_scholar(query, max_results, year_filter, min_citation_count)

@mcp.tool()
async def search_crossref(query: str, max_results: int = 10, from_year: Optional[int] = None) -> List[CorePaper]:
    """Search CrossRef for published papers with DOI."""
    return await research_service.search_crossref(query, max_results, from_year)

@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> List[CorePaper]:
    """Search PubMed for biomedical literature."""
    return await research_service.search_pubmed(query, max_results)

@mcp.tool()
async def search_all_sources(
    query: str, max_results_per_source: int = 5, sources: Optional[List[str]] = None
) -> Dict[str, List[CorePaper]]:
    """Search multiple databases concurrently."""
    return await research_service.search_all_sources(query, max_results_per_source, sources)

@mcp.tool()
async def search_by_author(author_name: str, max_results: int = 10) -> List[CorePaper]:
    """Search papers by specific author."""
    return await research_service.search_by_author(author_name, max_results)

# ============================================================================
# PROCESSING TOOLS
# ============================================================================

@mcp.tool()
async def deduplicate_papers(
    papers: List[CorePaper], similarity_threshold: float = 0.90, quick_dedup: bool = False
) -> List[CorePaper]:
    """Remove duplicate papers using fuzzy matching."""
    return await research_service.deduplicate_papers(papers, similarity_threshold, quick_dedup)

@mcp.tool()
async def assess_paper_quality(papers: List[CorePaper]) -> List[CorePaper]:
    """Score papers based on multiple quality indicators."""
    return await research_service.assess_paper_quality(papers)

@mcp.tool()
async def find_open_access(doi: str) -> Dict:
    """Find open access versions using Unpaywall."""
    return await research_service.find_open_access(doi)

@mcp.tool()
async def get_paper_citations(paper_id: str, max_citations: int = 10, max_references: int = 10) -> Dict:
    """Get citation network."""
    return await research_service.get_paper_citations(paper_id, max_citations, max_references)

@mcp.tool()
async def generate_research_summary(papers: List[CorePaper]) -> str:
    """Generate structured research summary."""
    return await research_service.generate_research_summary(papers)

# ============================================================================
# AUTOMATION TOOLS
# ============================================================================

@mcp.tool()
async def research_pipeline(
    query: str, max_results: int = 20, min_quality_score: int = 3, year_filter: Optional[str] = None
) -> Dict:
    """Complete automated research pipeline."""
    return await research_service.research_pipeline(query, max_results, min_quality_score, year_filter)

@mcp.tool()
async def batch_process_papers(paper_identifiers: List[str], operations: List[str]) -> List[Dict]:
    """Process multiple papers concurrently."""
    return await research_service.batch_process_papers(paper_identifiers, operations)

@mcp.tool()
async def extract_pdf_metadata(pdf_url: str) -> Dict:
    """Extract PDF metadata."""
    return await research_service.extract_pdf_metadata(pdf_url)

@mcp.tool()
async def export_to_bibtex(papers: List[CorePaper], output_file: str = "references.bib") -> str:
    """Export papers to BibTeX format."""
    return await research_service.export_to_bibtex(papers, output_file)
