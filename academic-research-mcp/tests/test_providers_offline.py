import os
import sys
import pytest
import httpx
from httpx import Response
import json

# Add parent directory to path
sys.path.append(os.path.join(os.getcwd(), "academic-research-mcp"))

# Import after modifying path
from research_tools import (
    arxiv_provider,
    search_arxiv,
    semantic_provider,
    crossref_provider,
    search_crossref,
    get_paper_citations,
    batch_process_papers,
    find_open_access
)
import enhancements.deep_research
from utils import ProviderError, RateLimitExceeded, _cache

# Get the original function if deep_research is decorated
deep_research = enhancements.deep_research.deep_research


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the cache before each test"""
    _cache.clear()
    yield
    _cache.clear()


@pytest.mark.asyncio
async def test_arxiv_provider_success(respx_mock):
    """Test Arxiv provider happy path"""
    # Mock XML response
    xml_content = """
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2101.00001</id>
            <title>Test Paper</title>
            <summary>Abstract content</summary>
            <author><name>Author One</name></author>
            <published>2021-01-01T00:00:00Z</published>
        </entry>
    </feed>
    """

    route = respx_mock.get("https://export.arxiv.org/api/query").mock(return_value=Response(200, text=xml_content))

    results = await arxiv_provider.search("test", max_results=5)

    assert route.called
    assert len(results) == 1
    assert results[0].title == "Test Paper"
    assert results[0].source == "arXiv"


@pytest.mark.asyncio
async def test_semantic_provider_success(respx_mock):
    """Test Semantic Scholar provider happy path"""
    json_content = {
        "data": [
            {
                "paperId": "123",
                "title": "Semantic Paper",
                "authors": [{"name": "Author A"}],
                "year": 2023,
                "citationCount": 10,
            }
        ]
    }

    route = respx_mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=Response(200, json=json_content)
    )

    results = await semantic_provider.search("test", max_results=5)

    assert route.called
    assert len(results) == 1
    assert results[0].title == "Semantic Paper"
    assert results[0].source == "Semantic Scholar"


@pytest.mark.asyncio
async def test_crossref_provider_success(respx_mock):
    """Test CrossRef provider happy path"""
    json_content = {
        "message": {
            "items": [
                {
                    "DOI": "10.1000/1",
                    "title": ["CrossRef Paper"],
                    "published-print": {"date-parts": [[2022]]},
                    "is-referenced-by-count": 5
                }
            ]
        }
    }

    route = respx_mock.get("https://api.crossref.org/works").mock(
        return_value=Response(200, json=json_content)
    )

    results = await crossref_provider.search("test", max_results=5)

    assert route.called
    assert len(results) == 1
    assert results[0].title == "CrossRef Paper"
    assert results[0].source == "CrossRef"


@pytest.mark.asyncio
async def test_provider_rate_limit(respx_mock):
    """Test rate limiting handling"""
    respx_mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(return_value=Response(429))

    with pytest.raises(ProviderError) as exc:
        await semantic_provider.search("test", max_results=5)

    # Check if the cause is correct (might be nested)
    assert "Rate limit exceeded" in str(exc.value) or isinstance(exc.value.__cause__, RateLimitExceeded)


@pytest.mark.asyncio
async def test_provider_network_error(respx_mock):
    """Test network error handling"""
    respx_mock.get("https://export.arxiv.org/api/query").mock(side_effect=httpx.ConnectError("Connection failed"))

    with pytest.raises(ProviderError) as exc:
        await arxiv_provider.search("test", max_results=5)

    assert "Failed to search arXiv" in str(exc.value)


@pytest.mark.asyncio
async def test_tool_wrapper_swallows_error(respx_mock):
    """Test that the tool wrapper returns empty list on error (as required for MCP tool stability)"""
    respx_mock.get("https://export.arxiv.org/api/query").mock(return_value=Response(500))

    # calling the wrapper function, not the provider method
    results = await search_arxiv("test", max_results=5)

    assert results == []


@pytest.mark.asyncio
async def test_get_paper_citations(respx_mock):
    """Test get_paper_citations functionality"""
    paper_id = "123"
    json_content = {
        "paperId": paper_id,
        "title": "Main Paper",
        "citations": [
            {"paperId": "abc", "title": "Citing Paper", "year": 2024}
        ],
        "references": [
            {"paperId": "def", "title": "Referenced Paper", "year": 2020}
        ]
    }

    url_pattern = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    route = respx_mock.get(url_pattern).mock(
        return_value=Response(200, json=json_content)
    )

    result = await get_paper_citations(paper_id)

    assert route.called
    # Phase 1 verification: check if fields were requested correctly
    # Decode query to check string content
    query_str = route.calls.last.request.url.query.decode("utf-8")
    assert "fields=paperId,title,citations.paperId" in  __import__("urllib.parse").parse.unquote(query_str)

    assert result["paper_id"] == paper_id
    assert len(result["citing_papers"]) == 1
    assert len(result["referenced_papers"]) == 1
    assert result["citing_papers"][0]["title"] == "Citing Paper"


@pytest.mark.asyncio
async def test_batch_process_papers(respx_mock):
    """Test batch processing with open access check"""
    # Mock Unpaywall API
    respx_mock.get("https://api.unpaywall.org/v2/10.1234/5678").mock(
        return_value=Response(200, json={"is_oa": True, "best_oa_location": {"url_for_pdf": "http://pdf"}})
    )
    respx_mock.get("https://api.unpaywall.org/v2/10.1111/2222").mock(
        return_value=Response(200, json={"is_oa": False})
    )

    identifiers = ["10.1234/5678", "10.1111/2222"]
    operations = ["open_access"]

    results = await batch_process_papers(identifiers, operations)

    assert len(results) == 2
    assert results[0]["identifier"] == "10.1234/5678"
    assert results[0]["operations"]["open_access"]["is_open_access"] is True
    assert results[1]["identifier"] == "10.1111/2222"
    assert results[1]["operations"]["open_access"]["is_open_access"] is False


@pytest.mark.asyncio
async def test_deep_research_mocked(respx_mock):
    """Test deep research with mocked search results"""
    # Mock Search API calls (Arxiv, Semantic Scholar, CrossRef)

    # Arxiv
    arxiv_xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>123</id><title>Deep Learning</title><summary>Summary</summary></entry></feed>"""
    respx_mock.get("https://export.arxiv.org/api/query").mock(return_value=Response(200, text=arxiv_xml))

    # Semantic Scholar
    ss_json = {"data": [{"paperId": "p1", "title": "Deep Learning Overview", "citationCount": 100}]}
    respx_mock.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(return_value=Response(200, json=ss_json))

    # CrossRef
    cr_json = {"message": {"items": []}}
    respx_mock.get("https://api.crossref.org/works").mock(return_value=Response(200, json=cr_json))

    # Run deep research with limited depth and requests to ensure speed
    # FastMCP wrapper might need bypassing
    if hasattr(deep_research, "fn"):
        func = deep_research.fn
    else:
        func = deep_research

    result = await func(
        "Deep Learning",
        max_depth=1,
        max_papers_per_level=2,
        max_total_requests=5,
        include_counter_arguments=False,
        critical_analysis=False
    )

    assert "error" not in result
    assert result["papers_analyzed"] > 0
    assert result["levels"][0]["level"] == 1
