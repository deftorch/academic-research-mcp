import pytest
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "academic-research-mcp", "src"))

from academic_research.domain.models import Paper
from academic_research.services import research_service

@pytest.mark.asyncio
async def test_deduplicate_papers():
    p1 = Paper(title="Paper 1", source="Source A")
    p2 = Paper(title="Paper 1", source="Source B") # Duplicate
    p3 = Paper(title="Paper 2", source="Source A")

    papers = [p1, p2, p3]
    unique = await research_service.deduplicate_papers(papers, quick_dedup=True)

    assert len(unique) == 2
    titles = sorted([p.title for p in unique])
    assert titles == ["Paper 1", "Paper 2"]

def test_paper_quality_scoring():
    p1 = Paper(title="Good Paper", source="Test", citation_count=1000)

    import asyncio
    loop = asyncio.new_event_loop()
    scored = loop.run_until_complete(research_service.assess_paper_quality([p1]))

    # 1000 citations -> score 5. No PDF -> score 5.
    # Score >= 3 is "Good". Score >= 6 is "Excellent".
    assert scored[0].quality_score == 5
    assert scored[0].quality_tier == "Good"
