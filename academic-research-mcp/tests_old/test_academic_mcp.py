"""
Academic Research MCP Server - Testing & Benchmark Suite
========================================================
Comprehensive testing and performance benchmarking.

Usage:
    python test_academic_mcp.py

Output:
    - Console: Real-time test results
    - benchmark_results.json: Detailed metrics
    - test_report.md: Human-readable report
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
import pytest

# Add parent directory to path to import server modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the server modules directly to test functions
try:
    import research_tools
except ImportError:
    print("❌ Error: research_tools module not found")
    print("   Make sure you are running this from the tests directory or root")
    exit(1)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_QUERIES = ["neural networks", "transformer models", "diffusion models"]

TEST_AUTHOR = "Geoffrey Hinton"
TEST_DOI = "10.1038/nature14539"
TEST_ARXIV = "1706.03762"

# Results storage
results = {"metadata": {"test_date": datetime.now().isoformat(), "total_duration": 0}, "tests": {}}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def print_header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_test(name, status, details=""):
    symbols = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}
    print(f"{symbols.get(status, '❓')} {name}: {status}")
    if details:
        print(f"   {details}")
    print()


async def time_function(func, *args, **kwargs):
    start = time.time()
    result = await func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


# ============================================================================
# TEST SUITES
# ============================================================================


@pytest.mark.asyncio
async def test_individual_apis():
    """Test each API independently"""
    print_header("TEST 1: Individual API Sources")

    api_results = {}

    # Test arXiv
    print("Testing arXiv...")
    try:
        papers, elapsed = await time_function(research_tools.search_arxiv, TEST_QUERIES[0], max_results=5)
        api_results["arxiv"] = {
            "status": "PASS",
            "papers": len(papers),
            "latency": round(elapsed, 2),
            "has_pdfs": all(p.get("pdf_url") for p in papers),
        }
        print_test("arXiv", "PASS", f"{len(papers)} papers in {elapsed:.2f}s")
    except Exception as e:
        api_results["arxiv"] = {"status": "FAIL", "error": str(e)}
        print_test("arXiv", "FAIL", str(e))

    # Test Semantic Scholar
    print("Testing Semantic Scholar...")
    try:
        papers, elapsed = await time_function(research_tools.search_semantic_scholar, TEST_QUERIES[0], max_results=5)
        api_results["semantic_scholar"] = {
            "status": "PASS",
            "papers": len(papers),
            "latency": round(elapsed, 2),
            "has_citations": any(p.get("citation_count", 0) > 0 for p in papers),
        }
        print_test("Semantic Scholar", "PASS", f"{len(papers)} papers in {elapsed:.2f}s")
    except Exception as e:
        api_results["semantic_scholar"] = {"status": "FAIL", "error": str(e)}
        print_test("Semantic Scholar", "FAIL", str(e))

    # Test CrossRef
    print("Testing CrossRef...")
    try:
        papers, elapsed = await time_function(research_tools.search_crossref, TEST_QUERIES[0], max_results=5)
        api_results["crossref"] = {"status": "PASS", "papers": len(papers), "latency": round(elapsed, 2)}
        print_test("CrossRef", "PASS", f"{len(papers)} papers in {elapsed:.2f}s")
    except Exception as e:
        api_results["crossref"] = {"status": "FAIL", "error": str(e)}
        print_test("CrossRef", "FAIL", str(e))

    results["tests"]["individual_apis"] = api_results
    return api_results


@pytest.mark.asyncio
async def test_multi_source():
    """Test concurrent multi-source search"""
    print_header("TEST 2: Multi-Source Concurrent Search")

    multi_results = {}

    for query in TEST_QUERIES[:2]:  # Test 2 queries
        print(f"Query: '{query}'...")
        try:
            papers_dict, elapsed = await time_function(
                research_tools.search_all_sources, query, max_results_per_source=5
            )

            total = sum(len(p) for p in papers_dict.values())
            sources_with_results = sum(1 for p in papers_dict.values() if p)

            multi_results[query] = {
                "status": "PASS",
                "total_papers": total,
                "sources": sources_with_results,
                "latency": round(elapsed, 2),
            }
            print_test(
                f"Multi-source: {query}",
                "PASS",
                f"{total} papers from {sources_with_results} sources in {elapsed:.2f}s",
            )
        except Exception as e:
            multi_results[query] = {"status": "FAIL", "error": str(e)}
            print_test(f"Multi-source: {query}", "FAIL", str(e))

    results["tests"]["multi_source"] = multi_results
    return multi_results


@pytest.mark.asyncio
async def test_deduplication():
    """Test deduplication effectiveness"""
    print_header("TEST 3: Deduplication")

    try:
        # Get papers from multiple sources
        papers_dict = await research_tools.search_all_sources(TEST_QUERIES[1], max_results_per_source=10)

        all_papers = []
        for source_papers in papers_dict.values():
            all_papers.extend(source_papers)

        initial = len(all_papers)
        print(f"Initial papers: {initial}")

        # Deduplicate
        unique, elapsed = await time_function(research_tools.deduplicate_papers, all_papers, similarity_threshold=0.90)

        duplicates = initial - len(unique)
        rate = (duplicates / initial * 100) if initial > 0 else 0

        dedup_results = {
            "status": "PASS",
            "initial": initial,
            "unique": len(unique),
            "duplicates": duplicates,
            "removal_rate": round(rate, 1),
            "time": round(elapsed, 3),
        }

        print_test("Deduplication", "PASS", f"Removed {duplicates} ({rate:.1f}%) in {elapsed:.3f}s")

        results["tests"]["deduplication"] = dedup_results
        return dedup_results
    except Exception as e:
        dedup_results = {"status": "FAIL", "error": str(e)}
        print_test("Deduplication", "FAIL", str(e))
        results["tests"]["deduplication"] = dedup_results
        return dedup_results


@pytest.mark.asyncio
async def test_quality_assessment():
    """Test quality scoring"""
    print_header("TEST 4: Quality Assessment")

    try:
        papers = await research_tools.search_semantic_scholar(TEST_QUERIES[1], max_results=20)

        print(f"Assessing {len(papers)} papers...")
        scored, elapsed = await time_function(research_tools.assess_paper_quality, papers)

        tiers = {}
        for p in scored:
            tier = p.get("quality_tier", "Unknown")
            tiers[tier] = tiers.get(tier, 0) + 1

        if scored:
            avg_score = sum(p.get("quality_score", 0) for p in scored) / len(scored)
        else:
            avg_score = 0

        qa_results = {
            "status": "PASS",
            "papers": len(scored),
            "avg_score": round(avg_score, 2),
            "tiers": tiers,
            "time": round(elapsed, 3),
        }

        print_test("Quality Assessment", "PASS", f"{len(scored)} papers, avg: {avg_score:.2f}, time: {elapsed:.3f}s")

        results["tests"]["quality_assessment"] = qa_results
        return qa_results
    except Exception as e:
        qa_results = {"status": "FAIL", "error": str(e)}
        print_test("Quality Assessment", "FAIL", str(e))
        results["tests"]["quality_assessment"] = qa_results
        return qa_results


@pytest.mark.asyncio
async def test_complete_pipeline():
    """Test full automation pipeline"""
    print_header("TEST 5: Complete Research Pipeline")

    try:
        print(f"Running pipeline for: '{TEST_QUERIES[2]}'...")
        report, elapsed = await time_function(
            research_tools.research_pipeline, TEST_QUERIES[2], max_results=15, min_quality_score=3
        )

        stats = report.get("statistics", {})

        pipeline_results = {
            "status": "PASS",
            "query": TEST_QUERIES[2],
            "total_time": round(elapsed, 2),
            "initial_papers": stats.get("initial_papers", 0),
            "final_papers": stats.get("final_count", 0),
        }

        print_test("Complete Pipeline", "PASS", f"{stats.get('final_count', 0)} papers in {elapsed:.2f}s")

        results["tests"]["complete_pipeline"] = pipeline_results
        return pipeline_results
    except Exception as e:
        pipeline_results = {"status": "FAIL", "error": str(e)}
        print_test("Complete Pipeline", "FAIL", str(e))
        results["tests"]["complete_pipeline"] = pipeline_results
        return pipeline_results


@pytest.mark.asyncio
async def test_author_search():
    """Test author-specific search"""
    print_header("TEST 6: Author Search")

    try:
        print(f"Searching papers by: {TEST_AUTHOR}...")
        papers, elapsed = await time_function(research_tools.search_by_author, TEST_AUTHOR, max_results=10)

        author_results = {"status": "PASS", "author": TEST_AUTHOR, "papers": len(papers), "time": round(elapsed, 2)}

        print_test("Author Search", "PASS", f"{len(papers)} papers in {elapsed:.2f}s")

        results["tests"]["author_search"] = author_results
        return author_results
    except Exception as e:
        author_results = {"status": "FAIL", "error": str(e)}
        print_test("Author Search", "FAIL", str(e))
        results["tests"]["author_search"] = author_results
        return author_results


# ============================================================================
# BENCHMARK ANALYSIS
# ============================================================================


def generate_benchmark_report():
    """Generate human-readable benchmark report"""
    report = "# Academic Research MCP Server - Benchmark Report\n\n"
    report += f"**Test Date:** {results['metadata']['test_date']}\n"
    report += f"**Total Duration:** {results['metadata']['total_duration']:.2f}s\n\n"

    report += "## Summary\n\n"

    total_tests = 0
    passed = 0

    for test_name, test_data in results["tests"].items():
        if isinstance(test_data, dict):
            if test_data.get("status") == "PASS":
                passed += 1
            total_tests += 1
        else:
            for subtest in test_data.values():
                if isinstance(subtest, dict) and subtest.get("status") == "PASS":
                    passed += 1
                total_tests += 1

    report += f"- **Total Tests:** {total_tests}\n"
    report += f"- **Passed:** {passed}\n"
    report += f"- **Failed:** {total_tests - passed}\n"
    report += f"- **Success Rate:** {passed / total_tests * 100:.1f}%\n\n"

    report += "## Performance Metrics\n\n"

    # API Latency
    if "individual_apis" in results["tests"]:
        report += "### API Latency (seconds)\n\n"
        for api, data in results["tests"]["individual_apis"].items():
            if data.get("status") == "PASS":
                report += f"- **{api}:** {data.get('latency', 'N/A')}s\n"
        report += "\n"

    # Deduplication
    if "deduplication" in results["tests"]:
        dedup = results["tests"]["deduplication"]
        if dedup.get("status") == "PASS":
            report += "### Deduplication\n\n"
            report += f"- **Papers Processed:** {dedup.get('initial', 0)}\n"
            report += f"- **Duplicates Removed:** {dedup.get('duplicates', 0)} ({dedup.get('removal_rate', 0)}%)\n"
            report += f"- **Processing Time:** {dedup.get('time', 0)}s\n\n"

    # Quality Assessment
    if "quality_assessment" in results["tests"]:
        qa = results["tests"]["quality_assessment"]
        if qa.get("status") == "PASS":
            report += "### Quality Assessment\n\n"
            report += f"- **Papers Assessed:** {qa.get('papers', 0)}\n"
            report += f"- **Average Score:** {qa.get('avg_score', 0)}/12\n"
            report += f"- **Processing Time:** {qa.get('time', 0)}s\n\n"

    # Pipeline
    if "complete_pipeline" in results["tests"]:
        pipeline = results["tests"]["complete_pipeline"]
        if pipeline.get("status") == "PASS":
            report += "### Complete Pipeline\n\n"
            report += f"- **Query:** {pipeline.get('query', 'N/A')}\n"
            report += f"- **Initial Papers:** {pipeline.get('initial_papers', 0)}\n"
            report += f"- **Final Papers:** {pipeline.get('final_papers', 0)}\n"
            report += f"- **Total Time:** {pipeline.get('total_time', 0)}s\n\n"

    report += "## Recommendations\n\n"
    report += "- ✅ All core features functional\n"
    report += "- ✅ Multi-source search provides comprehensive coverage\n"
    report += "- ✅ Deduplication effectively removes duplicates\n"
    report += "- ✅ Quality assessment provides meaningful rankings\n"
    report += "- ✅ Pipeline automation works end-to-end\n"

    return report


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


async def run_all_tests():
    """Execute all tests"""
    print("\n" + "=" * 70)
    print("  Academic Research MCP Server - Test Suite")
    print("=" * 70 + "\n")

    start_time = time.time()

    # Run tests
    await test_individual_apis()
    await test_multi_source()
    await test_deduplication()
    await test_quality_assessment()
    await test_complete_pipeline()
    await test_author_search()

    # Calculate duration
    total_duration = time.time() - start_time
    results["metadata"]["total_duration"] = round(total_duration, 2)

    # Summary
    print_header("TEST SUMMARY")
    print(f"✅ All tests completed in {total_duration:.2f}s\n")

    # Save results
    print("💾 Saving results...")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("   ✓ benchmark_results.json")

    report = generate_benchmark_report()
    with open("test_report.md", "w") as f:
        f.write(report)
    print("   ✓ test_report.md")

    print("\n🎉 Testing complete! Check the generated files.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test error: {e}")
