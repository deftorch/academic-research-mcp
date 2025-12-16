"""
Academic Research MCP Server - RAG Enhancement Module
=====================================================
Semantic search and Q&A capabilities using vector embeddings.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import chromadb
    try:
        from chromadb.config import Settings
    except ImportError:
        from chromadb import Settings
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import numpy as np
except ImportError:
    print("Installing RAG dependencies...")
    import subprocess

    subprocess.check_call(["pip", "install", "chromadb", "sentence-transformers", "rank_bm25", "numpy"])
    import chromadb
    try:
        from chromadb.config import Settings
    except ImportError:
        from chromadb import Settings
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import numpy as np

try:
    from ..mcp_instance import mcp
except ImportError:
    # If running as script/test or direct import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcp_instance import mcp

logger = logging.getLogger(__name__)

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

# Initialize embedding model (384-dimensional, fast and accurate)
# This model runs efficiently on CPU or GPU
try:
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✓ Embedding model loaded (all-MiniLM-L6-v2)")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    EMBEDDING_MODEL = None

# Initialize ChromaDB client (persistent vector database)
VECTORDB_PATH = Path("./paper_vectordb")
VECTORDB_PATH.mkdir(exist_ok=True)

try:
    chroma_client = chromadb.Client(Settings(persist_directory=str(VECTORDB_PATH), anonymized_telemetry=False))
    logger.info(f"✓ Vector database initialized: {VECTORDB_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    chroma_client = None

# Get or create collection
try:
    paper_collection = chroma_client.get_or_create_collection(
        name="academic_papers", metadata={"description": "Academic paper abstracts and content"}
    )
    logger.info(f"✓ Collection ready (current size: {paper_collection.count()})")
except Exception as e:
    logger.error(f"Failed to get collection: {e}")
    paper_collection = None

# ============================================================================
# TOOL 1: INDEX PAPERS TO VECTOR DATABASE
# ============================================================================


@mcp.tool()
async def index_papers_to_vectordb(
    papers: List[Dict], include_abstracts: bool = True, chunk_size: int = 500, overwrite_existing: bool = False
) -> Dict:
    """
    Index papers into vector database for semantic search.
    """
    if not EMBEDDING_MODEL or not paper_collection:
        return {"error": "RAG system not initialized", "papers_indexed": 0}

    logger.info(f"Indexing {len(papers)} papers to vector database...")

    indexed_count = 0
    skipped_count = 0
    errors = []

    for paper in papers:
        try:
            # Create unique ID
            paper_id = (
                paper.get("doi")
                or paper.get("arxiv_id")
                or paper.get("paper_id")
                or hashlib.md5(paper.get("title", "").encode()).hexdigest()
            )

            # Check if already exists
            if not overwrite_existing:
                try:
                    existing = paper_collection.get(ids=[paper_id])
                    if existing["ids"]:
                        skipped_count += 1
                        continue
                except:
                    pass

            # Prepare text for embedding
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if not title:
                skipped_count += 1
                continue

            if include_abstracts and abstract:
                text = f"Title: {title}\n\nAbstract: {abstract}"
            else:
                text = f"Title: {title}"

            # Truncate if too long (model has 512 token limit)
            if len(text) > 2000:
                text = text[:2000] + "..."

            # Generate embedding
            embedding = EMBEDDING_MODEL.encode(text).tolist()

            # Prepare metadata
            metadata = {
                "paper_id": paper_id,
                "title": title[:500],  # ChromaDB has metadata size limits
                "authors": ", ".join(paper.get("authors", [])[:3]),
                "year": str(paper.get("year", "Unknown")),
                "source": paper.get("source", "Unknown"),
                "citation_count": int(paper.get("citation_count", 0)),
                "url": paper.get("url", "")[:500],
                "pdf_url": paper.get("pdf_url", "")[:500],
                "has_abstract": bool(abstract),
            }

            # Add to collection
            paper_collection.upsert(ids=[paper_id], embeddings=[embedding], documents=[text], metadatas=[metadata])

            indexed_count += 1

            if indexed_count % 10 == 0:
                logger.info(f"Indexed {indexed_count}/{len(papers)} papers...")

        except Exception as e:
            error_msg = f"Failed to index paper: {paper.get('title', 'Unknown')[:50]} - {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            skipped_count += 1

    result = {
        "papers_indexed": indexed_count,
        "papers_skipped": skipped_count,
        "total_in_database": paper_collection.count(),
        "errors": errors[:5],  # Only show first 5 errors
    }

    logger.info(
        f"Indexing complete: {indexed_count} indexed, {skipped_count} skipped. Total in DB: {paper_collection.count()}"
    )

    return result


# ============================================================================
# TOOL 2: SEMANTIC SEARCH
# ============================================================================


@mcp.tool()
async def semantic_search_papers(
    query: str,
    max_results: int = 10,
    min_year: Optional[int] = None,
    min_citation_count: Optional[int] = None,
    similarity_threshold: float = 0.0,
    hybrid_weight: float = 0.7  # 0.7 vector, 0.3 keyword
) -> List[Dict]:
    """
    Hybrid semantic search combining Vector (ChromaDB) and Keyword (BM25) search.
    """
    if not EMBEDDING_MODEL or not paper_collection:
        return []

    logger.info(f"Hybrid search: '{query}'")

    # Check if database is empty
    count = paper_collection.count()
    if count == 0:
        logger.warning("Vector database is empty. Index papers first.")
        return []

    # --- 1. Vector Search ---
    query_embedding = EMBEDDING_MODEL.encode(query).tolist()

    where_filter = {}
    if min_year:
        where_filter["year"] = {"$gte": str(min_year)}
    if min_citation_count:
        where_filter["citation_count"] = {"$gte": min_citation_count}

    try:
        # Retrieve more candidates for reranking
        candidate_limit = min(max_results * 5, 200)
        vector_results = paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
            where=where_filter if where_filter else None,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

    if not vector_results['ids'][0]:
        return []

    # Map results to unified structure
    candidates = {}

    # Process vector scores
    for i, (doc, meta, dist, pid) in enumerate(zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0],
        vector_results["ids"][0]
    )):
        sim = max(0, 1 - (dist / 2)) # Approx cosine similarity

        candidates[pid] = {
            "paper_id": pid,
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("year", ""),
            "source": meta.get("source", ""),
            "citation_count": meta.get("citation_count", 0),
            "url": meta.get("url", ""),
            "pdf_url": meta.get("pdf_url", ""),
            "text": doc,
            "relevant_excerpt": doc[:300] + "..." if len(doc) > 300 else doc,
            "vector_score": sim,
            "keyword_score": 0.0
        }

    # --- 2. Keyword Search (BM25) on Candidates ---
    # We only run BM25 on the retrieved candidates to re-rank them
    # This is "Retrieve and Re-rank" strategy which is efficient

    tokenized_corpus = [c['text'].lower().split() for c in candidates.values()]
    tokenized_query = query.lower().split()

    if tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
        # Convert to numpy array for vector operations
        bm25_scores = np.array(bm25.get_scores(tokenized_query))

        # Normalize BM25 scores (0-1)
        if bm25_scores.max() > 0:
             bm25_scores = bm25_scores / bm25_scores.max()

        for idx, pid in enumerate(candidates.keys()):
            candidates[pid]['keyword_score'] = float(bm25_scores[idx])

    # --- 3. Hybrid Fusion ---
    final_results = []
    for pid, data in candidates.items():
        # Weighted sum fusion
        final_score = (data['vector_score'] * hybrid_weight) + (data['keyword_score'] * (1 - hybrid_weight))

        if final_score < similarity_threshold:
            continue

        data['similarity_score'] = round(final_score, 4)
        final_results.append(data)

    # Sort by final score
    sorted_papers = sorted(final_results, key=lambda x: x["similarity_score"], reverse=True)[:max_results]

    # Add rank
    for i, p in enumerate(sorted_papers):
        p['rank'] = i + 1

    logger.info(f"Found {len(sorted_papers)} hybrid search results")
    return sorted_papers


# ============================================================================
# TOOL 3: ANSWER QUESTIONS FROM PAPERS
# ============================================================================


@mcp.tool()
async def answer_question_from_papers(
    question: str, max_context_papers: int = 5, include_excerpts: bool = True
) -> Dict:
    """
    Answer research questions using indexed papers as context.
    """
    if not EMBEDDING_MODEL or not paper_collection:
        return {"error": "RAG system not initialized", "answer": None}

    logger.info(f"Answering question: '{question}'")

    # Retrieve relevant papers
    relevant_papers = await semantic_search_papers(query=question, max_results=max_context_papers)

    if not relevant_papers:
        return {
            "question": question,
            "answer": "No relevant papers found in database. Try indexing more papers.",
            "confidence": 0.0,
            "supporting_papers": [],
        }

    # Build answer from papers
    supporting_papers = []
    context_parts = []

    for i, paper in enumerate(relevant_papers, 1):
        excerpt = paper.get("relevant_excerpt", "")

        supporting_papers.append(
            {
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"],
                "similarity": paper["similarity_score"],
                "excerpt": excerpt if include_excerpts else None,
            }
        )

        context_parts.append(
            f"[Paper {i}] {paper['title']} ({paper['year']})\n"
            f"Authors: {paper['authors']}\n"
            f"Relevance: {paper['similarity_score']:.0%}\n"
            f"Excerpt: {excerpt}\n"
        )

    # Calculate confidence based on top similarity
    top_similarity = relevant_papers[0]["similarity_score"]
    confidence = min(top_similarity * 1.2, 1.0)  # Boost slightly, cap at 100%

    # Construct answer
    answer_text = (
        f"Based on analysis of {len(relevant_papers)} relevant papers:\n\n"
        f"{chr(10).join(context_parts)}\n\n"
        f"Note: This is a retrieval-based response. Review the supporting "
        f"papers above for detailed information."
    )

    result = {
        "question": question,
        "answer": answer_text,
        "confidence": round(confidence, 2),
        "supporting_papers": supporting_papers,
        "papers_analyzed": len(relevant_papers),
    }

    logger.info(f"Answer generated with {confidence:.0%} confidence from {len(relevant_papers)} papers")

    return result


# ============================================================================
# TOOL 4: FIND SIMILAR PAPERS
# ============================================================================


@mcp.tool()
async def find_similar_papers(
    reference_paper: Dict, max_results: int = 10, exclude_same_authors: bool = False
) -> List[Dict]:
    """
    Find papers similar to a given reference paper.
    """
    if not EMBEDDING_MODEL or not paper_collection:
        return []

    title = reference_paper.get("title", "")
    if not title:
        logger.error("Reference paper must have a title")
        return []

    logger.info(f"Finding papers similar to: '{title[:50]}...'")

    # Use title + abstract as query
    query_text = title
    if reference_paper.get("abstract"):
        query_text += f"\n\n{reference_paper['abstract']}"

    # Semantic search
    similar_papers = await semantic_search_papers(
        query=query_text,
        max_results=max_results + 5,  # Get extra to filter
    )

    # Filter out exact match and same authors if requested
    ref_authors = set(reference_paper.get("authors", []))

    filtered = []
    for paper in similar_papers:
        # Skip exact title match
        if paper["title"].lower() == title.lower():
            continue

        # Skip same authors if requested
        if exclude_same_authors and ref_authors:
            paper_authors = set(paper["authors"].split(", "))
            if ref_authors & paper_authors:  # Any author overlap
                continue

        filtered.append(paper)

        if len(filtered) >= max_results:
            break

    logger.info(f"Found {len(filtered)} similar papers")

    return filtered


# ============================================================================
# TOOL 5: GENERATE AI LITERATURE REVIEW
# ============================================================================


@mcp.tool()
async def generate_literature_review(
    topic: str, max_papers: int = 20, min_year: Optional[int] = None, group_by: str = "year"
) -> str:
    """
    Generate a structured literature review using indexed papers.
    """
    if not EMBEDDING_MODEL or not paper_collection:
        return "# Literature Review\n\nRAG system not initialized."

    logger.info(f"Generating literature review: '{topic}'")

    # Retrieve relevant papers
    papers = await semantic_search_papers(query=topic, max_results=max_papers, min_year=min_year)

    if not papers:
        return f"# Literature Review: {topic}\n\nNo papers found in database."

    # Start review
    review = f"# Literature Review: {topic}\n\n"
    review += f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n"
    review += f"**Papers Analyzed:** {len(papers)}\n"
    if min_year:
        review += f"**Time Period:** {min_year} onwards\n"
    review += "\n---\n\n"

    # Group papers
    if group_by == "year":
        groups = {}
        for paper in papers:
            year = paper.get("year", "Unknown")
            if year not in groups:
                groups[year] = []
            groups[year].append(paper)

        sorted_years = sorted(groups.keys(), reverse=True)

        review += "## Overview by Year\n\n"
        for year in sorted_years:
            year_papers = groups[year]
            review += f"### {year} ({len(year_papers)} papers)\n\n"

            for paper in year_papers:
                review += f"**{paper['title']}**  \n"
                review += f"*{paper['authors']}*  \n"
                review += f"Relevance: {paper['similarity_score']:.0%} | "
                review += f"Citations: {paper['citation_count']}  \n"
                if paper.get("url"):
                    review += f"[Link]({paper['url']})  \n"
                review += "\n"

    elif group_by == "citations":
        # Sort by citation count
        sorted_papers = sorted(papers, key=lambda x: x.get("citation_count", 0), reverse=True)

        review += "## Papers by Impact (Citations)\n\n"
        for i, paper in enumerate(sorted_papers, 1):
            review += f"### {i}. {paper['title']}\n\n"
            review += f"- **Authors:** {paper['authors']}\n"
            review += f"- **Year:** {paper['year']}\n"
            review += f"- **Citations:** {paper['citation_count']}\n"
            review += f"- **Relevance:** {paper['similarity_score']:.0%}\n"
            if paper.get("url"):
                review += f"- **Link:** {paper['url']}\n"
            review += "\n"

    else:  # group_by similarity
        review += "## Papers by Relevance\n\n"
        for i, paper in enumerate(papers, 1):
            review += f"### {i}. {paper['title']}\n\n"
            review += f"**Relevance Score:** {paper['similarity_score']:.0%}  \n"
            review += f"**Authors:** {paper['authors']}  \n"
            review += f"**Year:** {paper['year']} | **Citations:** {paper['citation_count']}  \n"
            review += "\n"

    # Summary statistics
    review += "## Summary Statistics\n\n"
    review += f"- **Total Papers:** {len(papers)}\n"
    review += f"- **Average Citations:** {sum(p.get('citation_count', 0) for p in papers) / len(papers):.1f}\n"
    review += f"- **Year Range:** {min(p.get('year', '9999') for p in papers)} - {max(p.get('year', '0000') for p in papers)}\n"

    logger.info(f"Literature review generated: {len(papers)} papers")

    return review


# ============================================================================
# TOOL 6: VECTOR DATABASE STATISTICS
# ============================================================================


@mcp.tool()
async def get_vectordb_stats() -> Dict:
    """
    Get statistics about the vector database.
    """
    if not paper_collection:
        return {"error": "Vector database not initialized"}

    total = paper_collection.count()

    if total == 0:
        return {"total_papers": 0, "message": "Database is empty. Index papers first."}

    # Get sample for analysis
    sample_size = min(100, total)
    sample = paper_collection.get(limit=sample_size)

    # Analyze metadata
    years = []
    sources = []
    citations = []

    for metadata in sample["metadatas"]:
        if metadata.get("year") and metadata["year"] != "Unknown":
            try:
                years.append(int(metadata["year"]))
            except:
                pass

        if metadata.get("source"):
            sources.append(metadata["source"])

        if metadata.get("citation_count"):
            citations.append(metadata["citation_count"])

    stats = {
        "total_papers": total,
        "database_path": str(VECTORDB_PATH),
        "year_range": f"{min(years)} - {max(years)}" if years else "Unknown",
        "sources": list(set(sources)),
        "avg_citations": round(sum(citations) / len(citations), 1) if citations else 0,
        "estimated_size_mb": round(total * 0.001, 2),  # Rough estimate
    }

    logger.info(f"Vector DB stats: {total} papers indexed")

    return stats


# ============================================================================
# TOOL 7: CLEAR VECTOR DATABASE
# ============================================================================


@mcp.tool()
async def clear_vectordb(confirm: bool = False) -> Dict:
    """
    Clear all papers from vector database.
    """
    global paper_collection

    if not paper_collection:
        return {"error": "Vector database not initialized"}

    if not confirm:
        return {"error": "Must set confirm=True to clear database", "current_size": paper_collection.count()}

    try:
        count_before = paper_collection.count()

        # Delete collection and recreate
        chroma_client.delete_collection("academic_papers")
        paper_collection = chroma_client.create_collection(
            name="academic_papers", metadata={"description": "Academic paper abstracts and content"}
        )

        logger.warning(f"Vector database cleared: {count_before} papers deleted")

        return {"status": "cleared", "papers_deleted": count_before, "current_size": 0}
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        return {"error": str(e)}
