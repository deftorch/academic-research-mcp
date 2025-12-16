# Academic Research MCP Server - API Specification

This document details the JSON schemas for the core tools provided by the server.

## 1. search_arxiv

Search for papers on arXiv.

### Input Schema
```json
{
  "query": "string",            // Required. The search keyword.
  "max_results": 10,            // Optional. Default 10. Max papers to return.
  "sort_by": "relevance"        // Optional. Default "relevance". Options: "relevance", "lastUpdatedDate", "submittedDate"
}
```

### Output Schema
Returns a list of paper objects:
```json
[
  {
    "source": "arXiv",
    "title": "Paper Title",
    "authors": ["Author One", "Author Two"],
    "abstract": "Abstract content...",
    "pdf_url": "http://arxiv.org/pdf/...",
    "url": "http://arxiv.org/abs/...",
    "published": "2023-01-01T00:00:00Z",
    "arxiv_id": "2301.00000",
    "categories": ["cs.AI"]
  }
]
```

## 2. research_pipeline

Automated pipeline: Search -> Deduplicate -> Assess Quality -> Filter.

### Input Schema
```json
{
  "query": "string",            // Required. Research topic.
  "max_results": 20,            // Optional. Default 20. Max final papers.
  "min_quality_score": 3,       // Optional. Default 3. Min quality score (0-12) to keep.
  "year_filter": "2023"         // Optional. Filter by specific year.
}
```

### Output Schema
```json
{
  "query": "string",
  "statistics": {
    "initial_papers": 100,
    "after_dedup": 80,
    "final_count": 20,
    "processing_time": 1.5
  },
  "final_papers": [ ... list of paper objects ... ]
}
```

## 3. deep_research

Autonomous recursive research agent.

### Input Schema
```json
{
  "initial_query": "string",            // Required.
  "max_depth": 3,                       // Optional. Default 3. Recursion depth.
  "max_papers_per_level": 10,           // Optional. Default 10.
  "max_total_requests": 50,             // Optional. Default 50. Safeguard limit.
  "include_counter_arguments": true,    // Optional. Default true.
  "critical_analysis": true             // Optional. Default true.
}
```

### Output Schema
```json
{
  "initial_query": "string",
  "levels": [
    {
      "level": 1,
      "papers_found": 10,
      "top_papers": [ ... ]
    },
    {
      "level": 2,
      "questions": ["Follow up Q1", ...],
      "papers_found": 5,
      "papers": [ ... ]
    }
  ],
  "synthesis": {
    "main_findings": ["Finding 1", ...],
    "evidence_strength": "strong",
    "limitations": ["Limitation 1", ...],
    "recommendations": ["Rec 1", ...]
  },
  "critical_analysis": { ... },
  "duration_seconds": 15.5,
  "total_requests": 25
}
```
