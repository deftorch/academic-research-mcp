# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-05-25

### Added
- **Monetization Persistence**: Implemented SQLite-backed user registry (`user_registry.db`) in `monetization.py` to prevent data loss on restart. Added `ACADEMIC_DB_PATH` env var support.
- **Deep Research Optimization**: Added `asyncio.Semaphore` based concurrency limiting to `deep_research.py` to prevent rate-limiting and resource exhaustion.
- **Configurable Caching**: Added `ACADEMIC_CACHE_DIR` environment variable support in `utils.py` to allow custom cache locations (essential for Docker volumes).
- **Optional Dependencies**: Heavy ML libraries (Torch, ChromaDB, Sentence Transformers) and external integrations (Zotero, Notion) are now fully optional. The server gracefully degrades functionality if they are missing.
- **Full Text Support Notice**: Added startup banner notification about `marker-pdf` requirement for full-text extraction capabilities.

### Changed
- **Dependency Management**: Split dependencies into `requirements.txt` (lightweight core) and `requirements-full.txt` (full features including ML/RAG) to resolve "Dependency Bloat".
- **Robustness**: Improved `knowledge_management.py` and `rag_enhancement.py` to safely handle missing optional dependencies without crashing at runtime.
- **Initialization**: Refactored database and client initialization to be lazy/on-demand where appropriate.

## [1.2.0] - 2025-05-24

### Added
- **Phase 1: Data Integrity**: Implemented Pydantic V2 models (`models.py`) for type-safe API responses and robust data validation.
- **Phase 2: Agent Intelligence**: Added Semantic Gap Detection using `sentence_transformers` to identify research limitations.
- **Phase 2: Personalized PageRank**: Enhanced `citation_network.py` with vector-biased PageRank for query-specific influence scoring.
- **Phase 3: Persistence**: Replaced `TTLCache` with `DiskCache` (SQLite-backed) in `utils.py` for persistent caching across restarts.
- **Phase 3: Hybrid Search**: Implemented `rag_enhancement.py` combining BM25 keyword search and vector similarity search (ChromaDB).
- **Phase 4: Docker Support**: Added multi-stage `Dockerfile` and `.dockerignore` for efficient containerized deployment.
- **Phase 4: Visualization**: Added `get_citation_graph_data` tool to export raw graph JSON for frontend integration.

### Changed
- **Performance**: Refactored `batch_process_papers` and `deep_research` loops to use `asyncio.gather` for concurrent API execution (Phase 3).
- **Testing**: Greatly expanded `tests/test_providers_offline.py` with `respx` to mock complex workflows including Deep Research and Citation Networks (Phase 4).

### Fixed
- **Citation Graph**: Fixed `get_paper_citations` to explicitly request `paperId` fields from Semantic Scholar API, ensuring correct graph connectivity (Phase 1).
- **Error Handling**: Improved retry logic for HTTP 429 (Rate Limit) errors with `Retry-After` header parsing (Phase 1).

## [1.1.0] - 2025-05-20

### Added
- **Deep Research Safeguards**: Implemented `CircuitBreaker` pattern and `max_total_requests` limit in `deep_research.py` to prevent excessive API usage and infinite loops.
- **Offline Testing**: Added `tests/test_providers_offline.py` using `respx` and `pytest-mock` to test search providers without network dependency.
- **API Documentation**: Added `API_SPECS.md` detailing JSON schemas for core tools.
- **Developer Tools**: Added `scripts/pre-commit` for running local checks (formatting, linting, types).
- **Configuration Validation**: Added `validate_config()` to ensure critical environment variables (like `ACADEMIC_CONTACT_EMAIL`) are set on startup.

### Changed
- **Refactoring**: Refactored `research_tools.py` to use a modular `BaseSearchProvider` architecture with specific implementations for `ArxivProvider`, `SemanticScholarProvider`, and `CrossRefProvider`.
- **Deduplication**: Optimized `deduplicate_papers` algorithm with `quick_dedup` (O(N) exact match) and improved fuzzy matching heuristics for better performance on large datasets.
- **Error Handling**: Standardized error handling across the application using custom exceptions (`ProviderError`, `NetworkError`, `RateLimitExceeded`) in `utils.py`.

### Fixed
- **Memory Leak**: Replaced unbounded in-memory cache dictionary with `cachetools.TTLCache` (LRU policy, max 1000 items) in `utils.py` to prevent Out-Of-Memory (OOM) crashes.
- **Cache Access**: Fixed potential `KeyError` when accessing expired items in the cache wrapper.

### Security
- **Config Enforcement**: Enforced `ACADEMIC_CONTACT_EMAIL` configuration to comply with API usage policies and prevent service blocking.
