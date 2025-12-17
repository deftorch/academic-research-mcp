"""
Academic Research MCP Server - COMPLETE PRODUCTION VERSION
==========================================================
All-in-one implementation with full functionality.

⭐ FEATURES:
- 15 powerful research tools (Core)
- 6+ API integrations (arXiv, Semantic Scholar, CrossRef, PubMed, etc.)
- GPU-accelerated PDF extraction with Marker
- Intelligent deduplication & quality assessment
- Complete automated research pipeline
- Citation network analysis
- Batch processing capabilities
- BibTeX export
- RAG & Semantic Search
- Trend Analysis
- Advanced Recommendations

📦 INSTALLATION:
    pip install -r requirements.txt

⚙️  CONFIGURATION:
    1. Change CONTACT_EMAIL in utils.py
    2. Configure Claude Desktop (see Quick Start Guide)
    3. Run: python academic_research_server.py

🚀 USAGE:
    In Claude Desktop:
    "Use research_pipeline to find papers about 'neural networks'
     from 2020-2024 with minimum quality score 6"

📖 DOCUMENTATION:
    See Quick Start Guide artifact for complete setup instructions

Version: 1.0.0 Production
Author: Academic Research MCP Project
License: MIT
"""

import logging

# Import Core Tools (automatically registers them)
from mcp_instance import mcp
from utils import CACHE_TTL, CONTACT_EMAIL, MAX_RETRIES, validate_config

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL MODULE IMPORTS
# ============================================================================

# Import Phase 2: RAG Enhancement (OPTIONAL)
try:
    from enhancements import rag_enhancement

    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG features not available: {e} (install: pip install chromadb)")

# Import Phase 3: Citation Network (OPTIONAL)
try:
    from enhancements import citation_network

    NETWORK_AVAILABLE = True
except ImportError as e:
    NETWORK_AVAILABLE = False
    print(f"⚠️  Citation network not available: {e} (install: pip install networkx plotly python-louvain)")

# Import Phase 4: Trend Analysis (OPTIONAL)
try:
    from enhancements import trend_analysis

    TRENDS_AVAILABLE = True
except ImportError as e:
    TRENDS_AVAILABLE = False
    print(f"⚠️  Trend analysis not available: {e} (install: pip install pandas matplotlib seaborn scipy)")

# Import Phase 5: Advanced Features (OPTIONAL)
try:
    from enhancements import advanced_features

    ADVANCED_AVAILABLE = True
except ImportError as e:
    ADVANCED_AVAILABLE = False
    print(f"⚠️  Advanced features not available: {e} (install: pip install scikit-learn)")

# Import Phase 6: Knowledge Management (OPTIONAL)
try:
    from enhancements import knowledge_management

    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    KNOWLEDGE_AVAILABLE = False
    print(f"⚠️  Knowledge management not available: {e} (install: pip install pyzotero requests)")

# Import Phase 7: Deep Research (OPTIONAL)
try:
    from enhancements import deep_research

    DEEP_RESEARCH_AVAILABLE = True
except ImportError as e:
    DEEP_RESEARCH_AVAILABLE = False
    print(f"⚠️  Deep research not available: {e}")

# Import Premium Integrations (BYOK)
try:
    # Asumsi Anda mengubah nama file jadi premium_integrations.py
    # Jika tetap monetization.py, tidak perlu ubah nama import, cuma isinya saja.
    from enhancements import premium_integrations
    PREMIUM_AVAILABLE = True
except ImportError as e:
    PREMIUM_AVAILABLE = False
    print(f"⚠️  Premium integrations not available: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║    Academic Research MCP Server - COMPLETE EDITION                ║
    ╚═══════════════════════════════════════════════════════════════════╝

    ✅ Phase 1 - Core: 15 tools loaded
    {"✅" if RAG_AVAILABLE else "⚠️ "} Phase 2 - RAG: {"Loaded" if RAG_AVAILABLE else "Not installed"}
    {"✅" if NETWORK_AVAILABLE else "⚠️ "} Phase 3 - Network: {"Loaded" if NETWORK_AVAILABLE else "Not installed"}
    {"✅" if TRENDS_AVAILABLE else "⚠️ "} Phase 4 - Trends: {"Loaded" if TRENDS_AVAILABLE else "Not installed"}
    {"✅" if ADVANCED_AVAILABLE else "⚠️ "} Phase 5 - Advanced: {"Loaded" if ADVANCED_AVAILABLE else "Not installed"}
    {"✅" if KNOWLEDGE_AVAILABLE else "⚠️ "} Phase 6 - Knowledge: {"Loaded" if KNOWLEDGE_AVAILABLE else "Not installed"}
    {"✅" if DEEP_RESEARCH_AVAILABLE else "⚠️ "} Phase 7 - Deep Research: {"Loaded" if DEEP_RESEARCH_AVAILABLE else "Not installed"}

    ⚠️  Configuration:
       • CONTACT_EMAIL: {CONTACT_EMAIL}
       • Cache TTL: {CACHE_TTL}s
       • Max retries: {MAX_RETRIES}

    ℹ️  Note: For full PDF text extraction, ensure 'marker-pdf' is installed.
        pip install marker-pdf (requires GPU/Torch)

    📝 Logs: mcp_server.log

    Starting server...
    """)

    # Validate configuration on startup
    validate_config()

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
