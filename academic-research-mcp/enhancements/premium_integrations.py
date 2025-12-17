"""
Academic Research MCP Server - Premium Integrations (BYOK)
==========================================================
Modul ini memungkinkan penggunaan API eksternal berbayar
dengan menggunakan API Key milik pengguna sendiri.
"""

import logging
from typing import Dict, Optional, List
import httpx
try:
    from mcp_instance import mcp
except ImportError:
    from ..mcp_instance import mcp

logger = logging.getLogger(__name__)

# Kita hapus semua logika database (sqlite3) dan Tier System

# ============================================================================
# SCOPUS INTEGRATION (BYOK)
# ============================================================================

@mcp.tool()
async def search_scopus(
    query: str,
    max_results: int = 25,
    api_key: str = None  # Wajib diisi oleh user atau via ENV
) -> Dict:
    """
    Mencari database Scopus.
    Wajib menyertakan 'api_key' Elsevier Anda sendiri.
    """
    # 1. Validasi API Key sederhana (tanpa cek database user)
    if not api_key:
        return {
            "error": "API Key Scopus diperlukan. Silakan masukkan argumen 'api_key'.",
            "instruction": "Dapatkan key di https://dev.elsevier.com/"
        }

    logger.info(f"Scopus search: '{query}' using provided API key")

    try:
        # Logika request tetap sama seperti sebelumnya
        url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json",
        }
        params = {"query": query, "count": min(max_results, 100)} # Batas teknis, bukan bisnis

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 401:
                return {"error": "API Key tidak valid atau tidak memiliki izin akses."}

            response.raise_for_status()
            data = response.json()

        # Proses hasil (sama seperti kode asli)
        papers = []
        for entry in data.get("search-results", {}).get("entry", []):
            paper = {
                "source": "Scopus",
                "title": entry.get("dc:title"),
                "authors": entry.get("dc:creator", "").split(", "),
                "year": entry.get("prism:coverDate", "")[:4],
                "citation_count": int(entry.get("citedby-count", 0)),
                "doi": entry.get("prism:doi"),
                "url": entry.get("prism:url"),
            }
            papers.append(paper)

        return {"papers": papers, "count": len(papers)}

    except Exception as e:
        logger.error(f"Scopus search failed: {e}")
        return {"error": str(e)}

# ============================================================================
# WEB OF SCIENCE (BYOK)
# ============================================================================

@mcp.tool()
async def search_web_of_science(
    query: str,
    max_results: int = 25,
    api_key: str = None
) -> Dict:
    """
    Mencari Web of Science. Wajib menyertakan API Key Clarivate Anda.
    """
    if not api_key:
        return {"error": "API Key Web of Science diperlukan."}

    # Implementasi logika request WoS di sini...
    return {
        "status": "Mock Response",
        "message": f"Mencari '{query}' di WoS menggunakan key user.",
        "note": "Implementasi endpoint WoS perlu ditambahkan sesuai dokumentasi Clarivate."
    }
